"""In-process jobs with persistent checkpoints supplied by the caller."""

import threading
import time
from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation

from loguru import logger

from app.services.library.store import library_ids


_lock = threading.Lock()
_jobs = {}
_claims = {}
DEFAULT_CALL_BUDGET = 10000
DEFAULT_API_CONCURRENCY = 32


class JobCancelled(RuntimeError):
    pass


def check_cancel(event):
    if event is not None and event.is_set():
        raise JobCancelled("任务已停止，可稍后继续。")


def api_concurrency(value=DEFAULT_API_CONCURRENCY):
    try:
        number = int(value)
        if isinstance(value, bool) or number < 1 or Decimal(str(value)) != number:
            raise ValueError
        return number
    except (TypeError, ValueError, OverflowError, InvalidOperation):
        raise ValueError("API并发数需为正整数。") from None


class AnalysisCoordinator:
    """Shared API admission and cancellation for video and complete-audio analysis."""

    def __init__(self, concurrency=DEFAULT_API_CONCURRENCY, stop=None, before_request=None):
        self.maximum = self.limit = api_concurrency(concurrency)
        self.stop, self.before_callback = stop, before_request
        self.abort = threading.Event()
        self.lock = threading.RLock()
        self.media_slots = threading.BoundedSemaphore(min(4, self.maximum))
        self.calls = self.active = self.done = self.recovered = 0
        self.cooldown = self.reported = 0.0

    def check(self):
        check_cancel(self.stop)
        check_cancel(self.abort)

    def before_request(self):
        while True:
            self.check()
            with self.lock:
                if self.active < self.limit and time.monotonic() >= self.cooldown:
                    if self.before_callback:
                        self.before_callback()
                    self.active += 1
                    self.calls += 1
                    return
            self.abort.wait(.1)

    def after_request(self, details):
        with self.lock:
            self.active = max(0, self.active - 1)
            if details.get("failure_category") == "rate_limit" or details.get("http_status") == 429:
                self.limit = max(1, self.limit // 2)
                self.cooldown = max(self.cooldown, time.monotonic() + max(2, details.get("retry_after", 0)))
                self.recovered = 0
            elif not details.get("failure_category"):
                self.recovered += 1
                if self.recovered >= self.limit * 4 and time.monotonic() >= self.cooldown:
                    self.limit = min(self.maximum, self.limit + 1)
                    self.recovered = 0

    def report(self, force=False):
        raise NotImplementedError


def run_analysis_jobs(coordinator, jobs, callback):
    """Bound pending work and drain in-flight replies before reporting stop or failure."""
    iterator, pending, exhausted, failure = iter(jobs), set(), False, None
    with ThreadPoolExecutor(max_workers=coordinator.maximum, thread_name_prefix="Library") as pool:
        while pending or not exhausted:
            if failure is None:
                try:
                    coordinator.check()
                except JobCancelled as exc:
                    failure = exc
            if failure is not None:
                coordinator.abort.set()
                exhausted = True
                for future in pending:
                    future.cancel()
            else:
                while len(pending) < coordinator.maximum and not exhausted:
                    try:
                        job = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    try:
                        pending.add(pool.submit(callback, job))
                    except Exception as exc:
                        failure = exc
                        coordinator.abort.set()
                        exhausted = True
                        break
            if pending:
                done, _ = wait(pending, timeout=.25, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.remove(future)
                    try:
                        future.result()
                    except CancelledError:
                        pass
                    except Exception as exc:
                        if failure is None:
                            failure = exc
            try:
                coordinator.report()
            except Exception as exc:
                if failure is None:
                    failure = exc
    if failure is not None:
        try:
            coordinator.report(force=True)
        except Exception as exc:
            logger.warning("素材分析终止统计保存失败：{}", exc)
        raise failure


def running(key):
    with _lock:
        job = _jobs.get(key)
        return bool(job and job[0].is_alive())


def library_busy(library_id, *, prefix="library:"):
    with _lock:
        return any(prefix + identifier in _claims for identifier in library_ids(library_id))


@contextmanager
def library_operation(library_id, *, prefix="library:"):
    resources = [prefix + identifier for identifier in library_ids(library_id)]
    owner = threading.current_thread()
    with _lock:
        if any(resource in _claims and _claims[resource][0] is not owner for resource in resources):
            raise ValueError("素材库或关联生成任务正在运行，请等待任务结束后操作。")
        for resource in resources:
            claim = _claims.get(resource)
            _claims[resource] = (owner, (claim[1] if claim else 0) + 1)
    try:
        yield
    finally:
        with _lock:
            for resource in resources:
                claim = _claims.get(resource)
                if claim and claim[0] is owner:
                    if claim[1] > 1:
                        _claims[resource] = (owner, claim[1] - 1)
                    else:
                        del _claims[resource]


def start(key, callback, resources=()):
    with _lock:
        job = _jobs.get(key)
        if job and job[0].is_alive():
            return False
        resources = tuple(dict.fromkeys((*resources, *((key,) if key.startswith("library:") else ()))))
        if any(resource in _claims for resource in resources):
            return False
        stop = threading.Event()
        def work():
            try:
                callback(stop)
            finally:
                with _lock:
                    for resource in resources:
                        if _claims.get(resource, (None,))[0] is threading.current_thread():
                            del _claims[resource]
        thread = threading.Thread(target=work, daemon=True, name="Narrato-" + key[:40])
        _jobs[key] = (thread, stop)
        for resource in resources:
            _claims[resource] = (thread, 1)
        try:
            thread.start()
        except Exception:
            for resource in resources:
                _claims.pop(resource, None)
            _jobs.pop(key, None)
            raise
    return True


def cancel(key):
    with _lock:
        job = _jobs.get(key)
        if job:
            job[1].set()
