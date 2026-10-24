"""Local media inspection, cached frames/cuts and cancellable FFmpeg execution."""

import hashlib
import math
import json
import os
import shutil
import subprocess
import queue
import re
import threading
import time
from bisect import bisect_left
from collections import Counter, deque
from functools import lru_cache
from tempfile import TemporaryDirectory
from uuid import uuid4

from loguru import logger
from fractions import Fraction
from dataclasses import asdict
from pathlib import Path

from app.models.library import FRAME_PROTOCOL, SourceFrameIndex, VideoInspection, media_id, validate_source_frames
from app.services.library.runner import JobCancelled, check_cancel
from app.services.library.store import read_json, write_json, stable_hash, _retry_json_io


def execute_ffmpeg(command, stop=None, timeout=240, duration=0, progress=None, *, directory=None, stage="ffmpeg", line_callback=None, diagnostic_group="ffmpeg-diagnostics", process_name="FFmpeg", terminate_tree=False, diagnostic=None):
    """Drain output while allowing cancellation and bounded execution."""
    output = queue.Queue()
    tail = deque(maxlen=40)
    process = reader = None
    record_path = None
    log = None
    record = diagnostic if diagnostic is not None else {}
    record.update(stage=stage, process=process_name, state="running", started_at=time.time(), command=list(command), timeout=timeout)
    if directory:
        record_id = uuid4().hex
        record_path = Path(directory) / diagnostic_group / f"{stage}-{record_id}.json"
        log_path = record_path.with_suffix(".log")
        record.update(id=record_id, log_file=log_path.name, diagnostic_file=str(record_path))
        try:
            write_json(record_path, record)
            log = Path(log_path).open("w", encoding="utf-8")
        except OSError as exc:
            logger.warning("FFmpeg 执行记录创建失败：{}", exc)
    deadline = time.monotonic() + timeout
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")

        def read():
            for line in process.stdout:
                output.put(line.strip())
            output.put(None)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        while True:
            check_cancel(stop)
            if time.monotonic() > deadline:
                record["reason"] = "timeout"
                raise RuntimeError(f"{process_name}超时，阶段文件保留，可以继续生成。")
            try:
                line = output.get(timeout=.5)
            except queue.Empty:
                continue
            if line is None:
                break
            tail.append(line)
            consumed = line_callback(line) is True if line_callback else False
            if log and not consumed:
                try:
                    log.write(line + "\n")
                except OSError as exc:
                    logger.warning("FFmpeg 输出日志写入失败：{}", exc)
                    try:
                        log.close()
                    except OSError:
                        pass
                    log = None
            if progress and duration > 0 and line.startswith("out_time_us="):
                try:
                    progress(min(1, int(line.split("=", 1)[1]) / 1000000 / duration))
                except ValueError:
                    pass
        return_code = process.wait()
        record.update(state="complete" if return_code == 0 else "failed", return_code=return_code)
        return return_code, "\n".join(tail)
    except Exception as exc:
        record.update(state="failed", error=str(exc))
        if isinstance(exc, JobCancelled):
            record["reason"] = "cancelled"
        raise
    finally:
        if process is not None and process.poll() is None:
            if terminate_tree and os.name == "nt":
                try:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if reader is not None:
            reader.join(timeout=2)
        if process is not None and process.stdout:
            process.stdout.close()
        if log:
            try:
                log.close()
            except OSError as exc:
                logger.warning("FFmpeg 输出日志关闭失败：{}", exc)
        if record_path:
            record.update(finished_at=time.time(), output_tail="\n".join(tail))
            try:
                write_json(record_path, record)
            except OSError as exc:
                logger.warning("FFmpeg 执行记录保存失败：{}", exc)



def ffmpeg_binary():
    executable = os.environ.get("IMAGEIO_FFMPEG_EXE") or shutil.which("ffmpeg")
    if not executable:
        raise ValueError("未找到 FFmpeg，请在系统设置中选择引擎。")
    return executable


def ffprobe_binary():
    sibling = Path(ffmpeg_binary()).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
    executable = str(sibling) if sibling.is_file() else shutil.which("ffprobe")
    if not executable:
        raise ValueError("未找到 FFprobe，请安装带 FFprobe 的 FFmpeg。")
    return executable


def sampled_fingerprint(path):
    path = Path(path)
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode("ascii"))
    with path.open("rb") as source:
        for fraction in (0, .25, .5, .75, 1):
            source.seek(max(0, int((size - 65536) * fraction)))
            digest.update(source.read(65536))
    return digest.hexdigest()


def inspect_video(path, stop=None, *, cache_directory=None):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("视频路径不是文件。")
    before = path.stat()
    sample = sampled_fingerprint(path)
    shared_cache = Path(__file__).resolve().parents[3] / "storage" / "libraries" / "_media_metadata"
    cache_root = Path(cache_directory) if cache_directory is not None else shared_cache
    cache_key = hashlib.sha256(json.dumps((str(path), before.st_size, before.st_mtime_ns, "source-stream-v3")).encode("utf-8")).hexdigest()
    cache_file = cache_root / (cache_key + ".json")
    for candidate in dict.fromkeys((cache_file, shared_cache / cache_file.name)):
        try:
            cached = json.loads(candidate.read_text(encoding="utf-8"))
            media = VideoInspection(**cached["media"])
            if cached.get("sample") == sample and (media.path, media.size, media.mtime_ns) == (str(path), before.st_size, before.st_mtime_ns):
                return media
        except (OSError, ValueError, TypeError, KeyError):
            pass
    result = subprocess.run([ffprobe_binary(), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], capture_output=True, timeout=45)
    if result.returncode != 0:
        raise ValueError(f"无法读取视频信息：{path.name}")
    payload = json.loads(result.stdout)
    stream = next((item for item in payload.get("streams", []) if item.get("codec_type") == "video"
                   and not item.get("disposition", {}).get("attached_pic")), None)
    if not stream:
        raise ValueError(f"文件没有视频流：{path.name}")
    duration = float(payload.get("format", {}).get("duration") or 0)
    try:
        video_duration = float(stream.get("duration") or 0)
    except (ValueError, TypeError):
        video_duration = 0
    if video_duration <= 0:
        tagged_duration = stream.get("tags", {}).get("DURATION", "")
        if tagged_duration and ":" in tagged_duration:
            try:
                hours, minutes, seconds = tagged_duration.split(":")
                video_duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            except (ValueError, TypeError):
                pass
    if video_duration > 0:
        duration = video_duration
    if duration <= 0:
        raise ValueError(f"无法确定视频时长：{path.name}")
    if int(stream.get("width", 0)) <= 0 or int(stream.get("height", 0)) <= 0:
        raise ValueError(f"视频分辨率无效：{path.name}")
    try:
        fps = float(Fraction(stream.get("avg_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError):
        fps = 0
    from app.services.library.runner import check_cancel
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while data := source.read(8 * 1024 * 1024):
            check_cancel(stop)
            digest.update(data)
    fingerprint = digest.hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("视频正在变化，请等待写入完成再导入。")
    media = VideoInspection(str(path), after.st_size, after.st_mtime_ns, fingerprint, duration,
                            int(stream.get("width", 0)), int(stream.get("height", 0)), fps, int(stream["index"]))
    write_json(cache_file, {"sample": sample, "media": asdict(media)})
    return media


@lru_cache(maxsize=8)
def _engine_identity(path, size, modified):
    result = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=10)
    if result.returncode or not result.stdout:
        raise ValueError("无法读取FFmpeg版本。")
    return {"version": result.stdout.splitlines()[0], "size": size, "mtime_ns": modified,
            "scan": "showinfo-source/scale320/select-all-scene/v1", "threshold": .35}


@lru_cache(maxsize=8)
def _read_source_index(path, modified, size):
    data = read_json(path)
    if not isinstance(data, dict) or data.get("key") != stable_hash({k: v for k, v in data.items() if k != "key"}):
        raise ValueError("源帧索引摘要不一致，请重新建立索引。")
    return SourceFrameIndex(data)


def _probe_tail(asset, data, root, stop, method, record=None):
    """Consume compact metadata; retain timestamps/counts, not the packet list."""
    packet = method == "packets"
    kind = "packet" if packet else "frame"
    clock = Fraction(*data["time_base"])
    target = data["pts"][-1]*clock
    position = max(0.0, float(target)-30)
    record = record if record is not None else {}
    record.update(method=method, asset_id=asset["id"], episode=asset.get("episode"),
                  fingerprint=asset["fingerprint"], stream_index=asset["stream_index"],
                  target_pts=data["pts"][-1], target_time_base=data["time_base"],
                  seek_start=None if packet else position, reason="invalid_result")
    counts, streams, largest, matches, messages = Counter(), [], None, [], deque(maxlen=10)
    item_count = 0

    def collect(line):
        nonlocal largest, matches, item_count
        section, _, content = line.partition("|")
        if section not in {kind, "stream"}:
            if line:
                messages.append(line)
            return False
        row = dict(part.split("=", 1) for part in content.split("|") if "=" in part)
        if section == "stream":
            if len(streams) >= 2:
                raise ValueError("探测返回多个视频流。")
            streams.append(row)
            return True
        if int(row["stream_index"]) != asset["stream_index"]:
            raise ValueError("探测视频流与扫描不一致。")
        pts = row.get("pts")
        if pts in (None, "N/A") and not packet:
            pts = row.get("best_effort_timestamp")
        pts = int(pts)
        if packet and (int(row["size"]) <= 0 or any(flag in row["flags"] for flag in "CD")):
            raise ValueError("包含空数据、损坏或丢弃标记，无法确认包与帧的对应关系。")
        counts[pts] += 1
        if item_count >= data["frame_count"]:
            raise ValueError("探测条目多于完整扫描帧数，无法确认对应关系。")
        item_count += 1
        if largest is None or pts > largest:
            largest, matches = pts, [row]
        elif pts == largest and len(matches) < 2:
            matches.append(row)
        return True

    fields = ("packet=stream_index,pts,dts,duration,size,pos,flags" if packet else
              "frame=stream_index,pts,best_effort_timestamp,duration,pkt_duration")
    command = [ffprobe_binary(), "-v", "error", "-select_streams", str(asset["stream_index"])]
    if not packet:
        command += ["-read_intervals", f"{position:.9f}%"]
    command += ["-show_packets" if packet else "-show_frames", "-show_streams", "-show_entries",
                fields+":stream=index,codec_type,codec_name,time_base", "-of", "compact=p=1:nk=0", asset["path"]]
    try:
        code, _ = execute_ffmpeg(command, stop, timeout=max(120, float(asset.get("duration", 0))*2) if packet else 90,
            directory=root, stage="frame-tail-"+asset["id"]+"-"+method, line_callback=collect,
            process_name="FFprobe", diagnostic=record)
        if code or messages:
            record["reason"] = "probe_error"
        else:
            if (len(streams) != 1 or int(streams[0]["index"]) != asset["stream_index"] or
                    streams[0]["codec_type"] != "video"):
                raise ValueError("尾帧探测的视频流身份无效。")
            probe_clock = Fraction(streams[0]["time_base"])
            if probe_clock <= 0:
                raise ValueError("尾帧探测时间基无效。")
            record.update(probe_time_base=str(probe_clock), codec=streams[0].get("codec_name"))
            raw_target = target/probe_clock
            count = counts.get(int(raw_target), 0) if raw_target.denominator == 1 else 0
            record.update(match_count=count, matched_items=matches if largest == raw_target else [])
            if count == 0:
                record["reason"] = "empty_result" if not item_count else "target_not_found"
            elif count > 1:
                record["reason"] = "ambiguous_target"
            elif largest != raw_target:
                raise ValueError("探测返回的显示范围超出完整扫描。")
            elif packet and Counter({p*probe_clock: n for p, n in counts.items()}) != Counter(p*clock for p in data["pts"]):
                record["reason"] = "correspondence_unconfirmed"
            else:
                values = {int(matches[0][k]) for k in ("duration", "pkt_duration")
                          if matches[0].get(k) not in (None, "N/A", "0")}
                duration = next(iter(values), 0)*probe_clock/clock
                if len(values) > 1:
                    record["reason"] = "ambiguous_target"
                elif duration <= 0 or duration.denominator != 1:
                    record["reason"] = "duration_missing"
                else:
                    record.update(reason="confirmed", duration_ticks=int(duration),
                                  correspondence="full_pts_bijection" if packet else "decoded_frame")
    except JobCancelled:
        record["reason"] = "cancelled"
        raise
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
        record.update(error=str(exc))
        if isinstance(exc, OSError):
            record["reason"] = "probe_error"
    finally:
        record.update(item_count=item_count, messages=list(messages), validation="complete" if record.get("duration_ticks") else "failed")
        if record.get("diagnostic_file"):
            write_json(record["diagnostic_file"], record)
    return record


def _tail_duration(asset, data, root, stop, progress, record):
    """One local attempt and one eligible sequential fallback, with exact clocks."""
    if data["pts"].count(data["pts"][-1]) != 1:
        record["reason"] = "ambiguous_target"
        raise ValueError("扫描尾帧存在重复PTS，不能唯一确认尾帧时长。")
    record["probes"] = []
    for method in ("frames", "packets"):
        check_cancel(stop)
        if progress:
            progress(1, "局部探测未取得尾帧，正在顺序读取包信息" if method == "packets" else "源帧扫描已完成，正在确认尾帧时长")
        probe = {}
        try:
            _probe_tail(asset, data, root, stop, method, probe)
        finally:
            record["probes"].append({k: probe[k] for k in ("method", "reason", "diagnostic_file") if k in probe})
        record["reason"] = probe["reason"]
        if probe.get("duration_ticks"):
            return probe["duration_ticks"]
        if probe["reason"] not in {"empty_result", "target_not_found", "duration_missing"}:
            break
    reasons = {"empty_result": "探测未返回视频条目", "target_not_found": "探测未找到目标PTS",
               "duration_missing": "目标缺少可用时长", "ambiguous_target": "目标PTS或时长存在歧义",
               "correspondence_unconfirmed": "包PTS与完整显示帧表不能一一对应",
               "timeout": "尾帧探测超时", "probe_error": "尾帧读取失败", "invalid_result": "尾帧探测格式或来源无效"}
    raise ValueError(reasons.get(record["reason"], record["reason"]))


def source_index(asset, root, stop=None, *, prepare=False, progress=None):
    """Read the single current index; only explicit preparation may scan media."""
    check_cancel(stop)
    if not re.fullmatch(r"[a-f0-9]{64}", asset["fingerprint"]):
        raise ValueError("源视频指纹无效。")
    target = Path(root)/"frame_index"/(asset["fingerprint"]+".json")
    engine_path = Path(ffmpeg_binary()).resolve()
    stat = engine_path.stat()
    engine = _engine_identity(str(engine_path), stat.st_size, stat.st_mtime_ns)
    try:
        stat = target.stat()
        index = _read_source_index(str(target), stat.st_mtime_ns, stat.st_size)
        if index.data["fingerprint"] != asset["fingerprint"] or index.data["engine"] != engine:
            raise ValueError("源帧索引与视频或解码引擎不一致。")
        if "index_key" in asset and index.data["key"] != asset["index_key"]:
            raise ValueError("源帧索引已更新，请重新拉片或创建Agent任务。")
        if progress:
            progress(1, "复用源帧索引")
        return index
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        if not prepare:
            raise ValueError("完整源帧索引缺失或已失效，请在视频素材库重新准备；旧解析不能用于2.9.1。") from None
    actual = inspect_video(asset["path"], stop)
    if actual.fingerprint != asset["fingerprint"]:
        raise ValueError("源视频已变化，请重新登记。")
    media = {**asset, "stream_index": actual.stream_index}
    pending = target.with_suffix(".scan.json")
    diagnostic = Path(root)/"ffmpeg-diagnostics"/f"source-index-{asset['id']}-{uuid4().hex}.json"
    record = {"stage": "source-index-publish", "asset_id": asset["id"], "episode": asset.get("episode"),
              "task_id": asset.get("agent_task_id"), "library_id": Path(root).name,
              "fingerprint": asset["fingerprint"], "stream_index": actual.stream_index,
              "started_at": time.time(), "state": "running", "checkpoint": str(pending)}
    try:
        data = None
        try:
            saved = read_json(pending)
            if (saved.get("version") != 1 or saved.get("phase") != "tail_pending" or
                    saved.get("key") != stable_hash({k: v for k, v in saved.items() if k != "key"}) or
                    saved.get("source") != asdict(actual) or saved["scan"].get("engine") != engine or
                    set(saved["scan"]) != {"protocol", "fingerprint", "stream_index", "time_base", "origin_pts",
                                          "frame_count", "pts", "durations", "cuts", "engine", "complete"} or
                    saved["scan"].get("fingerprint") != actual.fingerprint or
                    saved["scan"].get("stream_index") != actual.stream_index or saved["scan"].get("complete") is not False or
                    saved["completed"].get("return_code") != 0 or
                    saved["completed"].get("score_count") != saved["scan"].get("frame_count")):
                raise ValueError("完整扫描检查点与来源、引擎或摘要不一致。")
            validate_source_frames(saved["scan"], require_tail=False)
            data = saved["scan"]
            record["scan"] = "reused"
            if progress:
                progress(1, "复用完整扫描，继续确认尾帧时长")
        except FileNotFoundError:
            record["scan"] = "checkpoint_missing"
        except PermissionError:
            raise
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
            record.update(scan="checkpoint_invalid", checkpoint_error=str(exc))
        write_json(diagnostic, record)
        if data is None:
            check_cancel(stop)
            if progress:
                progress(0, "无有效完整扫描检查点，开始建立源帧索引")
            data, scan_record = _scan_source(media, actual, engine, root, stop, progress)
            validate_source_frames(data, require_tail=False)
            if asdict(inspect_video(asset["path"], stop)) != asdict(actual):
                raise ValueError("扫描期间源视频发生变化，未保存完整检查点。")
            saved = {"version": 1, "phase": "tail_pending", "source": asdict(actual), "scan": data,
                     "completed": {"return_code": 0, "score_count": data["frame_count"],
                                   "diagnostic_file": scan_record.get("diagnostic_file")}}
            saved["key"] = stable_hash(saved)
            write_json(pending, saved, compact=True)
            record["scan"] = "saved"
        record.update(target_pts=data["pts"][-1], time_base=data["time_base"], frame_count=data["frame_count"])
        check_cancel(stop)
        if data["durations"][-1] <= 0:
            data["durations"][-1] = _tail_duration(media, data, root, stop, progress, record)
        data["complete"] = True
        data["key"] = stable_hash(data)
        index = SourceFrameIndex(data)
        check_cancel(stop)
        if asdict(inspect_video(asset["path"], stop)) != asdict(actual):
            record["scan"] = "source_changed"
            raise ValueError("尾帧确认期间源视频发生变化，未发布索引。")
        check_cancel(stop)
        write_json(target, data, compact=True)
        record.update(state="complete", reason="confirmed", index_file=str(target), index_key=data["key"])
        try:
            pending.unlink(missing_ok=True)
        except OSError as exc:
            record["checkpoint_cleanup_error"] = str(exc)
        if progress:
            progress(1, "源帧索引已完成")
        return index
    except Exception as exc:
        record.update(state="failed", error=str(exc))
        if isinstance(exc, JobCancelled):
            record["reason"] = "cancelled"
        retained = pending.is_file() and record.get("scan") in {"saved", "reused"}
        message = ("；完整扫描已保存，继续准备仅补尾帧确认" if retained else "；尚无可恢复的完整扫描")
        if progress:
            progress(1 if retained else 0, str(exc)+message)
        error = JobCancelled if isinstance(exc, JobCancelled) else ValueError
        raise error(f"{exc}{message}。诊断：{diagnostic}") from exc
    finally:
        record["finished_at"] = time.time()
        try:
            write_json(diagnostic, record)
        except OSError as exc:
            if record["state"] == "complete":
                raise
            logger.warning("源帧索引失败记录保存异常：{}；原始错误：{}", exc, record.get("error"))


def _scan_source(asset, actual, engine, root, stop, progress):
    """Return only a complete scan; tail confirmation and publication are separate."""
    pts, durations, scores = [], [], []
    clock, metadata_frame = None, None

    def collect(line):
        nonlocal clock, metadata_frame
        if "showinfo@source" in line:
            match = re.search(r"config in time_base:\s*(\d+)/(\d+)", line)
            if match:
                found = [int(match[1]), int(match[2])]
                if clock is not None and clock != found:
                    raise ValueError("视频中途改变时间基，未生成源帧索引。")
                clock = found
            match = re.search(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)", line)
            if match:
                if int(match[1]) != len(pts) or not clock:
                    raise ValueError("源帧序列不连续或缺少时间基。")
                pts.append(int(match[2]))
                duration = re.search(r"\bduration:\s*(\d+)", line)
                durations.append(int(duration[1]) if duration else 0)
            elif re.search(r"\bn:\s*\d+\s+pts:", line):
                raise ValueError("源帧缺少有效PTS，未生成索引。")
            return True
        match = re.search(r"\bframe:\s*(\d+)\s+pts:\s*(-?\d+)", line)
        if match:
            metadata_frame = int(match[1])
            if metadata_frame != len(scores) or metadata_frame >= len(pts) or pts[metadata_frame] != int(match[2]):
                raise ValueError("场景分数与源帧序列无法一一对应。")
            return True
        match = re.search(r"lavfi\.scene_score=([0-9.eE+\-]+)", line)
        if match:
            if metadata_frame != len(scores):
                raise ValueError("场景检测分数顺序无效。")
            score = float(match[1])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("场景检测分数无效。")
            scores.append(score)
            return True
        return False

    command = [ffmpeg_binary(), "-nostdin", "-hide_banner", "-nostats", "-xerror", "-copyts",
        "-progress", "pipe:1", "-i", asset["path"], "-map", f"0:{actual.stream_index}", "-an", "-sn", "-dn",
        "-vf", "showinfo@source=checksum=0,scale=320:-2,select='gte(scene,0)',metadata=mode=print:key=lavfi.scene_score",
        "-vsync", "0", "-f", "null", "-"]
    scan_record = {}
    code, tail = execute_ffmpeg(command, stop, timeout=max(300, actual.duration*2), duration=actual.duration,
        line_callback=collect, progress=(lambda p: progress(p, "建立源帧索引")) if progress else None,
        directory=root, stage="frame-index-"+asset["id"], diagnostic=scan_record)
    if code or not pts or len(scores) != len(pts):
        raise ValueError("完整源帧扫描失败，未发布索引：" + tail[-1000:])
    cuts = [0]
    for number, score in enumerate(scores):
        if number and score > .35 and pts[number] > pts[cuts[-1]]:
            cuts.append(bisect_left(pts, pts[number]))
    data = {"protocol": FRAME_PROTOCOL, "fingerprint": asset["fingerprint"], "stream_index": actual.stream_index,
            "time_base": clock, "origin_pts": pts[0], "frame_count": len(pts), "pts": pts, "durations": durations,
            "cuts": [*cuts, len(pts)], "engine": engine, "complete": False}
    return data, scan_record


def publish_image(source, target):
    """Publish through a sibling file so private staging-directory ACLs are not retained."""
    target = Path(target)
    temporary = target.with_name(target.name + "." + uuid4().hex + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        _retry_json_io(lambda: temporary.replace(target))
    finally:
        temporary.unlink(missing_ok=True)


def extract_frames(asset, index, numbers, output_dir, *, crop=None, stop=None):
    """Batch exact source frames; seek is an optimization, never their identity."""
    from bisect import bisect_left, bisect_right
    if index.data["fingerprint"] != asset["fingerprint"]:
        raise ValueError("取帧索引与源视频不一致。")
    if crop is not None and (not isinstance(crop, (list, tuple)) or len(crop) != 4 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in crop) or
            not (0 <= crop[0] < crop[2] <= 1 and 0 <= crop[1] < crop[3] <= 1)):
        raise ValueError("裁剪须为归一化坐标[x1,y1,x2,y2]，范围0至1。")
    numbers = sorted(set(numbers))
    if not numbers:
        return []
    if len(numbers) > 100:
        raise ValueError("一次最多获取100个证据帧，请按范围分批。")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    saved, pending = {}, []
    for number in numbers:
        check_cancel(stop)
        frame = index.frame(number)
        key = stable_hash((index.data["key"], frame["id"], crop, "jpeg768-v1"))
        cache = root/(key+".json")
        value = read_json(cache, default={})
        if (isinstance(value, dict) and all(value.get(k) == v for k, v in
                {**index.binding(), **frame, "fingerprint": asset["fingerprint"]}.items())
                and value.get("crop") == (list(crop) if crop is not None else None)
                and re.fullmatch(r"I[0-9a-f]{20}\.jpg", str(value.get("file", "")))):
            path = root/value["file"]
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == value.get("sha256"):
                saved[number] = {**value, "path": str(path)}
                continue
        pending.append((number, frame, cache))
    if not pending:
        return [saved[n] for n in numbers]
    target_pts = [index.pts[n] for n, _, _ in pending]
    unique_pts = target_pts[0] > 0 and all(bisect_right(index.pts, p)-bisect_left(index.pts, p) == 1 for p in target_pts)
    with TemporaryDirectory(prefix="capture-", dir=root) as temporary:
        output = Path(temporary)
        for seek in ([True, False] if unique_pts else [False]):
            check_cancel(stop)
            clocks, decoded = [], []
            def collect(line):
                match = re.search(r"config in time_base:\s*(\d+)/(\d+)", line)
                if "showinfo" in line and match:
                    clocks.append(Fraction(int(match[1]), int(match[2])))
                match = re.search(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)", line)
                if "showinfo" in line and match and clocks:
                    decoded.append((int(match[1]), int(match[2])*clocks[-1]))
                return "showinfo" in line
            expression = "+".join(f"eq({'pts' if seek else 'n'},{p if seek else n})"
                                  for (n, _, _), p in zip(pending, target_pts))
            filters = [f"select='{expression}'", "showinfo@evidence=checksum=0"]
            if crop is not None:
                x1, y1, x2, y2 = crop
                filters.append(f"crop=iw*{x2-x1}:ih*{y2-y1}:iw*{x1}:ih*{y1}")
            filters.append("scale=768:-2")
            command = [ffmpeg_binary(), "-nostdin", "-hide_banner", "-nostats", "-xerror", "-y", "-copyts"]
            if seek:
                position = float((target_pts[0]-1)*index.clock)
                command += ["-seek_timestamp", "1", "-ss", f"{position:.12f}"]
            command += ["-i", asset["path"], "-map", f"0:{index.data['stream_index']}", "-an", "-sn", "-dn",
                        "-vf", ",".join(filters), "-frames:v", str(len(pending)), "-vsync", "0", "-q:v", "4",
                        str(output/"%06d.jpg")]
            timeout = max(120, index.time(pending[-1][0])*2) if not seek else max(
                120, (index.time(pending[-1][0])-index.time(pending[0][0]))*2+30)
            code, tail = execute_ffmpeg(command, stop, timeout=timeout,
                                         line_callback=collect)
            expected = [(i, p*index.clock) for i, p in enumerate(target_pts)]
            # FFmpeg can request one additional filtered frame before closing the image muxer.
            if not code and decoded[:len(expected)] == expected and all((output/f"{i+1:06d}.jpg").is_file() for i in range(len(pending))):
                break
        else:
            raise ValueError("未取得与源帧索引一致的画面；没有用邻近帧替代："+tail[-700:])
        for i, (number, frame, cache) in enumerate(pending):
            path = output/f"{i+1:06d}.jpg"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            iid = media_id("I", FRAME_PROTOCOL, frame["id"], crop, digest)
            target = root/(iid+".jpg")
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError("相同图片编号存在不同内容，未覆盖已有证据。")
            if not target.exists():
                publish_image(path, target)
            value = {**frame, **index.binding(), "image_id": iid, "crop": list(crop) if crop is not None else None,
                     "file": target.name, "sha256": digest, "fingerprint": asset["fingerprint"]}
            write_json(cache, value)
            saved[number] = {**value, "path": str(target)}
    return [saved[n] for n in numbers]


def evidence_frame_number(index, *, timestamp=None, frame_index=None, bounds=None):
    start, end = bounds or (0, index.count)
    if not 0 <= start < end <= index.count or (timestamp is None) == (frame_index is None):
        raise ValueError("请在当前帧范围内指定秒数或帧号，两者只能选一个。")
    if timestamp is not None:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or not index.time(start) <= timestamp < index.time(end):
            raise ValueError("取帧时间须在当前镜头范围内。")
        frame_index = index.nearest(timestamp, start, end)
    if type(frame_index) is not int or not start <= frame_index < end:
        raise ValueError("取帧序号须在当前镜头半开区间内。")
    return frame_index


def extract_evidence_frame(asset, index, output_dir, *, timestamp=None, frame_index=None, bounds=None, crop=None, stop=None):
    number = evidence_frame_number(index, timestamp=timestamp, frame_index=frame_index, bounds=bounds)
    return {**extract_frames(asset, index, [number], output_dir, crop=crop, stop=stop)[0],
            "requested_time": float(timestamp) if timestamp is not None else None}



def frame_signatures(paths):
    """Similarity hints only; repeated footage still requires human confirmation."""
    from PIL import Image
    signatures = []
    for path in paths:
        with Image.open(path) as image:
            values = list(image.convert("L").resize((9, 8)).getdata())
        if max(values)-min(values) < 24:
            continue
        bits = [values[y*9+x] > values[y*9+x+1] for y in range(8) for x in range(8)]
        signatures.append(f"{sum(int(bit)<<i for i,bit in enumerate(bits)):016x}")
    return signatures
