"""Local media inspection, cached frames/cuts and cancellable FFmpeg execution."""

import hashlib
import json
import os
import shutil
import subprocess
import queue
import re
import threading
import time
from collections import deque
from uuid import uuid4

from loguru import logger
from fractions import Fraction
from dataclasses import asdict
from pathlib import Path

from app.models.library import VideoInspection
from app.services.library.runner import check_cancel
from app.services.library.store import read_json, write_json


def execute_ffmpeg(command, stop=None, timeout=240, duration=0, progress=None, *, directory=None, stage="ffmpeg", line_callback=None, diagnostic_group="ffmpeg-diagnostics", process_name="FFmpeg", terminate_tree=False):
    """Drain output while allowing cancellation and bounded execution."""
    output = queue.Queue()
    tail = deque(maxlen=40)
    process = reader = None
    record_path = None
    log = None
    record = {"stage": stage, "process": process_name, "state": "running", "started_at": time.time(), "command": list(command), "timeout": timeout}
    if directory:
        record_id = uuid4().hex
        record_path = Path(directory) / diagnostic_group / f"{stage}-{record_id}.json"
        log_path = record_path.with_suffix(".log")
        record.update(id=record_id, log_file=log_path.name)
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
                raise RuntimeError(f"{process_name}超时，阶段文件保留，可以继续生成。")
            try:
                line = output.get(timeout=.5)
            except queue.Empty:
                continue
            if line is None:
                break
            tail.append(line)
            if line_callback:
                line_callback(line)
            if log:
                try:
                    log.write(line + "\n")
                    log.flush()
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


def inspect_video(path, stop=None):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("视频路径不是文件。")
    before = path.stat()
    sample = sampled_fingerprint(path)
    cache_root = Path(__file__).resolve().parents[3] / "storage" / "libraries" / "_media_metadata"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(json.dumps((str(path), before.st_size, before.st_mtime_ns, "full-hash-v2")).encode("utf-8")).hexdigest()
    cache_file = cache_root / (cache_key + ".json")
    if cache_file.is_file():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("sample") == sample:
            return VideoInspection(**cached["media"])
    result = subprocess.run([ffprobe_binary(), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], capture_output=True, timeout=45)
    if result.returncode != 0:
        raise ValueError(f"无法读取视频信息：{path.name}")
    payload = json.loads(result.stdout)
    stream = next((item for item in payload.get("streams", []) if item.get("codec_type") == "video"), None)
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
    media = VideoInspection(str(path), after.st_size, after.st_mtime_ns, fingerprint, duration, int(stream.get("width", 0)), int(stream.get("height", 0)), fps)
    write_json(cache_file, {"sample": sample, "media": asdict(media)})
    return media


def extract_frames(video_path, start, end, count, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(max(1, count)):
        timestamp = start + (end - start) * (index + .5) / max(1, count)
        output = output_dir / f"{timestamp:.3f}.jpg"
        if not output.is_file() or output.stat().st_size == 0:
            temporary = output.with_suffix(".tmp.jpg")
            result = subprocess.run(
                [ffmpeg_binary(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{timestamp:.3f}",
                 "-i", str(video_path), "-frames:v", "1", "-vf", "scale=768:-2", "-q:v", "4", str(temporary)],
                capture_output=True, timeout=90,
            )
            if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f"抽帧失败：{Path(video_path).name} {timestamp:.2f}秒")
            temporary.replace(output)
        frames.append(str(output))
    return frames


def scene_boundaries(asset, cache_file, stop=None, progress=None, diagnostic_directory=None):
    """Cache local scene-change timestamps by source fingerprint, without an API."""
    threshold = .35
    cache_file = Path(cache_file)
    try:
        cached = read_json(cache_file, default=None)
    except (OSError, ValueError):
        cached = None
    if (isinstance(cached, dict) and cached.get("schema") == 1
            and cached.get("fingerprint") == asset["fingerprint"] and cached.get("threshold") == threshold):
        boundaries = cached.get("boundaries")
        if (isinstance(boundaries, list) and len(boundaries) >= 2 and boundaries[0] == 0
                and boundaries[-1] == asset["duration"]
                and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in boundaries)
                and all(0 <= left < right <= asset["duration"] for left, right in zip(boundaries, boundaries[1:]))):
            return boundaries
    cuts = []

    def collect(line):
        match = re.search(r"pts_time:([0-9]+(?:\.[0-9]+)?)", line)
        if match:
            point = float(match.group(1))
            if 0 < point < asset["duration"]:
                cuts.append(point)

    command = [ffmpeg_binary(), "-nostdin", "-hide_banner", "-nostats", "-progress", "pipe:1",
               "-i", asset["path"], "-an", "-sn", "-vf",
               f"setpts=PTS-STARTPTS,scale=320:-2,select='gt(scene,{threshold})',showinfo",
               "-vsync", "vfr", "-f", "null", "-"]
    code, output = execute_ffmpeg(command, stop, timeout=max(300, asset["duration"] * 2),
                                  duration=asset["duration"], progress=progress, line_callback=collect,
                                  directory=diagnostic_directory, stage="scene-scan-" + asset["id"])
    if code != 0:
        raise RuntimeError(f"原片镜头检测失败（FFmpeg 返回码 {code}）。\n{output[-1500:]}")
    boundaries = [0.0, *sorted(set(cuts)), asset["duration"]]
    write_json(cache_file, {"schema": 1, "fingerprint": asset["fingerprint"], "threshold": threshold,
                            "boundaries": boundaries})
    return boundaries
