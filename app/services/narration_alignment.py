"""Cached local alignment and original-text timing; the UI process imports no torch."""

import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import threading
import time
import unicodedata
from uuid import uuid4
import wave

from app.services.library.frame_extractor import execute_ffmpeg, ffmpeg_binary, sampled_fingerprint
from app.services.library.runner import check_cancel
from app.services.library.store import read_json, stable_hash, write_json

LEGACY_MODE = "legacy"
LONG_MODE = "long_align"
_GPU_LOCK = threading.Lock()
_WORKER = Path(__file__).with_name("narration_alignment_worker.py")


def default_alignment_options():
    workspace = Path(__file__).resolve().parents[3]
    return {"alignment_python": str(workspace / ".conda" / "narratoai-align" / "python.exe"),
            "alignment_model": str(workspace / "models" / "Qwen3-ForcedAligner-0.6B-hf"),
            "alignment_language": "Chinese"}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def validate_alignment_options(options):
    settings = {**default_alignment_options(), **{key: value for key, value in options.items()
                if key in default_alignment_options()}}
    python, model = Path(settings["alignment_python"]), Path(settings["alignment_model"])
    if not python.is_file():
        raise ValueError("本地对齐环境Python不存在，请检查高级设置中的路径。")
    filenames = ("config.json", "processor_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    missing = [name for name in filenames if not (model / name).is_file()]
    if missing:
        raise ValueError("对齐模型文件不完整：" + "、".join(missing))
    config = read_json(model / "config.json")
    if "Qwen3ASRForTokenClassification" not in config.get("architectures", []):
        raise ValueError("请选择Qwen3-ForcedAligner的HF原生模型目录。")
    runtime = {}
    for name in ("torch", "transformers", "numpy", "tokenizers", "safetensors"):
        metadata = list((python.parent / "Lib" / "site-packages").glob(name + "-*.dist-info/METADATA"))
        if not metadata:
            raise ValueError(f"对齐环境缺少{name}依赖。")
        runtime[name] = next(line[9:] for line in metadata[0].read_text(encoding="utf-8").splitlines() if line.startswith("Version: "))
    if settings["alignment_language"] not in {"Chinese", "English", "Cantonese", "French", "German", "Italian", "Portuguese", "Russian", "Spanish"}:
        raise ValueError("当前对齐环境未配置所选语言。")
    fingerprints = [(name, (model / name).stat().st_size, (model / name).stat().st_mtime_ns,
                     sampled_fingerprint(model / name)) for name in filenames]
    settings.update(alignment_python=str(python.resolve()), alignment_model=str(model.resolve()))
    signature = stable_hash(("qwen-hf-align-v1", fingerprints, runtime, file_hash(_WORKER),
                             settings["alignment_language"], "cuda:0", "bfloat16", "sdpa"))
    return settings, signature


def _canonical(text):
    return "".join(char for char in text if char == "'" or unicodedata.category(char)[0] in {"L", "N"})


def _validate_words(words, text, seconds):
    if not isinstance(words, list) or not words:
        raise ValueError("对齐结果没有字词时间。")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("对齐音频时长不是有效正数。")
    char_end, previous_end = 0, 0.0
    for index, word in enumerate(words):
        if not isinstance(word, dict):
            raise ValueError("对齐字词记录格式无效。")
        a, b = word.get("text_start"), word.get("text_end")
        if type(a) is not int or type(b) is not int or not char_end <= a < b <= len(text):
            raise ValueError("对齐结果原文位置无效。")
        if _canonical(text[a:b]) != word.get("text"):
            raise ValueError("对齐结果文字与原文不一致。")
        try:
            start, end = float(word["start_time"]), float(word["end_time"])
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError(f"第{index+1}个字词（原文位置{a}–{b}）缺少有效对齐时间。") from None
        # The 80ms decoder does not know the WAV duration. Bound only its final
        # endpoint to two ticks when the last spoken word starts inside the audio.
        tail = index == len(words) - 1 and 0 <= start < seconds and end <= seconds + .161
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end
                and (end <= seconds + .081 or tail)):
            raise ValueError(f"对齐时间超出音频范围或不是有效数字：第{index+1}个字词（原文位置{a}–{b}）"
                             f"为{start:.3f}–{end:.3f}秒，音频长{seconds:.3f}秒。")
        if start < previous_end - .001:
            raise ValueError("对齐字词时间出现重叠或倒序。")
        word.update(start_time=min(start, seconds), end_time=min(end, seconds))
        char_end, previous_end = b, word["end_time"]
    if "".join(word["text"] for word in words) != _canonical(text):
        raise ValueError("对齐字词没有完整覆盖原文。")
    if not any(word["end_time"] > word["start_time"] for word in words):
        raise ValueError("对齐结果全部为零时长。")
    return words


class _AlignmentInputLimit(RuntimeError):
    def __init__(self, message, block_id, encoded_tokens, limit):
        super().__init__(message)
        self.block_id, self.encoded_tokens, self.limit = block_id, encoded_tokens, limit


def _pcm_frames(path):
    try:
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 24000) and Path(path).stat().st_size >= frames * 2 + 44:
                return frames
    except (OSError, EOFError, wave.Error):
        pass
    return 0


def _coarse_block(block, digest, target_seconds, directory, stop):
    """Pitch-preserving acceleration supplies anchors, never final subtitle timing."""
    path = directory / "alignment" / "windows" / (stable_hash(("alignment-coarse-v1", digest, target_seconds)) + ".wav")
    frames = _pcm_frames(path)
    if not frames or abs(frames / 24000 - target_seconds) > .5:
        path.parent.mkdir(parents=True, exist_ok=True)
        factor = block["samples"] / 24000 / target_seconds
        filters = []
        while factor > 2:
            filters.append("atempo=2")
            factor /= 2
        filters.append(f"atempo={factor:.10f}")
        temporary = path.with_suffix(".tmp.wav")
        command = [ffmpeg_binary(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", block["path"],
                   "-af", ",".join(filters), "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(temporary)]
        code, detail = execute_ffmpeg(command, stop, timeout=max(240, block["samples"] / 12000),
                                     directory=directory, stage="alignment-coarse-" + block["id"])
        frames = _pcm_frames(temporary)
        if code or not 0 < frames <= 24000 * 300:
            raise RuntimeError("长配音定位副本生成失败，原配音已保留：" + detail[-1000:])
        check_cancel(stop)
        temporary.replace(path)
    return {"id": block["id"] + ".seed", "text": block["text"], "path": str(path), "samples": frames, "target": target_seconds}


def _refinement_windows(block, digest, coarse, seed, directory, stop):
    scale = block["samples"] / seed["samples"]
    anchors = [{**word, "start_time": word["start_time"] * scale, "end_time": word["end_time"] * scale} for word in coarse["words"]]
    core_seconds = min(200, seed["samples"] / 24000 - 20)
    windows, first = [], 0
    while first < len(anchors):
        check_cancel(stop)
        last = first + 1
        while last < len(anchors) and anchors[last]["end_time"] - anchors[first]["start_time"] <= core_seconds:
            last += 1
        if last < len(anchors):
            natural = [end for end in range(first + 1, last + 1)
                       if anchors[end-1]["end_time"] - anchors[first]["start_time"] >= max(0, core_seconds - 25)
                       and re.search(r"[。！？!?；;\n]", block["text"][anchors[end-1]["text_end"]:anchors[end]["text_start"]])]
            if natural:
                last = natural[-1]
        left, right = max(0, first - 8), min(len(anchors), last + 8)
        while left < first and anchors[first]["start_time"] - anchors[left]["start_time"] > 8:
            left += 1
        while right > last and anchors[right-1]["end_time"] - anchors[last-1]["end_time"] > 8:
            right -= 1
        start = max(0, math.floor((anchors[left]["start_time"] - .75) * 24000))
        end = min(block["samples"], math.ceil((anchors[right-1]["end_time"] + .75) * 24000))
        if not 0 < end - start <= 24000 * 300:
            raise ValueError("长配音粗定位边界异常，无法生成安全对齐窗口；原配音与定位结果已保留。")
        text_start, text_end = anchors[left]["text_start"], anchors[right-1]["text_end"]
        path = directory / "alignment" / "windows" / (stable_hash(("alignment-window-v1", digest, start, end)) + ".wav")
        if _pcm_frames(path) != end - start:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.wav")
            with wave.open(block["path"], "rb") as source, wave.open(str(temporary), "wb") as target:
                target.setparams(source.getparams())
                source.setpos(start)
                remaining = end - start
                while remaining:
                    check_cancel(stop)
                    count = min(remaining, 24000)
                    data = source.readframes(count)
                    if len(data) != count * 2:
                        raise ValueError("原配音数据不完整，无法截取对齐窗口。")
                    target.writeframesraw(data)
                    remaining -= count
            check_cancel(stop)
            temporary.replace(path)
        windows.append({"id": f"{block['id']}.W{len(windows)+1:04d}", "path": str(path), "samples": end - start,
                        "text": block["text"][text_start:text_end], "text_offset": text_start, "audio_offset": start / 24000,
                        "core_start": anchors[first]["text_start"], "core_end": anchors[last-1]["text_end"]})
        first = last
    write_json(directory / "alignment" / "windows" / (digest + "-plan.json"),
               {"key": stable_hash(("alignment-windowed-v1", digest, block["text"], coarse["signature"])),
                "block_id": block["id"], "source_sha256": digest, "coarse_key": coarse["key"],
                "coarse_samples": seed["samples"], "coarse_target": seed["target"], "windows": windows})
    return windows


def _merge_windows(block, windows, aligned, coarse):
    words, previous, errors, corrections = [], {}, [], 0
    for window in windows:
        mapped = [{**word, "text_start": word["text_start"] + window["text_offset"],
                   "text_end": word["text_end"] + window["text_offset"],
                   "start_time": word["start_time"] + window["audio_offset"],
                   "end_time": word["end_time"] + window["audio_offset"]} for word in aligned[window["id"]]["words"]]
        overlap = []
        for word in mapped:
            old = previous.get((word["text_start"], word["text_end"]))
            if old and old["end_time"] > old["start_time"] and word["end_time"] > word["start_time"]:
                overlap.extend(abs(old[field] - word[field]) for field in ("start_time", "end_time"))
        if overlap:
            discrepancy = median(overlap)
            errors.append(discrepancy)
            if discrepancy > .321:
                raise ValueError(f"配音块{block['id']}窗口{window['id']}交界定位差{discrepancy:.3f}秒，超过0.32秒；原配音与窗口缓存已保留。")
        core = [word for word in mapped if window["core_start"] <= word["text_start"] < word["text_end"] <= window["core_end"]]
        if not core or not any(word["end_time"] > word["start_time"] for word in core):
            raise ValueError("对齐窗口正文没有有效发声时间，原配音与窗口缓存已保留。")
        for word in core:
            if words and word["start_time"] < words[-1]["end_time"]:
                if words[-1]["end_time"] - word["start_time"] > .161:
                    raise ValueError(f"配音块{block['id']}窗口{window['id']}交界重叠{words[-1]['end_time']-word['start_time']:.3f}秒，超过0.16秒；原配音与窗口缓存已保留。")
                word["start_time"] = words[-1]["end_time"]
                word["end_time"] = max(word["start_time"], word["end_time"])
                corrections += 1
            words.append(word)
        previous = {(word["text_start"], word["text_end"]): word for word in mapped}
    _validate_words(words, block["text"], block["samples"] / 24000)
    stats = [aligned[window["id"]]["stats"] for window in windows]
    return words, {"method": "windowed_forced_alignment", "duration": block["samples"] / 24000,
                   "windows": len(windows), "encoded_tokens": max(coarse["stats"]["encoded_tokens"], *(item["encoded_tokens"] for item in stats)),
                   "coarse_encoded_tokens": coarse["stats"]["encoded_tokens"], "coarse_duration": coarse["stats"]["duration"],
                   "inference_seconds": sum(item["inference_seconds"] for item in stats) + coarse["stats"]["inference_seconds"],
                   "peak_vram_gib": max(coarse["stats"]["peak_vram_gib"], *(item["peak_vram_gib"] for item in stats)), "timestamp_grid_ms": 80,
                   "zero_duration_words": sum(word["end_time"] <= word["start_time"] for word in words),
                   "overlap_median_error": max(errors, default=0), "boundary_corrections": corrections}


def align_blocks(blocks, options, directory, stop=None, progress=None):
    """Keep TTS blocks intact, refining long ones in original-speed audio windows."""
    long = [block for block in blocks if block["samples"] / 24000 > 300]
    if not long:
        return _align_units(blocks, options, directory, stop, progress)
    _, signature = validate_alignment_options(options)
    directory = Path(directory)
    results, pending = {}, {}
    for block in long:
        check_cancel(stop)
        if _pcm_frames(block["path"]) != block["samples"]:
            raise ValueError("长配音文件与记录的PCM时长不一致，请检查原配音文件。")
        digest = file_hash(block["path"])
        key = stable_hash(("alignment-windowed-v1", digest, block["text"], signature))
        cache = directory / "alignment" / "cache" / (key + ".json")
        try:
            record = read_json(cache, default={})
            if isinstance(record, dict) and record.get("key") == key:
                _validate_words(record.get("words"), block["text"], block["samples"] / 24000)
                results[block["id"]] = record
                continue
        except (OSError, ValueError, KeyError, TypeError):
            pass
        try:
            plan = read_json(directory / "alignment" / "windows" / (digest + "-plan.json"), default={})
        except (OSError, ValueError):
            plan = {}
        target = plan.get("coarse_target", 240.0) if isinstance(plan, dict) and plan.get("key") == key else 240.0
        if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target) or not 30 <= target <= 240:
            target = 240.0
        pending[block["id"]] = {"block": block, "digest": digest, "key": key, "cache": cache, "target": target, "retries": 0}
    seeds = {}
    while pending:
        check_cancel(stop)
        seeds = {identifier: _coarse_block(item["block"], item["digest"], item["target"], directory, stop) for identifier, item in pending.items()}
        if progress:
            progress(0, "使用完整原文定位长配音边界，原配音保留原速")
        try:
            coarse = _align_units(list(seeds.values()), options, directory, stop,
                (lambda p, message: progress(p * .35, "定位长配音：" + message)) if progress else None)
            break
        except _AlignmentInputLimit as exc:
            identifier = exc.block_id.removesuffix(".seed")
            item = pending.get(identifier)
            if not item or item["retries"] >= 2 or item["target"] <= 30:
                raise RuntimeError("长配音定位仍超过模型文本上下文，请在句末拆分手动配音段。原配音与已完成缓存已保留。") from exc
            item["target"] = max(30, item["target"] - (exc.encoded_tokens - exc.limit + 128) / 13)
            item["retries"] += 1
            write_json(directory / "alignment" / "windows" / (item["digest"] + "-plan.json"),
                       {"key": item["key"], "block_id": identifier, "coarse_target": item["target"]})
    units = [block for block in blocks if block not in long]
    for identifier, item in pending.items():
        item["windows"] = _refinement_windows(item["block"], item["digest"], coarse[seeds[identifier]["id"]],
                                             seeds[identifier], directory, stop)
        units.extend(item["windows"])
    if units:
        aligned = _align_units(units, options, directory, stop,
            (lambda p, message: progress(.35 + p * .6, "原速精确对齐：" + message)) if progress else None)
        results.update({block["id"]: aligned[block["id"]] for block in blocks if block not in long})
        for identifier, item in pending.items():
            check_cancel(stop)
            words, stats = _merge_windows(item["block"], item["windows"], aligned, coarse[seeds[identifier]["id"]])
            record = {"schema": 1, "key": item["key"], "audio_hash": item["digest"], "signature": signature,
                      "words": words, "stats": stats, "window_keys": [aligned[window["id"]]["key"] for window in item["windows"]]}
            write_json(item["cache"], record)
            results[identifier] = record
    if progress:
        progress(1, "全部原速字词时间已对齐或从缓存恢复")
    return results


def _align_units(blocks, options, directory, stop=None, progress=None):
    """Align only missing blocks; save each result through existing atomic JSON I/O."""
    settings, signature = validate_alignment_options(options)
    directory = Path(directory)
    alignment = directory / "alignment"
    results, missing = {}, []
    for block in blocks:
        check_cancel(stop)
        seconds = block["samples"] / 24000
        if not 0 < seconds <= 300:
            raise ValueError(f"配音块{block['id']}长{seconds:.2f}秒，超过本地对齐5分钟限制；音频已保留。")
        digest = file_hash(block["path"])
        key = stable_hash(("alignment-result-v1", digest, block["text"], signature))
        cache = alignment / "cache" / (key + ".json")
        try:
            record = read_json(cache, default={})
            if isinstance(record, dict) and record.get("key") == key:
                _validate_words(record.get("words"), block["text"], seconds)
                results[block["id"]] = record
                continue
        except (OSError, ValueError, KeyError, TypeError):
            pass
        pcm = alignment / "pcm16" / (digest + ".wav")
        valid_pcm = False
        if pcm.is_file():
            try:
                with wave.open(str(pcm), "rb") as audio:
                    valid_pcm = ((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 16000)
                                 and abs(audio.getnframes() / 16000 - seconds) < .001)
            except (OSError, wave.Error):
                pass
        if not valid_pcm:
            pcm.parent.mkdir(parents=True, exist_ok=True)
            temporary = pcm.with_suffix(".tmp.wav")
            command = [ffmpeg_binary(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", block["path"],
                       "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(temporary)]
            code, detail = execute_ffmpeg(command, stop, timeout=240, directory=directory,
                                         stage="alignment-input-" + block["id"])
            if code or not temporary.is_file():
                raise RuntimeError("对齐音频转换失败：" + detail[-1000:])
            temporary.replace(pcm)
        missing.append({"id": block["id"], "text": block["text"], "input_path": str(pcm),
                        "key": key, "cache": cache, "duration": seconds, "audio_hash": digest})
    if not missing:
        if progress:
            progress(1, "全部字词时间缓存已命中")
        return results
    acquired = False
    try:
        while not acquired:
            check_cancel(stop)
            acquired = _GPU_LOCK.acquire(timeout=.5)
            if not acquired and progress:
                progress(len(results) / len(blocks), "等待本地对齐GPU任务结束")
        run_id = uuid4().hex
        job_file = alignment / "runs" / (run_id + ".json")
        write_json(job_file, {"model_path": settings["alignment_model"], "language": settings["alignment_language"],
                   "signature": signature, "jobs": [{key: item[key] for key in ("id", "text", "input_path")} for item in missing]})
        pending = {item["id"]: item for item in missing}
        last_error = ""

        def receive(line):
            nonlocal last_error
            if not line.startswith("{"):
                return
            try:
                event = json.loads(line)
            except ValueError:
                return
            kind = event.get("event")
            if kind == "error":
                last_error = event.get("message", "本地模型对齐失败")
                budget = re.search(r"配音块([^\s]+)编码长度(\d+)超过(\d+)", last_error)
                if budget:
                    raise _AlignmentInputLimit(last_error, budget[1], int(budget[2]), int(budget[3]))
                raise RuntimeError(last_error + "。配音和成功块已保留；详情见本地对齐日志。")
            elif kind == "result":
                item = pending[event["id"]]
                try:
                    words = _validate_words(event.get("words"), item["text"], item["duration"])
                except ValueError as exc:
                    raise ValueError(f"配音块{item['id']}：{exc}") from exc
                record = {"schema": 1, "key": item["key"], "audio_hash": item["audio_hash"], "signature": signature,
                          "words": words, "stats": event.get("stats", {})}
                write_json(item["cache"], record)
                results[item["id"]] = record
                if progress:
                    progress(len(results) / len(blocks), f"完成配音块{item['id']}对齐，{len(results)}/{len(blocks)}")
            elif kind in {"ready", "start"} and progress:
                message = "本地模型已加载：" + event.get("gpu", "") if kind == "ready" else "正在对齐配音块" + event.get("id", "")
                progress(len(results) / len(blocks), message)

        command = [settings["alignment_python"], "-I", "-B", "-X", "utf8", str(_WORKER), "--job", str(job_file)]
        code, detail = execute_ffmpeg(command, stop, timeout=max(600, sum(item["duration"] for item in missing) * 6),
            directory=directory, stage="qwen-" + run_id, line_callback=receive,
            diagnostic_group="alignment-diagnostics", process_name="本地对齐")
        if code or len(results) != len(blocks):
            raise RuntimeError((last_error or "本地对齐进程未完成：" + detail[-900:]) + "。配音和成功块已保存；详情见本地对齐日志。")
        return results
    finally:
        if acquired:
            _GPU_LOCK.release()


def estimate_block_words(blocks, stop=None):
    """Explicit opt-out: distribute measured block duration using text weights, without ASR."""
    from app.services.subtitle_segmentation import narration_weight, word_boundary_parts
    results = {}
    for block in blocks:
        check_cancel(stop)
        duration = block["samples"] / 24000
        position, parts = 0, []
        for part in word_boundary_parts(block["text"]):
            positions = [position + index for index, char in enumerate(part) if unicodedata.category(char)[0] in {"L", "N"}]
            if positions:
                parts.append((positions, narration_weight(part)))
            position += len(part)
        if not parts or not math.isfinite(duration) or duration <= 0:
            raise ValueError("当前配音块缺少有效时长或可估算的文字。")
        total = sum(weight for positions, weight in parts)
        words, elapsed = [], 0.0
        for positions, weight in parts:
            step = duration * weight / total / len(positions)
            for index in positions:
                words.append({"text": block["text"][index], "text_start": index, "text_end": index + 1,
                              "start_time": elapsed, "end_time": elapsed + step})
                elapsed += step
        words[-1]["end_time"] = duration
        results[block["id"]] = {"key": stable_hash(("text-timing-estimate-v1", block["text"], block["samples"], block["audio_key"])),
                                 "words": words, "stats": {"method": "text_estimate", "duration": duration}}
    return results


def map_timing(segments, blocks, ranges, manuscript, results, *, options=None, stop=None):
    """Keep visual intervals contiguous; give subtitles their spoken-word bounds."""
    layout = None
    if options and options.get("subtitles", True) and options.get("subtitle_single_line", False):
        from app.services.subtitle_segmentation import SubtitleLayout
        ratio = float(options.get("subtitle_width_ratio", .7))
        if not .5 <= ratio <= .95:
            raise ValueError("字幕单行宽度应在画面宽度的50%至95%之间。")
        width = 1080 if options.get("aspect") == "portrait" else 1920
        layout = SubtitleLayout(options.get("font"), options.get("font_size", 72), width * ratio,
                                options.get("subtitle_min_seconds", 1.0))
    if not blocks:
        raise ValueError("没有可对齐的配音块。")
    words = []
    block_map = {block["id"]: block for block in blocks}
    for block in blocks:
        check_cancel(stop)
        offset = block["start"]
        words.extend({**word, "text_start": word["text_start"] + block["text_start"],
                      "text_end": word["text_end"] + block["text_start"], "block_id": block["id"],
                      "start_time": word["start_time"] + offset, "end_time": word["end_time"] + offset}
                     for word in results[block["id"]]["words"])
    boundaries = [blocks[0]["start"]]
    for segment in segments[1:]:
        selected = [word for word in words if word["text_end"] > segment.text_start and word["text_start"] < segment.text_end]
        if not selected or selected[0]["text_start"] < segment.text_start:
            raise ValueError("论述段落边界落在不可独立定位的词中。")
        block = block_map[selected[0]["block_id"]]
        at_block_start = not manuscript[block["text_start"]:segment.text_start].strip()
        boundaries.append(block["start"] if at_block_start else selected[0]["start_time"])
    boundaries.append(blocks[-1]["end"])
    cues = []
    for index, segment in enumerate(segments):
        check_cancel(stop)
        segment.audio_start, segment.audio_end = boundaries[index:index + 2]
        associated = [block for block in blocks if block["text_end"] > segment.text_start and block["text_start"] < segment.text_end]
        segment.audio_path = associated[0]["path"] if len(associated) == 1 else str(Path(blocks[0]["path"]).parent.parent / "narration.wav")
        if segment.audio_end <= segment.audio_start:
            raise ValueError(f"段落{segment.id}对齐后没有有效时长。")
        segment_cues = []
        cue_ranges = (layout.plan(segment.text, segment.text_start, ranges[segment.id], words, stop)
                      if layout else ranges[segment.id])
        for cue_range in cue_ranges:
            selected = [(i, word) for i, word in enumerate(words) if word["text_end"] > cue_range["text_start"]
                        and word["text_start"] < cue_range["text_end"]]
            if not selected:
                raise ValueError(f"段落{segment.id}的字幕没有匹配到发声字词。")
            first, last = selected[0], selected[-1]
            if segment_cues and segment_cues[-1]["last_word"] == first[0]:
                cue = segment_cues[-1]
                cue.update(text_end=cue_range["text_end"], end=last[1]["end_time"], last_word=last[0])
                cue["text"] = manuscript[cue["text_start"]:cue["text_end"]].strip()
            else:
                segment_cues.append({"segment_id": segment.id, "block_id": first[1]["block_id"], "index": len(segment_cues),
                    **({key: cue_range[key] for key in ("font_size", "pixel_width", "short")} if layout else {}),
                    "text": manuscript[cue_range["text_start"]:cue_range["text_end"]].strip(),
                    "text_start": cue_range["text_start"], "text_end": cue_range["text_end"],
                    "start": first[1]["start_time"], "end": last[1]["end_time"], "last_word": last[0]})
        for cue in segment_cues:
            cue.pop("last_word")
            if cue["end"] <= cue["start"]:
                raise ValueError("字幕时间结果为零时长，配音已保留。")
            if cues and cue["start"] < cues[-1]["end"] - .001:
                raise ValueError("字幕时间结果出现重叠，配音已保留。")
            cues.append(cue)
    return cues
