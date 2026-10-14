"""Legacy cue voices and long narration blocks with sample-counted audio."""

import hashlib
import os
import re
import shutil
import wave
from pathlib import Path

from app.services.library.frame_extractor import execute_ffmpeg, ffmpeg_binary
from app.services.library.runner import DEFAULT_CALL_BUDGET, check_cancel
from app.services.library.store import read_json, stable_hash, write_json
from app.services.mimo_tts import MAX_TTS_HAN, synthesize, tts_han_count, validate_tts_text
from app.services.subtitle_segmentation import word_boundary_parts


LONG_TTS_GAP_SECONDS = 3.0


def public_profile(profile):
    return {key: value for key, value in profile.items() if key not in {"api_key", "key", "proxy"}}


def duration(path):
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def _samples(path):
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 24000) or audio.getnframes() <= 0:
            raise ValueError("配音缓存不是有效24kHz单声道PCM16音频。")
        return audio.getnframes()


def _join_audio(paths, output, stop=None, *, gap_seconds=0):
    gap_seconds = float(gap_seconds)
    if not 0 <= gap_seconds < float("inf"):
        raise ValueError("配音块间隔必须是有限的非负秒数。")
    silence = b"\0" * (round(gap_seconds * 24000) * 2)
    temporary = Path(output).with_suffix(".tmp.wav")
    with wave.open(str(temporary), "wb") as target:
        target.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        for index, path in enumerate(paths):
            check_cancel(stop)
            _samples(path)
            if index and silence:
                target.writeframes(silence)
            with wave.open(str(path), "rb") as source:
                while block := source.readframes(24000 * 10):
                    check_cancel(stop)
                    target.writeframes(block)
    temporary.replace(output)


def _cache_profile(profile):
    result = public_profile(profile)
    reference = profile.get("reference_audio")
    if reference and Path(reference).is_file():
        result["reference_hash"] = hashlib.sha256(Path(reference).read_bytes()).hexdigest()
    return result


def _request_raw(text, profile, raw, design_seed=None):
    validate_tts_text(text)
    engine = profile.get("engine", "mimo_tts")
    proxy = profile.get("proxy", {})
    temporary = raw.with_name(raw.name + ".tmp")
    if engine == "mimo_tts":
        proxies = {"http": None, "https": None} if proxy.get("enabled") is False else {
            "http": proxy.get("http") or proxy.get("https"), "https": proxy.get("https") or proxy.get("http")
        } if proxy.get("http") or proxy.get("https") else None
        effective = dict(profile)
        if design_seed:
            effective.update(model="mimo-v2.5-tts-voiceclone", reference_audio=str(design_seed))
        data, _ = synthesize(text, effective, proxies=proxies)
        temporary.write_bytes(data)
    elif engine == "edge_tts":
        import asyncio
        import edge_tts
        rate = float(profile.get("rate", 1.0))
        proxy_url = None if proxy.get("enabled") is False else proxy.get("https") or proxy.get("http")
        async def generate():
            client = edge_tts.Communicate(text, voice=profile.get("voice", "zh-CN-XiaoxiaoNeural"),
                                         rate=f"{round((rate-1)*100):+d}%", proxy=proxy_url)
            await client.save(str(temporary))
        asyncio.run(generate())
    else:
        raise ValueError("文案驱动模式目前支持小米MiMo TTS和Edge TTS。")
    temporary.replace(raw)


def _convert_raw(raw, output, stop, directory, stage, trim_edges):
    temporary = output.with_suffix(".tmp.wav")
    command = [ffmpeg_binary(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw)]
    if trim_edges:
        silence = "silenceremove=start_periods=1:start_duration=0.06:start_threshold=-50dB:start_silence=0.03"
        command += ["-af", silence + ",areverse," + silence + ",areverse"]
    command += ["-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(temporary)]
    code, detail = execute_ffmpeg(command, stop, timeout=240, directory=directory, stage=stage)
    if code or not temporary.is_file():
        raise RuntimeError(f"配音音频转换失败（FFmpeg返回码{code}），原始配音已保留。\n{detail[-1500:]}")
    _samples(temporary)
    temporary.replace(output)
    raw.unlink(missing_ok=True)


def _voice_seed(output, audio_dir):
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    seed = audio_dir / f"designed_voice-{digest[:20]}.wav"
    if not seed.is_file():
        with wave.open(str(output), "rb") as source, wave.open(str(seed), "wb") as target:
            target.setparams(source.getparams())
            target.writeframes(source.readframes(24000 * 20))
    return seed, digest


def _obtain_audio(text, profile, output, seed, trim, budget_check, stop, directory, stage, label, cache_source=None):
    if not output.is_file() and cache_source:
        source_directory = Path(cache_source)
        source = source_directory / output.name
        if not source_directory.is_symlink() and not source_directory.is_junction() and source.is_file() and not source.is_symlink():
            try:
                _samples(source)
            except (OSError, ValueError, wave.Error):
                pass
            else:
                try:
                    os.link(source, output)
                except OSError:
                    temporary = output.with_suffix(".reuse.tmp.wav")
                    shutil.copy2(source, temporary)
                    temporary.replace(output)
    if not output.is_file():
        raw = output.with_name(output.stem + (".raw.mp3" if profile.get("engine") == "edge_tts" else ".raw.wav"))
        if not raw.is_file():
            budget_check()
            try:
                _request_raw(text, profile, raw, seed)
            except Exception as exc:
                raise RuntimeError(f"{label}失败：{exc}") from None
        _convert_raw(raw, output, stop, directory, stage, trim)
    return _samples(output)


def synthesize_narration(segments, profile, directory, stop=None, progress=None, tts_budget=DEFAULT_CALL_BUDGET, *, subtitle_units, cache_source=None):
    """Original mode, including its existing cache keys and edge trimming."""
    directory = Path(directory)
    audio_dir = directory / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    cache_profile = _cache_profile(profile)
    units = [(segment, index, text) for segment in segments for index, text in enumerate(subtitle_units[segment.id])]
    if not units:
        raise ValueError("文案未生成有效配音单元。")
    calls, elapsed_samples = 0, 0
    cues, paths_by_segment, keys_by_segment = [], {}, {}
    design_seed = design_hash = None
    def budget_check():
        nonlocal calls
        if calls >= tts_budget:
            from app.services.narration_task import JobPaused
            raise JobPaused("已达到本次配音调用预算，已生成单元保留，点击继续生成。")
        calls += 1
    for unit_index, (segment, cue_index, text) in enumerate(units):
        check_cancel(stop)
        if progress:
            progress(unit_index / len(units), f"生成配音单元{unit_index+1}/{len(units)}，本次调用{calls}/{tts_budget}次")
        key = stable_hash(("narration-cue-audio-v2", text, cache_profile, design_hash))
        output = audio_dir / (key + ".wav")
        samples = _obtain_audio(text, profile, output, design_seed, True, budget_check, stop, directory,
                                f"audio-{unit_index+1:05d}", f"配音单元{segment.id}/{cue_index+1}", cache_source)
        if profile.get("model") == "mimo-v2.5-tts-voicedesign" and unit_index == 0:
            design_seed, design_hash = _voice_seed(output, audio_dir)
        start = elapsed_samples / 24000
        elapsed_samples += samples
        end = elapsed_samples / 24000
        if cue_index == 0:
            segment.audio_start = start
        segment.audio_end = end
        cues.append({"segment_id": segment.id, "index": cue_index, "text": text, "start": start, "end": end, "path": str(output)})
        paths_by_segment.setdefault(segment.id, []).append(output)
        keys_by_segment.setdefault(segment.id, []).append(key)
    for segment in segments:
        path = audio_dir / (stable_hash(("narration-segment-v2", keys_by_segment[segment.id])) + ".wav")
        _join_audio(paths_by_segment[segment.id], path, stop)
        segment.audio_path = str(path)
    combined = directory / "narration.wav"
    _join_audio([segment.audio_path for segment in segments], combined, stop)
    write_json(directory / "audio_manifest.json", {"version": 2, "duration": elapsed_samples / 24000,
        "profile_hash": stable_hash(cache_profile), "cues": cues,
        "segments": [{"id": s.id, "start": s.audio_start, "end": s.audio_end, "path": s.audio_path} for s in segments]})
    return str(combined)


def audio_break_candidates(manuscript):
    """Original word ends at sentence/paragraph boundaries; no generated or edited text."""
    paragraph_ends = [match.end() for match in re.finditer(r"(?:\r?\n[ \t]*){2,}", manuscript)]
    rows, start, position = [], 0, 0
    for part in word_boundary_parts(manuscript):
        position += len(part)
        paragraph = any(position - len(part) < end <= position for end in paragraph_ends)
        sentence = bool(re.search(r"[。！？!?；;][\s”’」』）)\]]*$", part))
        if paragraph or sentence or position == len(manuscript):
            fragment = manuscript[start:position]
            if fragment.strip():
                rows.append({"end": position, "text": fragment.strip(), "paragraph_end": paragraph,
                             "han": tts_han_count(fragment)})
                start = position
    return rows


def manual_audio_ranges(manuscript, break_after):
    if not manuscript.strip():
        raise ValueError("请输入完整文案后设置配音分段。")
    if not isinstance(break_after, list) or any(type(end) is not int for end in break_after):
        raise ValueError("手动配音断点必须为原文位置的整数列表。")
    if break_after != sorted(set(break_after)) or any(not 0 < end <= len(manuscript) for end in break_after):
        raise ValueError("手动配音断点已失效，请在当前文案上重新设置。")
    allowed = {row["end"] for row in audio_break_candidates(manuscript)}
    if any(end not in allowed for end in break_after):
        raise ValueError("配音断点需位于当前原文的句末或自然段末。")
    ends = sorted(set([*break_after, len(manuscript)]))
    result, start = [], 0
    for end in ends:
        fragment = manuscript[start:end]
        if fragment.strip():
            count = validate_tts_text(fragment)
            result.append({"text_start": start, "text_end": end, "text": fragment, "han": count})
            start = end
    if result and start < len(manuscript):
        result[-1].update(text_end=len(manuscript), text=manuscript[result[-1]["text_start"]:])
    if not result or "".join("".join(row["text"].split()) for row in result) != "".join(manuscript.split()):
        raise ValueError("手动配音分段未完整覆盖原文。")
    return result


def plan_audio_blocks(segments, manuscript, target_chars=800, *, break_after=None):
    if break_after is not None:
        blocks = [{**row, "id": f"B{i+1:04d}", "segment_ids": [segment.id for segment in segments
                   if segment.text_end > row["text_start"] and segment.text_start < row["text_end"]]}
                  for i, row in enumerate(manual_audio_ranges(manuscript, break_after))]
        return blocks
    target_chars = int(target_chars)
    if not 400 <= target_chars <= MAX_TTS_HAN:
        raise ValueError(f"长段配音目标字符数需在400至{MAX_TTS_HAN}之间。")
    groups, current, count = [], [], 0
    for segment in segments:
        size = sum(not char.isspace() for char in segment.text)
        if current and (count >= target_chars or count + size > min(target_chars * 1.25, MAX_TTS_HAN)):
            groups.append(current)
            current, count = [], 0
        current.append(segment)
        count += size
    if current:
        groups.append(current)
    if not groups:
        raise ValueError("没有可配音的文案段落。")
    if len(groups) > 1:
        tail = sum(sum(not c.isspace() for c in s.text) for s in groups[-1])
        previous = sum(sum(not c.isspace() for c in s.text) for s in groups[-2])
        if tail < target_chars / 3 and previous + tail <= min(target_chars * 1.25, MAX_TTS_HAN):
            groups[-2].extend(groups.pop())
    blocks = [{"id": f"B{i+1:04d}", "text_start": group[0].text_start, "text_end": group[-1].text_end,
               "text": manuscript[group[0].text_start:group[-1].text_end], "segment_ids": [s.id for s in group]}
              for i, group in enumerate(groups)]
    if "".join("".join(b["text"].split()) for b in blocks) != "".join(manuscript.split()):
        raise ValueError("长配音块没有完整覆盖原文。")
    for block in blocks:
        block["han"] = validate_tts_text(block["text"])
    return blocks


def synthesize_long_narration(segments, manuscript, profile, directory, stop=None, progress=None,
                              tts_budget=DEFAULT_CALL_BUDGET, target_chars=800, cache_source=None, break_after=None):
    directory = Path(directory)
    audio_dir = directory / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    blocks = plan_audio_blocks(segments, manuscript, target_chars, break_after=break_after)
    cache_profile = _cache_profile(profile)
    write_json(directory / "audio_plan.json", {"version": 1, "target_chars": target_chars,
               "chunking": "manual" if break_after is not None else "auto", "blocks": blocks})
    calls, elapsed_samples = 0, 0
    design_seed = design_hash = None
    gap_samples = round(LONG_TTS_GAP_SECONDS * 24000)
    def budget_check():
        nonlocal calls
        if calls >= tts_budget:
            from app.services.narration_task import JobPaused
            raise JobPaused("已达到本次配音调用预算，已生成长配音块保留。")
        calls += 1
    for index, block in enumerate(blocks):
        check_cancel(stop)
        if progress:
            progress(index / len(blocks), f"生成长配音块{index+1}/{len(blocks)}，本次调用{calls}/{tts_budget}次")
        key = stable_hash(("narration-block-audio-v3", block["text"], cache_profile, design_hash))
        output = audio_dir / (key + ".wav")
        samples = _obtain_audio(block["text"], profile, output, design_seed, False, budget_check, stop, directory,
                                "long-audio-" + block["id"], "长配音块" + block["id"], cache_source)
        if profile.get("model") == "mimo-v2.5-tts-voicedesign" and index == 0:
            design_seed, design_hash = _voice_seed(output, audio_dir)
        if index:
            elapsed_samples += gap_samples
        block.update(path=str(output), samples=samples, start=elapsed_samples / 24000, audio_key=key)
        elapsed_samples += samples
        block["end"] = elapsed_samples / 24000
    _join_audio([block["path"] for block in blocks], directory / "narration.wav", stop, gap_seconds=LONG_TTS_GAP_SECONDS)
    write_json(directory / "long_audio.json", {"version": 1, "duration": elapsed_samples / 24000,
               "profile_hash": stable_hash(cache_profile), "gap_seconds": LONG_TTS_GAP_SECONDS, "blocks": blocks})
    return blocks


def finalize_long_narration(segments, blocks, cues, alignment_results, directory):
    directory = Path(directory)
    audio = read_json(directory / "long_audio.json")
    write_json(directory / "audio_manifest.json", {"version": 3, "mode": "long_align",
        "timing_method": "text_estimate" if all(value.get("stats", {}).get("method") == "text_estimate" for value in alignment_results.values()) else "qwen_forced_alignment",
        "duration": audio["duration"], "profile_hash": audio["profile_hash"], "gap_seconds": audio.get("gap_seconds", 0), "blocks": blocks, "cues": cues,
        "segments": [{"id": s.id, "start": s.audio_start, "end": s.audio_end, "path": s.audio_path} for s in segments],
        "alignment": {key: {"cache_key": value["key"], "stats": value.get("stats", {})} for key, value in alignment_results.items()}})
    return str(directory / "narration.wav")


def create_narration_subtitles(segments, output):
    from app.services.script_subtitle import format_srt_time
    manifest = read_json(Path(output).parent / "audio_manifest.json")
    if manifest.get("version") not in {2, 3} or not manifest.get("cues"):
        raise ValueError("配音缺少有效字幕时间记录，请按任务保存的模式继续生成。")
    segment_ids = {segment.id for segment in segments}
    blocks = []
    for cue in manifest["cues"]:
        if cue["segment_id"] not in segment_ids or cue["end"] <= cue["start"]:
            raise ValueError("字幕单元时间记录无效。")
        display = " ".join(cue["text"].split())
        blocks.append(f"{len(blocks)+1}\n{format_srt_time(cue['start'])} --> {format_srt_time(cue['end'])}\n{display}")
    Path(output).write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return str(output)
