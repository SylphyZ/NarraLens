"""Persistent, resumable manuscript-to-video jobs; API credentials stay in memory."""

import json
import os
import shutil
import time
import re
import wave
from pathlib import Path
from uuid import UUID, uuid4

from loguru import logger

from app.services.library.gateway import ModelGateway, ModelUsage
from app.services.library.retriever import MaterialRetriever, RETRIEVAL_REPAIR_VERSION, CANDIDATE_QUEUE_VERSION, REVIEW_PROTOCOL
from app.services.library.runner import DEFAULT_CALL_BUDGET, JobCancelled, check_cancel, library_operation
from app.services.library import runner
from app.services.library.store import LibraryStore, library_ids, stable_hash, write_json, read_json
from app.services.library.indexer import audit_subtitles
from app.services.narration_planner import (
    BOUNDARY_ALIGNMENT_VERSION, plan_document, plan_subtitles, plan_visual_units, revise_visual_requirements, validate_visual_plan,
)
from app.services.narration_bgm import bgm_libraries, bgm_options, plan_bgm, prepare_bgm, validate_bgm
from app.services.narration_motion import (
    motion_options, validate_motion, plan_motion, prepare_motion, load_motion_assets,
    manual_motion_entries, motion_config_key,
)
from app.services.narration_alignment import (
    LEGACY_MODE, LONG_MODE, align_blocks, default_alignment_options, estimate_block_words, map_timing, validate_alignment_options,
)


class JobPaused(RuntimeError):
    pass


TASK_ROOT = Path(__file__).resolve().parents[2] / "storage" / "narration_tasks"
RESUME_STAGES = {
    "tts": "1. 从配音继续", "subtitles": "2. 从字幕处理继续", "visual": "3. 从寻找画面继续",
    "bgm": "4. 从配乐继续", "render": "5. 仅合成视频",
}


def task_directory(task_id):
    return TASK_ROOT / str(UUID(str(task_id)))


def read_task(task_id):
    path = task_directory(task_id) / "state.json"
    return read_json(path, default=None)


def _resume_level(stage):
    if stage == "auto":
        return 0
    if stage not in RESUME_STAGES:
        raise ValueError("不支持的继续阶段。")
    return list(RESUME_STAGES).index(stage) + 1


def _resume_checkpoint(directory, settings, text, level):
    """Read previous stages without regenerating them or touching source media."""
    from app.models.timeline import NarrationSegment, VisualShot
    from app.services.narration_audio import _samples, duration
    from app.services.narration_planner import subtitle_text_ranges

    def saved(name, label):
        try:
            value = read_json(directory / name)
        except (OSError, ValueError):
            raise ValueError(f"{label}结果缺失或损坏，请使用通用继续生成补齐。") from None
        if not isinstance(value, dict):
            raise ValueError(f"{label}结果不完整，请使用通用继续生成补齐。")
        return value

    if stable_hash(text) != settings["text_hash"]:
        raise ValueError("任务原文发生变化，请创建新任务。")
    plan = saved("document_plan.json", "文案分析")
    segments = [NarrationSegment(**row) for row in plan["segments"]]
    if plan.get("text_hash") != settings["text_hash"] or not segments or not plan.get("overview"):
        raise ValueError("文案分析与任务原文不一致，请使用通用继续生成。")
    if any(text[s.text_start:s.text_end] != s.text or not isinstance(s.requirement, dict) for s in segments) or len({s.id for s in segments}) != len(segments) or \
            re.sub(r"\s", "", "".join(s.text for s in segments)) != re.sub(r"\s", "", text):
        raise ValueError("文案段落缺失或顺序异常，请使用通用继续生成。")
    result = {"overview": plan["overview"], "segments": segments}
    options = settings["options"]
    mode = audio_mode(options)
    if level == 1:
        result["subtitle_units"] = saved("subtitle_plan.json", "字幕分段")["units"]
        subtitle_text_ranges(segments, result["subtitle_units"])
        return result
    audio_path = directory / "narration.wav"
    try:
        seconds = _samples(audio_path) / 24000
    except (OSError, ValueError, EOFError, wave.Error):
        raise ValueError("完整配音缺失或损坏，请先从配音继续。") from None
    if level == 2 and mode == LONG_MODE:
        audio = saved("long_audio.json", "长段配音")
        blocks = audio["blocks"]
        gap_seconds = float(audio.get("gap_seconds", 0))
        if not 0 <= gap_seconds < float("inf"):
            raise ValueError("配音块间隔记录无效，请先从配音继续。")
        position = 0.0
        for index, block in enumerate(blocks):
            path = Path(block["path"])
            if not path.is_file():
                raise ValueError("长配音块缺失，请先从配音继续。")
            if text[block["text_start"]:block["text_end"]] != block["text"] or \
                    not 0 <= block["start"] < block["end"] < float("inf") or not path.resolve().is_relative_to((directory / "audio").resolve()) or \
                    abs(block["start"] - position - (gap_seconds if index else 0)) > .01 or \
                    abs(duration(path) - (block["end"] - block["start"])) > .01:
                raise ValueError("长配音块缺失或时间不一致，请先从配音继续。")
            _samples(path)
            position = block["end"]
        if not blocks or not 0 < audio["duration"] < float("inf") or abs(position - seconds) > .01 or abs(audio["duration"] - seconds) > .01 or \
                re.sub(r"\s", "", "".join(b["text"] for b in blocks)) != re.sub(r"\s", "", text):
            raise ValueError("长段配音尚未完整完成，请先从配音继续。")
        result["blocks"] = blocks
    else:
        audio = saved("audio_manifest.json", "配音和字幕时间")
        timings = {row["id"]: row for row in audio["segments"]}
        if audio.get("version") not in {2, 3} or len(audio["segments"]) != len(segments) or set(timings) != {s.id for s in segments} or not audio.get("cues"):
            raise ValueError("配音或字幕时间记录不完整，请先从配音或字幕处理继续。")
        if any(cue["segment_id"] not in timings or not 0 <= cue["start"] < cue["end"] <= seconds + .05 for cue in audio["cues"]):
            raise ValueError("字幕时间记录无效，请先从字幕处理继续。")
        position = 0.0
        for segment in segments:
            row = timings[segment.id]
            if not 0 <= row["start"] < row["end"] < float("inf") or abs(row["start"] - position) > .01:
                raise ValueError("段落配音时间不连续，请先从配音或字幕处理继续。")
            segment.audio_start, segment.audio_end, segment.audio_path = row["start"], row["end"], row["path"]
            position = row["end"]
        if not 0 < audio["duration"] < float("inf") or abs(position - seconds) > .01 or abs(audio["duration"] - seconds) > .01:
            raise ValueError("配音时间记录与音轨不一致，请先从配音或字幕处理继续。")
        result["audio"] = audio
    result["audio_path"] = str(audio_path)
    if level >= 4:
        from app.services.timeline_planner import VISUAL_POLICY_VERSION, MATCH_POLICY_VERSION, matching_policy, validate_timeline
        timeline = saved("timeline.json", "镜头编排")
        policy = timeline.get("visual_policy", {})
        if not isinstance(policy, dict) or policy.get("version") != VISUAL_POLICY_VERSION:
            raise ValueError("此时间轴尚未执行2.6镜头去重与片尾过滤，请从3. 寻找画面继续；配音与字幕可复用。")
        if policy.get("matching", {}).get("version") != MATCH_POLICY_VERSION:
            raise ValueError("此时间轴尚未执行2.7句意匹配，请从3. 寻找画面继续；配音与字幕可复用。")
        visual_plan = validate_visual_plan(saved("visual_plan.json", "句意画面计划"), segments, text, directory)
        if visual_plan.get("requirements_pending"):
            raise ValueError("画面事实需求尚未整理完成，请从3. 寻找画面继续；已有配音、字幕与旧成片保留。")
        if policy["matching"] != matching_policy(visual_plan):
            raise ValueError("句意画面计划与时间线不一致，请从3. 寻找画面继续。")
        expected_motion = motion_config_key(options)
        recorded_motion = timeline.get("motion", {}).get("manual_config_key")
        if (expected_motion is not None or recorded_motion is not None) and expected_motion != recorded_motion:
            raise ValueError("手动动画设置与现有时间轴不同，请从3. 寻找画面继续，或创建动画副本。")
        previous = timeline["segments"]
        if len(previous) != len(segments) or any(row["id"] != s.id or row["text"] != s.text or
                abs(row["audio_start"] - s.audio_start) > .01 or abs(row["audio_end"] - s.audio_end) > .01
                for row, s in zip(previous, segments)):
            raise ValueError("镜头时间轴与现有配音不一致，请从寻找画面继续。")
        shots = [VisualShot(**row) for row in timeline["shots"]]
        assets = {}
        for shot in shots:
            if not 0 <= shot.timeline_start < shot.timeline_end < float("inf") or not 0 <= shot.source_start < shot.source_end < float("inf"):
                raise ValueError("镜头时间记录无效，请从寻找画面继续。")
            asset = assets.setdefault(shot.asset_id, {"id": shot.asset_id, "path": shot.video_path, "duration": 0})
            asset["duration"] = max(asset["duration"], shot.source_end)
        for identifier, source in policy["sources"].items():
            if identifier in assets:
                assets[identifier].update({key: source[key] for key in ("fingerprint", "fps", "duration")})
        for asset in load_motion_assets(directory, shots, plan_key=timeline.get("motion", {}).get("plan_key")):
            assets[asset["id"]] = asset
        validate_timeline(segments, shots, list(assets.values()), policy=policy)
        result.update(shots=shots, timeline=timeline)
    if level == 5 and options.get("bgm_mode") == "auto" and options.get("bgm_volume", .15) > 0:
        music = saved("bgm_plan.json", "自动配乐")
        if not music.get("render_ready") or not (directory / "bgm.wav").is_file() or not 0 < music["duration"] < float("inf") or \
                abs(music["duration"] - seconds) > .05 or abs(duration(directory / "bgm.wav") - seconds) > .05:
            raise ValueError("完整背景音乐尚未完成，请先从配乐继续。")
        result["music"] = music
    elif level == 5 and options.get("bgm_path") and options.get("bgm_volume", .15) > 0 and not Path(options["bgm_path"]).is_file():
        raise ValueError("保存的背景音乐文件不存在，请调整配乐并创建副本。")
    return result


def resume_stage_status(task_id):
    """The UI and runner use the same prerequisites; this function only reads."""
    directory = task_directory(task_id)
    settings = read_json(directory / "settings.json")
    with (directory / "input.txt").open("r", encoding="utf-8", newline="") as source:
        text = source.read()
    result = {}
    for stage in RESUME_STAGES:
        try:
            _resume_checkpoint(directory, settings, text, _resume_level(stage))
            result[stage] = {"ready": True, "reason": "复用前序结果，从此阶段继续到成片；有效缓存仍会复用。"}
        except (OSError, ValueError, KeyError, TypeError, EOFError, wave.Error, ZeroDivisionError) as exc:
            result[stage] = {"ready": False, "reason": str(exc) if isinstance(exc, ValueError) else
                             "前序结果缺失或不完整，请使用通用继续生成补齐。"}
    return result


def list_tasks():
    if not TASK_ROOT.is_dir():
        return []
    result = []
    for path in TASK_ROOT.glob("*/state.json"):
        try:
            state = read_json(path)
            if state is None:
                continue
            state["modified"] = path.stat().st_mtime
            result.append(state)
        except (OSError, ValueError):
            continue
    return sorted(result, key=lambda item: item["modified"], reverse=True)


def _checked_task_directory(task_id):
    identifier = str(UUID(str(task_id)))
    if TASK_ROOT.is_symlink() or TASK_ROOT.is_junction():
        raise ValueError("任务根目录是链接，拒绝删除。")
    root = TASK_ROOT.resolve(strict=True)
    workspace = Path(__file__).resolve().parents[3].resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("任务根目录解析后超出工作区，拒绝删除。")
    logical = TASK_ROOT / identifier
    if logical.is_symlink() or logical.is_junction():
        raise ValueError("任务目录是链接，拒绝删除。")
    target = logical.resolve(strict=True)
    if target.parent != root or target.name != identifier or not target.is_dir():
        raise ValueError("任务目录越界，拒绝删除。")
    for folder, directories, filenames in os.walk(target, followlinks=False):
        for name in directories + filenames:
            entry = Path(folder) / name
            if entry.is_symlink() or entry.is_junction():
                raise ValueError("任务包含链接目录或文件，拒绝递归删除。")
    return target


def task_deletion_info(task_id):
    target = _checked_task_directory(task_id)
    state = read_task(task_id) or {}
    files = [entry for entry in target.rglob("*") if entry.is_file()]
    return {"id": str(UUID(str(task_id))), "state": state.get("state", ""), "phase": state.get("phase", ""),
            "directory": str(target), "files": len(files), "bytes": sum(entry.stat().st_size for entry in files),
            "has_video": (target / "final.mp4").is_file(), "running": runner.running("narration:" + str(UUID(str(task_id))))}


def delete_task(task_id, confirmed=False, interrupted_confirmed=False):
    """Delete a reviewed task; retain audio needed by existing direct forks."""
    if not confirmed:
        raise ValueError("请先确认删除选定任务及其产物。")
    identifier = str(UUID(str(task_id)))
    _checked_task_directory(identifier)
    settings = read_json(task_directory(identifier) / "settings.json", default={})
    state = read_task(identifier) or {}
    with library_operation(identifier, prefix="narration:"), library_operation(library_ids(settings) or library_ids(state) or identifier):
        if runner.running("narration:" + identifier):
            raise ValueError("任务仍在运行，请先停止并等待结束。")
        info = task_deletion_info(identifier)
        if info["state"] == "running" and not interrupted_confirmed:
            raise ValueError("任务记录仍为运行中，请确认旧进程及其他窗口已经停止。")
        target = _checked_task_directory(identifier)
        preserved = 0
        for child in list_tasks():
            if child["id"] == identifier:
                continue
            child_settings = read_json(task_directory(child["id"]) / "settings.json", default={})
            if child_settings.get("parent_task_id") != identifier:
                continue
            if runner.running("narration:" + child["id"]) or (child.get("state") == "running" and not interrupted_confirmed):
                raise ValueError("关联副本仍在运行，请先停止或确认旧进程已结束。")
            child_directory = _checked_task_directory(child["id"])
            destination = child_directory / "audio"
            destination.mkdir(exist_ok=True)
            for source in (target / "audio").glob("*.wav"):
                output = destination / source.name
                if output.exists():
                    continue
                try:
                    os.link(source, output)
                except OSError:
                    temporary = output.with_suffix(".preserve.tmp.wav")
                    shutil.copy2(source, temporary)
                    temporary.replace(output)
                preserved += 1
        # Resolve and verify both final paths before a directory move/delete.
        target = _checked_task_directory(identifier)
        root = TASK_ROOT.resolve(strict=True)
        tombstone = root / (".deleting-" + identifier + "-" + uuid4().hex)
        if tombstone.resolve().parent != root or target.parent != root:
            raise ValueError("删除目录越界。")
        target.rename(tombstone)
        try:
            if tombstone.resolve(strict=True).parent != root:
                raise ValueError("删除目录越界。")
            shutil.rmtree(tombstone)
        except OSError:
            return f"任务已从列表移除，文件仍被占用，请关闭占用后清理：{tombstone}"
        return f"任务已删除，保留了{preserved}个副本可复用音频文件。"


def public_models(profile):
    return {name: {key: value for key, value in settings.items() if key not in {"key", "api_key"}}
            if isinstance(settings, dict) else settings for name, settings in profile.items() if name != "proxy"}


def audio_mode(options):
    """Tasks created before the mode field always retain the original route."""
    mode = options.get("audio_mode", LEGACY_MODE)
    if mode not in {LEGACY_MODE, LONG_MODE}:
        raise ValueError("任务保存的配音模式不受支持，请检查任务配置。")
    return mode


def normalized_audio_options(options):
    from app.services.mimo_tts import MAX_TTS_HAN
    result = dict(options)
    result["audio_mode"] = audio_mode(result)
    if result["audio_mode"] == LONG_MODE:
        value = result.get("tts_block_chars", 800)
        try:
            valid = not isinstance(value, bool) and int(value) == float(value) and 400 <= int(value) <= MAX_TTS_HAN
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f"长段配音目标字符数需为400至{MAX_TTS_HAN}的整数。")
        result = {**default_alignment_options(), **result, "tts_block_chars": int(value)}
        result["tts_chunking"] = result.get("tts_chunking", "auto")
        if result["tts_chunking"] not in {"auto", "manual"}:
            raise ValueError("配音分段方式无效。")
        result["alignment_enabled"] = result.get("alignment_enabled", True)
        if type(result["alignment_enabled"]) is not bool:
            raise ValueError("本地对齐开关需为启用或关闭。")
    return result


def restore_voice(recorded, current, *, profile_id=None):
    """Refresh credentials only for the recorded provider endpoint."""
    from app.services.model_profiles import restore_narration_voice
    return restore_narration_voice(recorded, current, profile_id=profile_id)


def restore_models(recorded, current, require_vision=True, *, required_kinds=None):
    required = {"text", "text_fast", *(["vision"] if require_vision else [])} if required_kinds is None else set(required_kinds)
    restored = dict(recorded)
    from app.services.model_profiles import credentials_for
    bindings = recorded.get("_bindings", {})
    for kind in ("text", "text_fast", "vision"):
        saved = dict(recorded.get(kind, recorded.get("text", {})) if kind == "text_fast" else recorded.get(kind, {}))
        active = current.get(kind, current.get("text", {})) if kind == "text_fast" else current.get(kind, {})
        same_endpoint = (saved.get("base") or "").rstrip("/") == (active.get("base") or "").rstrip("/")
        saved["key"] = credentials_for(saved, kind, identifier=bindings.get(kind), current=active) if kind in required else active.get("key", "") if same_endpoint else ""
        restored[kind] = saved
    for kind in ("text_backup", "vision_backup"):
        saved = dict(recorded.get(kind, {}))
        active = current.get(kind, {})
        same_endpoint = (saved.get("base") or "").rstrip("/") == (active.get("base") or "").rstrip("/")
        if bindings.get(kind) and saved.get("enabled"):
            saved["key"] = credentials_for(saved, kind, identifier=bindings[kind], current=active)
        else:
            saved["enabled"] = bool(saved.get("enabled") and active.get("enabled"))
            saved["key"] = active.get("key", "") if same_endpoint else ""
        restored[kind] = saved
    restored["proxy"] = dict(current.get("proxy", {}))
    return restored


def _task_options(text, options, *, draft=False, reuse_audio=False):
    """Drafts check data; generation also checks bindings and local resources."""
    from app.services.narration_audio import manual_audio_ranges
    if not isinstance(options, dict):
        raise ValueError("项目设置应为对象。")
    result = motion_options(bgm_options(normalized_audio_options(options)))
    value = result.get("visual_no_gain_rounds", 2)
    if type(value) is not int or value < 1:
        raise ValueError("连续无改善轮数需为正整数。")
    result["visual_no_gain_rounds"] = value
    for name in ("subtitles", "verify_visual", "mask_source_subtitles", "subtitle_allow_short"):
        if name in result and type(result[name]) is not bool:
            raise ValueError(name + " 应为开关。")
    if result.get("aspect", "landscape") not in {"landscape", "portrait"}:
        raise ValueError("画面比例无效。")
    if type(result.get("font_size", 40)) is not int or result.get("font_size", 40) < 1:
        raise ValueError("字幕字号需为正整数。")
    if "font" in result and not isinstance(result["font"], str):
        raise ValueError("字幕字体路径应为文字。")
    if audio_mode(result) == LONG_MODE and result["tts_chunking"] == "manual":
        ends = result.get("tts_break_after", [])
        if not isinstance(ends, list) or any(type(end) is not int for end in ends) or not isinstance(result.get("tts_break_text_hash", ""), str):
            raise ValueError("手动配音断点格式无效。")
        if not draft:
            if result.get("tts_break_text_hash") != stable_hash(text):
                raise ValueError("手动配音断点与当前原文不一致，请重新设置分段。")
            manual_audio_ranges(text, ends)
    if not draft:
        result = validate_motion(validate_bgm(result))
        if not reuse_audio and audio_mode(result) == LONG_MODE and result["alignment_enabled"]:
            validate_alignment_options(result)
        manual_motion_entries(text, result)
    return result


def project_edit_status(task_id):
    """One conservative rule for editing new projects, including old pending records."""
    identifier = str(UUID(str(task_id)))
    directory = task_directory(identifier)
    settings = read_json(directory / "settings.json")
    state = read_task(identifier) or {}
    inputs = {"input.txt", "settings.json", "state.json", "source.json", "original.txt", "original.md", "original.markdown"}
    unstarted = (state.get("state") == "pending" and state.get("phase", "待开始") == "待开始"
                 and state.get("progress", 0) == 0 and not state.get("video") and not state.get("model_calls")
                 and not settings.get("generation_started") and not settings.get("parent_task_id")
                 and not any(path.name not in inputs for path in directory.iterdir()))
    busy = (runner.running("narration:" + identifier) or runner.library_busy(identifier, prefix="narration:")
            or runner.library_busy(library_ids(settings))
            or runner.library_busy(bgm_libraries(settings["options"]), prefix="audio-library:"))
    return {"unstarted": bool(unstarted), "editable": bool(unstarted and not busy),
            "reason": "项目或关联资源正在使用，结束后可应用设置。" if busy else
                      "尚未开始，可保存本项目设置后开始生成。" if unstarted else
                      "项目已开始或已有成果，修改设置请通过复用副本应用；普通继续沿用原设置。"}


def update_project_settings(task_id, selected_libraries, options, voice_profile, model_profile, *, for_start=False):
    """Atomically replace a new project's config; never edit a generated project."""
    from app.services.narration_audio import public_profile
    identifier = str(UUID(str(task_id)))
    directory = task_directory(identifier)
    with library_operation(identifier, prefix="narration:"):
        saved = read_json(directory / "settings.json")
        selected_libraries = library_ids(selected_libraries)
        with library_operation(library_ids(saved) + selected_libraries), \
                library_operation(bgm_libraries(saved["options"]) + bgm_libraries(options), prefix="audio-library:"):
            status = project_edit_status(identifier)
            # Our own resource claims make the read-only status busy; recheck the immutable condition here.
            if not status["unstarted"] or runner.running("narration:" + identifier):
                raise ValueError("项目已经开始或已有成果，请创建复用副本应用修改。")
            _checked_task_directory(identifier)
            if not selected_libraries:
                raise ValueError("请至少选择一个视频素材库。")
            store = LibraryStore()
            for library in selected_libraries:
                store.directory(library)
            with (directory / "input.txt").open("r", encoding="utf-8", newline="") as source:
                text = source.read()
            if stable_hash(text) != saved["text_hash"]:
                raise ValueError("项目原文发生变化，请创建新项目。")
            options = _task_options(text, options, draft=not for_start)
            if for_start:
                manual_motion_entries(text, options, complete=True)
            from app.services.model_profiles import validate_narration_voice
            validate_narration_voice(voice_profile, require_credentials=for_start)
            updated = {**saved, "library_id": selected_libraries[0], "library_ids": selected_libraries,
                       "options": options, "voice": public_profile(voice_profile), "models": public_models(model_profile),
                       "project_version": (TASK_ROOT.parent.parent / "project_version").read_text(encoding="utf-8").strip()}
            state = read_task(identifier)
            summary = {**state, "library_id": selected_libraries[0], "library_ids": selected_libraries, "audio_mode": audio_mode(options)}
            # Config is the commit point. If its atomic write fails, restore the old progress summary.
            write_json(directory / "state.json", summary)
            try:
                write_json(directory / "settings.json", updated)
            except Exception:
                write_json(directory / "state.json", state)
                raise
            return updated


def create_task(library_id, text, options, voice_profile, source=None, model_profile=None, *, draft=False):
    with library_operation(library_id), library_operation([] if draft else bgm_libraries(options), prefix="audio-library:"):
        return _create_task(library_id, text, options, voice_profile, source, model_profile, draft=draft)


def _create_task(library_id, text, options, voice_profile, source, model_profile, *, reuse_audio=False, draft=False):
    from app.services.narration_audio import public_profile
    selected_libraries = library_ids(library_id)
    if not selected_libraries:
        raise ValueError("请至少选择一个素材库。")
    options = _task_options(text, options, draft=draft, reuse_audio=reuse_audio)
    store = LibraryStore()
    for identifier in selected_libraries:
        store.directory(identifier)
    if not text.strip():
        raise ValueError("请输入完整文案。")
    task_id = str(uuid4())
    directory = task_directory(task_id)
    directory.mkdir(parents=True)
    (directory / "input.txt").write_text(text, encoding="utf-8", newline="")
    if source and source.get("extension") in {".txt", ".md", ".markdown"}:
        (directory / ("original" + source["extension"])).write_text(source.get("text", ""), encoding="utf-8", newline="")
        write_json(directory / "source.json", {"name": source.get("name", ""), "extension": source["extension"]})
    settings = {"library_id": selected_libraries[0], "library_ids": selected_libraries, "options": options,
                "voice": public_profile(voice_profile), "text_hash": stable_hash(text), "schema": 2, "generation_started": False,
                "project_version": (TASK_ROOT.parent.parent / "project_version").read_text(encoding="utf-8").strip()}
    if model_profile:
        settings["models"] = public_models(model_profile)
    write_json(directory / "settings.json", settings)
    state = {"id": task_id, "library_id": selected_libraries[0], "library_ids": selected_libraries,
             "state": "pending", "progress": 0, "phase": "待开始", "message": "任务已创建", "video": "", "audio_mode": audio_mode(options)}
    write_json(directory / "state.json", state)
    return task_id


def fork_task(task_id, options, voice_profile, model_profile, use_current_models=False, *, selected_libraries=None, reuse_audio=False):
    """Create a new task with the same manuscript and reusable, validated caches."""
    source_directory = task_directory(task_id)
    source_settings = read_json(source_directory / "settings.json")
    source_libraries = library_ids(source_settings)
    target_libraries = library_ids(selected_libraries) if selected_libraries is not None else source_libraries
    with library_operation(task_id, prefix="narration:"), library_operation(source_libraries + target_libraries), \
            library_operation(bgm_libraries(options), prefix="audio-library:"):
        if runner.running("narration:" + str(UUID(str(task_id)))):
            raise ValueError("来源任务正在运行，请先停止并等待结束，再创建复用副本。")
        with (source_directory / "input.txt").open("r", encoding="utf-8", newline="") as source:
            text = source.read()
        if stable_hash(text) != source_settings["text_hash"]:
            raise ValueError("来源任务原文发生变化，无法创建复用缓存的副本。")
        if reuse_audio:
            from app.services.narration_audio import public_profile
            original_options = {key: value for key, value in source_settings["options"].items() if not key.startswith("motion_")}
            requested_options = {key: value for key, value in options.items() if not key.startswith("motion_")}
            if original_options != requested_options or public_profile(voice_profile) != source_settings["voice"]:
                raise ValueError("完整配音复用仅适用于保持原配音、字幕及输出设置的动画副本。")
            _resume_checkpoint(source_directory, source_settings, text, 3)
        models = dict(model_profile if use_current_models or "models" not in source_settings else source_settings["models"])
        from app.services.narration_audio import public_profile
        bindings = dict(models.get("_bindings", {}))
        for role in ("tts", "alignment"):
            same = public_profile(voice_profile) == source_settings["voice"] if role == "tts" else all(
                options.get(name) == source_settings["options"].get(name) for name in ("alignment_python", "alignment_model"))
            identifier = (source_settings.get("models", {}) if same else model_profile).get("_bindings", {}).get(role)
            if identifier:
                bindings[role] = identifier
            else:
                bindings.pop(role, None)
        if bindings:
            models["_bindings"] = bindings
        child_id = _create_task(target_libraries, text, options, voice_profile, None, models, reuse_audio=reuse_audio)
        child_directory = task_directory(child_id)
        settings = read_json(child_directory / "settings.json")
        settings["parent_task_id"] = str(UUID(str(task_id)))
        write_json(child_directory / "settings.json", settings)
        copied = 0
        try:
            for folder in ("planning", "candidates", "alignment/cache", "bgm/planning", "motion/planning"):
                source_folder = source_directory / folder
                if source_folder.is_symlink() or source_folder.is_junction():
                    raise ValueError("来源任务缓存目录是链接，请检查后再创建副本。")
                for source in source_folder.glob("*.json"):
                    if source.is_symlink():
                        continue
                    target = child_directory / folder / source.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    copied += 1
            for name in ("document_plan.json", "subtitle_plan.json", "source.json"):
                source = source_directory / name
                if source.is_file() and not source.is_symlink():
                    shutil.copy2(source, child_directory / name)
            if reuse_audio:
                for source in [source_directory / "narration.wav", source_directory / "subtitles.srt",
                               *(source_directory / "audio").glob("*.wav")]:
                    if not source.is_file():
                        continue
                    if source.is_symlink() or source.parent.is_symlink() or source.parent.is_junction():
                        raise ValueError("来源配音缓存是链接，请检查后再复用。")
                    destination = child_directory / source.relative_to(source_directory)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(source, destination)
                    except OSError:
                        shutil.copy2(source, destination)
                for name in ("audio_manifest.json", "long_audio.json"):
                    if not (source_directory / name).is_file():
                        continue
                    record = read_json(source_directory / name)
                    for row in record.get("segments", []) + record.get("blocks", []):
                        path = Path(row["path"])
                        if not path.resolve().is_relative_to(source_directory.resolve()):
                            raise ValueError("来源配音记录引用任务外音频，无法完整复用。")
                        row["path"] = str(child_directory / path.relative_to(source_directory))
                    write_json(child_directory / name, record)
                settings["audio_reused"] = True
                write_json(child_directory / "settings.json", settings)
            source_info = read_json(source_directory / "source.json", default={})
            extension = source_info.get("extension")
            if extension in {".txt", ".md", ".markdown"}:
                source = source_directory / ("original" + extension)
                if source.is_file() and not source.is_symlink():
                    shutil.copy2(source, child_directory / source.name)
        except Exception as exc:
            state = read_task(child_id)
            state.update(state="failed", message=f"副本缓存复制未完成：{exc}", parent_task_id=settings["parent_task_id"])
            write_json(child_directory / "state.json", state)
            raise
        state = read_task(child_id)
        state.update(parent_task_id=settings["parent_task_id"], inherited_cache_files=copied,
                     message="已创建动画副本，复用完整配音和字幕，可从3. 寻找画面继续。" if reuse_audio else
                             f"已创建副本，继承{copied}个缓存文件；匹配的音频将在生成时复用。")
        write_json(child_directory / "state.json", state)
        return child_id


def run_task(task_id, model_profile, voice_profile, stop=None, use_current_models=False, *, start_stage="auto"):
    with library_operation(task_id, prefix="narration:"):
        directory = task_directory(task_id)
        settings = read_json(directory / "settings.json")
        with library_operation(library_ids(settings)), library_operation(bgm_libraries(settings["options"]), prefix="audio-library:"):
            if not settings.get("generation_started"):
                settings["generation_started"] = True
                write_json(directory / "settings.json", settings)
            return _run_task(task_id, model_profile, voice_profile, stop, use_current_models, start_stage=start_stage)


def _preserve_visual_result(directory):
    """Keep the previous timeline and completed video before a new visual pass."""
    final, timeline = directory / "final.mp4", directory / "timeline.json"
    if not timeline.is_file():
        return
    signature = stable_hash((timeline.read_text(encoding="utf-8-sig", errors="replace"),
                             (final.stat().st_size, final.stat().st_mtime_ns) if final.is_file() else None))
    archive = directory / "history" / ("visual-" + signature[:20])
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("timeline.json", "visual_plan.json", "visual_matches.json", "visual_proposal.json", "visual_search_state.json", "settings.json", "state.json", "motion/plan.json", "motion/manifest.json", "bgm_plan.json"):
        source = directory / name
        if source.is_file():
            target = archive / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
    retained = archive / "final.mp4"
    if final.is_file() and not retained.exists():
        try:
            os.link(final, retained)
        except OSError:
            # Other filesystems may not support links; preserve before overwriting.
            temporary = archive / "final.tmp.mp4"
            shutil.copy2(final, temporary)
            temporary.replace(retained)
    write_json(archive / "retained-files.json", {"original_task": str(directory), "video": str(retained) if retained.is_file() else None})


def _find_visual_shots(segments, text, visual_plan, assets, reservations, retriever, gateway, directory, options, stop, update, *, previous_plan=None):
    """Propose globally, inspect only proposed pairs, and retain a resumable local search cursor."""
    from app.models.timeline import NarrationSegment
    from app.services.library.frame_extractor import scene_boundaries
    from app.services.timeline_planner import (
        MATCH_POLICY_VERSION, PROPOSAL_VERSION, VisualCapacityError, source_shot_key, unused_source,
        visual_unit_intervals, plan_visual_timeline, propose_visual_timeline,
    )
    units = visual_plan["units"]
    visual_segments = {u["id"]: NarrationSegment(u["id"], u["text_start"], u["text_end"], text[u["text_start"]:u["text_end"]],
                      u["requirement"], audio_start=u["start"], audio_end=u["end"]) for u in units}
    asset_map = {a["id"]: a for a in assets}
    verify = bool(options.get("verify_visual", True))
    target = float(options.get("shot_seconds", 7))
    no_gain_limit = int(options.get("visual_no_gain_rounds", 2))
    search_identity = (retriever.version, verify, target, gateway.signature("text_fast"),
                       gateway.signature("text"), gateway.signature("vision"))
    key = stable_hash((PROPOSAL_VERSION, visual_plan["key"], *search_identity))
    search_signature = stable_hash((PROPOSAL_VERSION, *search_identity))
    unit_signatures = {u["id"]: stable_hash(u) for u in units}
    alignment = visual_plan.get("boundary_alignment", {})
    base_key = alignment.get("base_plan_key") if alignment.get("version") == BOUNDARY_ALIGNMENT_VERSION else None
    revisions = visual_plan.get("requirement_revisions", {})
    retriever.configure_requirements(visual_segments, revisions, gateway, previous_plan=previous_plan)
    compatible_keys = {key}
    if not any(row.get("changed") for row in revisions.get("entries", [])):
        compatible_keys.update(stable_hash((PROPOSAL_VERSION, value, *search_identity)) for value in
                               (base_key, revisions.get("base_plan_key")) if value)
    path = directory/"visual_search_state.json"
    try:
        saved = read_json(path, default={})
    except (OSError, ValueError):
        saved = {}
    if not isinstance(saved, dict) or \
            not isinstance(saved.get("pools", {}), dict) or not isinstance(saved.get("examined", {}), dict):
        saved = {}
    migration = None
    repaired_indices = set()
    if saved and saved.get("key") not in compatible_keys:
        previous_units = {u["id"]: stable_hash(u) for u in previous_plan["units"]} if previous_plan else {}
        previous_source = next((version for version in retriever.previous_versions if saved.get("search_signature") ==
            stable_hash((PROPOSAL_VERSION,version,*search_identity[1:]))),None)
        if previous_source and isinstance(saved.get("unit_signatures"),dict):
            known = saved["unit_signatures"]
            repaired_indices = set(retriever.previous_versions[previous_source])
        elif saved.get("search_signature") == search_signature and isinstance(saved.get("unit_signatures"), dict):
            known = saved["unit_signatures"]
        elif previous_plan and saved.get("key") == stable_hash((PROPOSAL_VERSION, previous_plan["key"], *search_identity)):
            known = previous_units
        else:
            known = None
        if known is None:
            # Unproven cursors are reset; physical ranges still pass prepare's current source/content checks.
            saved = {"pools":saved.get("pools",{}),"key":saved.get("key"),"search_round":saved.get("search_round",0),"examined":{}}
        else:
            retained = {uid for uid, signature in unit_signatures.items() if known.get(uid) == signature}
            saved = {**saved, "examined": {uid: [i for i in rows if i not in repaired_indices] for uid, rows in saved["examined"].items() if uid in retained},
                     "exhausted": [uid for uid in saved.get("exhausted", []) if uid in retained and not repaired_indices]}
            # Old ranges can be proposed again, but prepare() revalidates every current requirement/key.
            migration = {"from_key": saved.get("key"), "retained_units": sorted(retained),
                         "reset_units": sorted(set(unit_signatures)-retained),"repaired_source_indices":sorted(repaired_indices)}
    migrated_from = saved.get("key") if saved and saved.get("key") != key else saved.get("migrated_from_key")
    if migrated_from and migrated_from != saved.get("migrated_from_key"):
        logger.info("已核对搜索身份并复用候选范围；仅变更合同的单元重置游标：{}，{}", migrated_from, migration)
    examined = {u["id"]: set(saved.get("examined", {}).get(u["id"], [])) for u in units}
    exhausted = set(saved.get("exhausted", []))
    no_gain = {u["id"]: 0 for u in units}  # Explicit continue grants another bounded expansion from the saved cursor.
    waves, processed, paused_units = {}, {u["id"]: set() for u in units}, {}
    uncertain_waves, pause_dependencies = {u["id"]: 0 for u in units}, {}
    local_search = {u["id"]: {} for u in units}
    occupied, occupancy = [], stable_hash([])
    retriever.configure_queue(saved.get("candidate_progress", {}), gateway)
    update("寻找画面", 47, "本地重新核对已付费候选和旧判断，复用已确认结果；此步骤不请求模型")
    pools = retriever.restore_candidates(directory, visual_segments, assets, gateway, verify, stop=stop)
    boundaries, fingerprint_boundaries = {}, {}
    search_round = int(saved.get("search_round", 0))

    def merge(identifier, values):
        unique = {}
        for c in pools[identifier]+values:
            if c.get("_rejected") or not retriever.range_allowed(c):
                continue
            physical = source_shot_key(c)
            rank = lambda row: (not row.get("pending_review", True), len(row.get("coverage",{}).get("supports",[])),
                                row.get("confidence", row.get("proposal_score", 0)), row["source_end"]-row["source_start"])
            if physical not in unique or rank(c) > rank(unique[physical]):
                unique[physical] = c
        pools[identifier] = sorted(unique.values(), key=lambda c: (not c.get("pending_review", True),
                                     c.get("confidence", c.get("proposal_score", 0))), reverse=True)

    for u in units:
        identifier = u["id"]
        previous = saved.get("pools", {}).get(identifier, [])
        merge(identifier, retriever.prepare(visual_segments[identifier], previous, assets, gateway, verify))
        processed[identifier].update(c["_review_key"] for c in pools[identifier] if c["_review_key"] in retriever.candidate_progress)
    current_keys = retriever.recorded | {c["_review_key"] for rows in pools.values() for c in rows}
    retriever.candidate_progress = {k: v for k, v in retriever.candidate_progress.items() if k in current_keys}

    def ensure_boundaries(identifiers):
        for identifier in dict.fromkeys(identifiers):
            if identifier in boundaries:
                continue
            asset = asset_map[identifier]
            fingerprint = asset["fingerprint"]
            if fingerprint not in fingerprint_boundaries:
                cache = retriever.store.directory(asset["library_id"])/"frames"/identifier/"shot-boundaries.json"
                fingerprint_boundaries[fingerprint] = scene_boundaries(asset, cache, stop, diagnostic_directory=directory)
            boundaries[identifier] = fingerprint_boundaries[fingerprint]

    def candidate_for_shot(shot):
        return next((c for c in pools.get(shot.visual_unit_id, []) if c["asset_id"] == shot.asset_id and c["scene_id"] == shot.scene_id
            and c["source_start"] <= shot.source_start+.000001 and c["source_end"] >= shot.source_end-.000001), None)

    def defer(identifier, candidate, owners, reason):
        previous = retriever.candidate_progress.get(candidate["_review_key"], {})
        retriever.candidate_progress[candidate["_review_key"]] = {"version": CANDIDATE_QUEUE_VERSION,
            "model_signature": retriever.queue_signature, "unit_id": identifier, "state": "deferred", "stage": "allocation",
            "occupancy": occupancy, "occupied_by": owners, "reason": reason,
            "preselected": bool(candidate.pop("proposal_version", None) or previous.get("preselected"))}

    def waiting(identifier):
        result = []
        for candidate in pools[identifier]:
            key = candidate["_review_key"]
            if not candidate.get("pending_review") or key in retriever.unconfirmed:
                continue
            progress = retriever.candidate_progress.get(key, {})
            if progress.get("state") == "deferred" and progress.get("occupancy") == occupancy:
                continue
            owners = [{"unit_id": owner, "source": source_shot_key(other)} for owner, other in occupied
                      if owner != identifier and not unused_source(candidate, [other])]
            if owners:
                defer(identifier, candidate, owners, "当前来源由其他单元临时占用，先取其他候选。")
                continue
            if progress.get("state") == "deferred":
                if progress.get("preselected"):
                    candidate["proposal_version"] = PROPOSAL_VERSION
                retriever.candidate_progress.pop(key, None)
            result.append(candidate)
        return result

    def fill_local(unit):
        identifier = unit["id"]
        started, scanned, before = time.monotonic(), 0, len(retriever.evaluated[identifier])
        while identifier not in exhausted:
            check_cancel(stop)
            batch = retriever.retrieve(unit["requirement"], limit=4, excluded=examined[identifier])
            if not batch:
                exhausted.add(identifier)
                break
            batch = [c for c in batch if c["scene_id"] not in examined[identifier]]
            if not batch:
                return "stalled"
            examined[identifier].update(c["scene_id"] for c in batch)
            scanned += len(batch)
            update("寻找画面", 47, f"本地整理{identifier}新增{len(batch)}个窗口及切镜缓存；未请求模型")
            ensure_boundaries(c["asset_id"] for c in batch)
            diagnostic = {"windows": len(batch)}
            previous_keys = {c["_review_key"] for c in pools[identifier]}
            candidates = retriever.prepare(visual_segments[identifier], batch, assets, gateway, verify,
                scene_boundaries=boundaries, diagnostic=diagnostic)
            diagnostic["existing_ranges"] = sum(c["_review_key"] in previous_keys for c in candidates)
            merge(identifier, candidates)
            for name, value in diagnostic.items():
                local_search[identifier][name] = local_search[identifier].get(name, 0)+value
            update("寻找画面", 47, f"本地续查{identifier}：本次{scanned}个窗口，累计{len(examined[identifier])}个；未请求模型")
            if waiting(identifier):
                return "pending"
            if len(retriever.evaluated[identifier]) > before:
                return "evaluated"  # Reallocate before settling new exact-cache judgments.
            if scanned >= 32 or time.monotonic()-started >= 5:
                return "yielded"  # Local work checkpoint, never an API limit or an exhausted query.
        return "exhausted"

    def save(status, gaps, proposal):
        write_json(path, {"version": PROPOSAL_VERSION, "key": key, "pools": {uid:[{k:v for k,v in c.items() if k!="_requirement"} for c in rows] for uid,rows in pools.items()},
            "examined": {k: sorted(v) for k, v in examined.items()}, "exhausted": sorted(exhausted),
            "no_gain": no_gain, "search_round": search_round, "migrated_from_key": migrated_from,
            "search_signature": search_signature, "unit_signatures": unit_signatures,
            "migration": migration or saved.get("migration"), "evaluation_version": RETRIEVAL_REPAIR_VERSION,
            "evaluation_waves": waves, "paused_units": paused_units, "queue_version": CANDIDATE_QUEUE_VERSION,
            "candidate_progress": retriever.candidate_progress, "pause_dependencies": pause_dependencies,
            "uncertain_waves": uncertain_waves, "local_search": local_search, "review_protocol":REVIEW_PROTOCOL})
        write_json(directory/"visual_proposal.json", {"version": PROPOSAL_VERSION, "plan_key": visual_plan["key"],
            "approved_for_render": False, "shots": [vars(shot) for shot in proposal], "gaps": gaps})
        write_json(directory/"visual_matches.json", {"version": MATCH_POLICY_VERSION, "plan_key": visual_plan["key"],
            "state": status, "timing_method": visual_plan["timing_method"], "search_policy": PROPOSAL_VERSION,
            "excluded_windows": len(retriever.rejected_scenes), "excluded_ranges": len(retriever.rejected_ranges),
            "units": {u["id"]: {"examined_windows": len(examined[u["id"]]),
                "candidate_count": sum(c.get("pending_review") is False for c in pools[u["id"]]),
                "pending_candidates": sum(c.get("pending_review") is True for c in pools[u["id"]]),
                "exhausted": u["id"] in exhausted, "no_gain_rounds": no_gain[u["id"]],
                "evaluation_wave": waves.get(u["id"]), "pause_reason": paused_units.get(u["id"]),
                "processed_candidates": len(processed[u["id"]]),
                "compatible_candidates": sum(bool(c.get("compatible_from")) for c in pools[u["id"]] if c.get("pending_review") is False),
                "preselection": retriever.preselection.get(u["id"], {}),
                **retriever.statistics.get(u["id"], {}), "local_search": local_search[u["id"]],
                "unconfirmed": sum(c["_review_key"] in retriever.unconfirmed for c in pools[u["id"]]),
                "deferred_candidates": sum(c.get("state") == "deferred" and c.get("unit_id") == u["id"]
                    for c in retriever.candidate_progress.values())} for u in units}, "gaps": gaps})

    structural = [{"visual_unit_id": u["id"], "segment_id": u["segment_id"], "start": a, "end": b,
                   "category": "short_interval", "missing_seconds": round(b-a, 3), "reason": "动画与句意边界留下不足1秒原片区间。"}
                  for u in units for a, b in visual_unit_intervals(u, reservations)
                  if b-a < min(1.0, visual_segments[u["id"]].audio_end-visual_segments[u["id"]].audio_start)-.001]
    if structural:
        for gap in structural:
            nearby = {s.asset_id for s in reservations if abs(s.timeline_end-gap["start"]) < .001 or
                      abs(s.timeline_start-gap["end"]) < .001}
            sources = {row["source"] for row in alignment.get("motions", []) if row["animation_id"] in nearby}
            gap["animation_sources"] = sorted(sources)
            reasons = [row["reason"] for row in alignment.get("rejected", []) if
                       gap["visual_unit_id"] in row.get("unit_ids", []) or row.get("animation_id") in nearby or
                       ("unit_ids" not in row and "animation_id" not in row)]
            if not reasons:
                reasons = ["父段或单元边界与动画原文绑定不一致，无法安全自动归并。"]
            gap["reason"] = "；".join(dict.fromkeys(reasons))
        proposal, _ = propose_visual_timeline(segments, pools, target_shot_seconds=target, visual_units=units, reservations=reservations)
        save("incomplete", structural, proposal)
        examples = "、".join(row["visual_unit_id"] for row in structural[:8])
        reasons = "；".join(dict.fromkeys(row["reason"] for row in structural))
        origins = {origin for row in structural for origin in row["animation_sources"]}
        source_note = "、".join(label for value, label in (("manual", "手动"), ("auto", "自动")) if value in origins) or "未确认来源的"
        raise JobPaused(f"发现{len(structural)}处不足1秒的原片区间（{examples}），已在候选模型复核前暂停。"
                        f"相邻{source_note}动画未能安全归并：{reasons}"
                        "请查看画面缺口详情，调整对应动画原文范围或合并句意范围后从3. 寻找画面继续；"
                        "原配音、字幕、合格候选与旧成片保留。")

    # Prepare all units locally before reserving scarce sources; this makes no provider calls.
    for index, unit in enumerate(units):
        check_cancel(stop)
        if visual_unit_intervals(unit, reservations) and not pools[unit["id"]] and not examined[unit["id"]]:
            fill_local(unit)
        update("寻找画面", 47, f"复用已付费结果并本地整理候选 {index+1}/{len(units)}")
    needed_seconds = sum(b-a for u in units for a, b in visual_unit_intervals(u, reservations))
    while True:
        check_cancel(stop)
        for identifier in pools:
            merge(identifier, [])
        try:
            shots = plan_visual_timeline(segments, pools, target_shot_seconds=target, visual_units=units, reservations=reservations)
            ensure_boundaries(shot.asset_id for shot in shots if shot.visual_unit_id)
            save("complete", [], shots)
            return shots, boundaries, pools
        except VisualCapacityError as exc:
            gaps = exc.gaps
        confirmed, _ = propose_visual_timeline(segments, {k: [c for c in rows if c.get("pending_review") is False] for k, rows in pools.items()},
                                             target_shot_seconds=target, visual_units=units, reservations=reservations)
        occupied = [(shot.visual_unit_id, candidate_for_shot(shot)) for shot in confirmed if shot.visual_unit_id]
        occupied = [(owner, candidate) for owner, candidate in occupied if candidate is not None]
        occupancy = stable_hash(sorted((owner, *source_shot_key(c), c["source_start"], c["source_end"]) for owner, c in occupied))
        for identifier, dependency in list(pause_dependencies.items()):
            if dependency != occupancy:
                paused_units.pop(identifier, None)
                pause_dependencies.pop(identifier)
        # Check availability both before ranking and after other approvals change allocation.
        for identifier in waves:
            available = {c["_review_key"] for c in waiting(identifier)}
            for candidate in pools[identifier]:
                if candidate["_review_key"] in waves[identifier]["keys"] and candidate["_review_key"] not in available:
                    candidate.pop("proposal_version", None)
        proposal_pools = {identifier: [c for c in rows if c.get("pending_review") is False or
                          (c["_review_key"] in waves.get(identifier, {}).get("keys", []) and
                           c["_review_key"] not in retriever.unconfirmed and identifier not in paused_units and
                           retriever.candidate_progress.get(c["_review_key"], {}).get("state") != "deferred")]
                          for identifier, rows in pools.items()}
        proposal, _ = propose_visual_timeline(segments, proposal_pools, target_shot_seconds=target, visual_units=units, reservations=reservations)
        covered = sum(s.timeline_end-s.timeline_start for s in confirmed if s.visual_unit_id)
        allocated = {u["id"]: sum(s.timeline_end-s.timeline_start for s in confirmed if s.visual_unit_id == u["id"]) for u in units}
        allocated_facts = {u["id"]:len({identifier for shot in confirmed if shot.visual_unit_id == u["id"] for identifier in shot.fact_ids}) for u in units}
        progress = 47+22*min(1, covered/max(.001, needed_seconds))
        pair_keys, pairs = set(), []
        for shot in proposal:
            identifier = shot.visual_unit_id
            if not identifier or identifier not in waves or identifier in paused_units:
                continue
            candidate = candidate_for_shot(shot)
            if candidate and candidate.get("pending_review") and candidate["_review_key"] not in retriever.unconfirmed:
                pair = (identifier, candidate["_review_key"])
                if pair not in pair_keys:
                    pair_keys.add(pair); pairs.append((visual_segments[identifier], candidate))
        save("searching", gaps, proposal)
        if pairs:
            batch = [pair for pair in pairs if pair[0].id == pairs[0][0].id][:6]
            update("寻找画面", progress, f"确认拟用镜头{len(batch)}个；当前{len(gaps)}处缺口，已确认可分配画面{covered:.1f}秒")
            retriever.review_pairs(batch, gateway, directory, stop, search_round=search_round)
            for segment, _ in batch:
                merge(segment.id, [])
            continue
        material = {r["visual_unit_id"] for r in gaps if r.get("category") != "short_interval"}
        for identifier, wave in waves.items():
            processed[identifier].update(wave["keys"])
            if identifier not in material:
                no_gain[identifier] = 0
                continue
            unresolved = set(wave["keys"])-retriever.evaluated[identifier]
            for candidate in pools[identifier]:
                if candidate["_review_key"] in unresolved and candidate["_review_key"] not in retriever.unconfirmed:
                    entry = retriever.candidate_progress.get(candidate["_review_key"], {})
                    if entry.get("state") != "deferred" or entry.get("occupancy") != occupancy:
                        defer(identifier, candidate, [], "本轮未形成可分配镜头，先取其他候选；占用变化后重新核对。")
            if unresolved & retriever.unconfirmed:
                uncertain_waves[identifier] += 1
                if uncertain_waves[identifier] >= no_gain_limit:
                    paused_units[identifier] = f"连续{no_gain_limit}批实际审核未确认，相关候选已保存待后续处理"
            elif not unresolved and len(retriever.evaluated[identifier]) > wave["evaluated_before"]:
                uncertain_waves[identifier] = 0
                improved = allocated[identifier] > wave["allocated_before"]+.001 or allocated_facts[identifier] > wave.get("facts_before",0)
                no_gain[identifier] = 0 if improved else no_gain[identifier]+1
            elif not wave["keys"]:
                paused_units[identifier] = "预选输入或协议未完成" if wave["protocol_issues"] else "本轮没有可审候选且无新有效结论"
        waves = {}
        groups = []
        for unit in units:
            identifier = unit["id"]
            if identifier not in material or identifier in paused_units or no_gain[identifier] >= no_gain_limit:
                continue
            before = len(retriever.evaluated[identifier])
            candidates = waiting(identifier)
            local_state = None
            if not candidates:
                local_state = fill_local(unit)
                candidates = waiting(identifier)
            if not candidates:
                if len(retriever.evaluated[identifier]) > before:
                    waves[identifier] = {"keys": [], "evaluated_before": before,
                        "allocated_before": allocated[identifier], "facts_before":allocated_facts[identifier], "protocol_issues": 0}
                elif local_state == "yielded":
                    paused_units[identifier] = "本地续查已保存游标让出，查询尚未耗尽"
                elif local_state == "stalled":
                    paused_units[identifier] = "本地检索未推进，已保存现有候选"
                else:
                    deferred = any(c.get("unit_id") == identifier and c.get("state") == "deferred" and c.get("occupancy") == occupancy
                                   for c in retriever.candidate_progress.values())
                    paused_units[identifier] = "当前可检索备选均被占用或尚未可分配，已保存候选队列" if deferred else "当前检索查询已无新窗口，未确认范围保留"
                    if deferred:
                        pause_dependencies[identifier] = occupancy
                continue
            waves[identifier] = {"keys": [], "evaluated_before": before,
                "allocated_before": allocated[identifier], "facts_before":allocated_facts[identifier], "protocol_issues": 0}
            groups.append((visual_segments[identifier], candidates, sum(row["missing_seconds"] for row in gaps if row["visual_unit_id"] == identifier)))
        if not groups and not waves:
            save("incomplete", gaps, proposal)
            unconfirmed = sum(c["_review_key"] in retriever.unconfirmed for rows in pools.values() for c in rows)
            held = sum(no_gain[identifier] >= no_gain_limit for identifier in material)
            short = sum(row.get("category") == "short_interval" for row in gaps)
            fact_gaps = [row for row in gaps if row.get("category") in {"fact_coverage","fact_order"}]
            protocol = sum(row.get("invalid_references", 0) > 0 for row in retriever.preselection.values())
            pending = sum(c.get("pending_review") is True for rows in pools.values() for c in rows)
            deferred = sum(row.get("state") == "deferred" for row in retriever.candidate_progress.values())
            reasons = "；".join(f"{reason}{list(paused_units.values()).count(reason)}个单元" for reason in dict.fromkeys(paused_units.values()))
            raise JobPaused(f"画面仍有{len(gaps)}处缺口，共缺{sum(r['missing_seconds'] for r in gaps):.2f}秒；"
                f"{held}个单元完成有效评估后连续{no_gain_limit}轮无新增可分配时长或事实覆盖，{short}处为短残段，{len(fact_gaps)}处为事实覆盖/顺序不足，"
                f"预选协议或编号问题涉及{protocol}个单元，实际镜头未确认{unconfirmed}个，候选队列待审{pending}个，分配延后{deferred}个。"
                f"{reasons + '。' if reasons else ''}"
                + ("事实缺项："+"；".join(row["visual_unit_id"]+"："+row["reason"] for row in fact_gaps[:6])+"。" if fact_gaps else "") +
                "本次按有效评估及候选进度暂停；不代表全库一定没有素材。已有配音、字幕、确认结果和旧成片保留，"
                "从3. 寻找画面继续会复用已付费结果，沿候选队列及保存的本地检索游标向后推进。")
        if groups:
            update("寻找画面", progress, f"按需文字预选{len(groups)}个缺口单元；已本地跳过占用范围，当前已确认{covered:.1f}秒")
            for identifier, result in retriever.shortlist(groups, gateway, directory, stop, search_round=search_round).items():
                waves[identifier].update(result)
                for review_key in result["keys"]:
                    retriever.candidate_progress[review_key] = {"version": CANDIDATE_QUEUE_VERSION,
                        "model_signature": retriever.queue_signature, "unit_id": identifier, "state": "selected", "stage": "text_preselect"}
        search_round += 1
        save("searching", gaps, proposal)


def _run_task(task_id, model_profile, voice_profile, stop, use_current_models, *, start_stage="auto"):
    from app.models.timeline import timeline_dict
    from app.services.narration_audio import (
        create_narration_subtitles, finalize_long_narration, synthesize_long_narration, synthesize_narration,
    )
    from app.services.narration_planner import fit_single_line_units, subtitle_text_ranges
    from app.services.timeline_planner import (
        validate_timeline, coalesce_motion_shots, make_visual_policy,
    )
    from app.services.timeline_renderer import render_timeline

    directory = task_directory(task_id)
    settings = read_json(directory / "settings.json")
    with (directory / "input.txt").open("r", encoding="utf-8", newline="") as manuscript:
        text = manuscript.read()
    state = read_task(task_id)
    level = _resume_level(start_stage)
    options = motion_options(bgm_options(normalized_audio_options(settings["options"]) if level <= 2 else dict(settings["options"])))
    mode = audio_mode(options)
    checkpoint = _resume_checkpoint(directory, settings, text, level) if level else {}
    required = ["text"] if level <= 3 or (level == 4 and options["bgm_mode"] == "auto" and options["bgm_volume"] > 0) else []
    if level == 0 or (level == 2 and mode == LONG_MODE):
        required.append("text_fast")
    if level <= 3 and "text_fast" not in required:
        required.append("text_fast")
    # The relevance switch does not disable the independent credits/usability check.
    if level <= 3:
        required.append("vision")
    if required:
        if "models" not in settings or use_current_models:
            refreshed = public_models(model_profile)
            previous_bindings = settings.get("models", {}).get("_bindings", {})
            bindings = dict(refreshed.get("_bindings", {}))
            for role in ("tts", "alignment"):
                if previous_bindings.get(role):
                    bindings[role] = previous_bindings[role]
                else:
                    bindings.pop(role, None)
            refreshed["_bindings"] = bindings
            settings["models"] = refreshed
            write_json(directory / "settings.json", settings)
        if "text_fast" not in settings["models"]:
            settings["models"]["text_fast"] = public_models({"text_fast": model_profile.get("text_fast", model_profile.get("text", {}))})["text_fast"]
            write_json(directory / "settings.json", settings)
        missing_backups = [kind for kind in ("text_backup", "vision_backup") if kind not in settings["models"]]
        if missing_backups:
            settings["models"].update(public_models({kind: model_profile.get(kind, {}) for kind in missing_backups}))
            write_json(directory / "settings.json", settings)
        model_profile = restore_models(settings["models"], model_profile, required_kinds=required)
    else:
        model_profile = {}
    if level <= 2 and mode == LONG_MODE and options["tts_chunking"] == "manual":
        from app.services.narration_audio import manual_audio_ranges
        if options.get("tts_break_text_hash") != stable_hash(text):
            raise ValueError("任务保存的手动配音断点与原文不一致。")
        manual_audio_ranges(text, options.get("tts_break_after"))
    state["audio_mode"] = mode
    state["resume_stage"] = start_stage
    if stable_hash(text) != settings["text_hash"]:
        raise ValueError("任务原文发生变化，请创建新任务。")
    # Restore the recorded voice configuration, refreshing only authentication/proxy.
    if level <= 1:
        voice_profile = restore_voice(settings["voice"], voice_profile, profile_id=settings.get("models", {}).get("_bindings", {}).get("tts"))
    cache_source = task_directory(settings["parent_task_id"]) / "audio" if level <= 1 and settings.get("parent_task_id") else None
    model_calls = 0
    usage = ModelUsage(directory)
    state["token_usage"] = usage.snapshot()

    def update(phase, progress, message):
        state.update(state="running", phase=phase, progress=round(progress, 1), message=message, model_calls=model_calls)
        write_json(directory / "state.json", state)
        gateway.context["phase"] = phase

    def before_request():
        nonlocal model_calls
        check_cancel(stop)
        model_calls += 1

    def diagnostic(record):
        entry = {"time": time.time(), "task_id": task_id, "phase": state.get("phase", ""), **record}
        with (directory / "model-diagnostics.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")
        usage.record(entry)
        state.update(token_usage=usage.snapshot(), model_calls=model_calls)
        state["last_model_request"] = entry
        write_json(directory / "state.json", state)

    def record_request(request):
        request_id = uuid4().hex
        relative_path = f"model-requests/{request_id}.json"
        write_json(directory / relative_path, {"schema": 1, "request_id": request_id,
                                             "time": time.time(), "task_id": task_id, **request})
        return relative_path

    gateway = ModelGateway(model_profile, before_request=before_request, diagnostic=diagnostic, request_recorder=record_request)
    try:
        if level <= 3:
            _preserve_visual_result(directory)
        if level <= 4:
            validate_bgm(options)
        if level <= 3:
            validate_motion(options)
            manual_motion_entries(text, options, complete=True)
        if level <= 2 and mode == LONG_MODE and options["alignment_enabled"]:
            validate_alignment_options(options)
        for kind in required:
            gateway.require(kind)
        store = LibraryStore()
        selected_libraries = library_ids(settings)
        if level <= 3:
            update("校验字幕资料", 0 if level == 0 else 40, "本地检查正文资料；异常资料修复前不发起模型找片")
            subtitle_issues = [row for identifier in selected_libraries for row in audit_subtitles(store, identifier, stop)
                               if row["state"] in {"recovery_needed", "error"}]
            if subtitle_issues:
                write_json(directory/"subtitle_quality.json", {"state": "blocked", "issues": subtitle_issues})
                raise JobPaused("视频素材库字幕正文校验未通过：" + "；".join(
                    f"第{r['episode']}集：" + (r.get("error") or f"缺少{r.get('missing_body_events', 0)}条正文") for r in subtitle_issues) +
                    "。请到视频素材库→已登记素材→字幕资料校验与恢复，点“校验字幕”及“恢复正文与本地索引”，再从3. 寻找画面继续。配音、字幕、动画及旧成片保留。")
        retriever = MaterialRetriever(store, selected_libraries) if level <= 3 else None
        if level == 0:
            update("分析文案", 0, "保留原文，分析全文结构和画面需求")
            overview, segments = plan_document(text, gateway, directory, stop,
                                              progress=lambda p, message: update("分析文案", p * 15, message))
        else:
            overview, segments = checkpoint["overview"], checkpoint["segments"]
            update("恢复阶段", (20, 33, 40, 74, 80)[level - 1], RESUME_STAGES[start_stage] + "，复用前序结果")
        if level == 0 or (level == 2 and mode == LONG_MODE):
            update("字幕分段", 15 if level == 0 else 33, "使用高效率文案模型选择字幕边界")
            subtitle_units = plan_subtitles(segments, gateway, directory, stop,
                                            progress=lambda p, message: update("字幕分段", (15 + p * 5) if level == 0 else (33 + p), message))
        elif level == 1:
            subtitle_units = checkpoint["subtitle_units"]
        check_cancel(stop)
        if level <= 1:
            budget = max(1, int(options.get("tts_budget", DEFAULT_CALL_BUDGET)))
        if level <= 2 and mode == LONG_MODE:
            if level <= 1:
                update("生成长段配音", 20, "按完整论述段组合配音块，保留自然停顿")
                blocks = synthesize_long_narration(segments, text, voice_profile, directory, stop,
                    progress=lambda p, message: update("生成长段配音", 20 + p * 13, message),
                    tts_budget=budget, target_chars=options["tts_block_chars"], cache_source=cache_source,
                    break_after=options.get("tts_break_after") if options["tts_chunking"] == "manual" else None)
            else:
                blocks = checkpoint["blocks"]
            if options["alignment_enabled"]:
                state["subtitle_timing"] = "qwen_forced_alignment"
                update("字幕对齐", 33, "使用原文定位字词时间，复用已完成的对齐块")
                aligned = align_blocks(blocks, options, directory, stop,
                    progress=lambda p, message: update("字幕对齐", 33 + p * 7, message))
            else:
                state["subtitle_timing"] = "text_estimate"
                update("估算字幕时间", 33, "本地对齐已关闭，按实际音频时长和文字权重估算字幕时间")
                aligned = estimate_block_words(blocks, stop)
            check_cancel(stop)
            update("字幕排版", 39, "根据当前字体、字号、行宽和字词时间规划单行字幕")
            cues = map_timing(segments, blocks, subtitle_text_ranges(segments, subtitle_units), text, aligned,
                              options=options, stop=stop)
            if options.get("subtitles", True) and options.get("subtitle_single_line", False):
                short_cues = [cue for cue in cues if cue.get("short")]
                layout_report = {"version": 1, "font": options.get("font"), "font_size": options.get("font_size", 72),
                                 "aspect": options.get("aspect", "landscape"), "width_ratio": options.get("subtitle_width_ratio", .7),
                                 "minimum_seconds": options.get("subtitle_min_seconds", 1.0), "cue_count": len(cues),
                                 "short_count": len(short_cues), "cues": cues}
                write_json(directory / "subtitle_layout.json", layout_report)
                state.update(subtitle_cues=len(cues), subtitle_short_count=len(short_cues))
                if short_cues and not options.get("subtitle_allow_short", False):
                    raise JobPaused(f"当前字号、行宽和原配音时间下有{len(short_cues)}条字幕未达阅读目标。"
                                    "配音与字幕排版结果已保留；请查看结果，用样式副本调整字号、行宽或阅读目标，"
                                    "也可在输出设置中明确允许短字幕后创建副本。")
            audio_path = finalize_long_narration(segments, blocks, cues, aligned, directory)
            state.update(audio_blocks=len(blocks), subtitle_cues=len(cues))
        elif level <= 1:
            subtitle_units = fit_single_line_units(subtitle_units, options)
            audio_path = synthesize_narration(segments, voice_profile, directory, stop,
                progress=lambda p, message: update("生成配音", 20 + p * 20, message),
                tts_budget=budget, subtitle_units=subtitle_units, cache_source=cache_source)
        else:
            audio_path = checkpoint["audio_path"]
            state.update(subtitle_timing=checkpoint["audio"].get("timing_method", "tts_units"),
                         subtitle_cues=len(checkpoint["audio"]["cues"]))
        subtitle_path = str(directory / "subtitles.srt")
        if level <= 2 or (options.get("subtitles", True) and not Path(subtitle_path).is_file()):
            subtitle_path = create_narration_subtitles(segments, subtitle_path)
        if level <= 3:
            motion_plan, motion_assets, reservations = None, [], []
            if options["motion_enabled"]:
                update("规划解释动画", 40, "按原文绑定手动动画与自动候选，使用已保存的旁白时间" if options.get("motion_placement", "auto") != "auto" else
                       "根据原文句子选择动画位置，使用已保存的旁白时间")
                motion_plan = plan_motion(segments, text, overview, gateway, directory, options, stop,
                    progress=lambda p, message: update("规划解释动画", 40 + p * 2, message))
                update("渲染解释动画", 42, "本地渲染无声动画片段，复用已完成缓存")
                motion_cache_source = task_directory(settings["parent_task_id"]) / "motion" if settings.get("parent_task_id") else None
                motion_assets, reservations = prepare_motion(motion_plan, directory, options, stop,
                    progress=lambda p, message: update("渲染解释动画", 42 + p * 3, message), cache_source=motion_cache_source)
                state.update(motion_count=len({s.asset_id for s in reservations}), motion_seconds=round(sum(s.timeline_end-s.timeline_start for s in reservations), 2))
            else:
                state.update(motion_count=0, motion_seconds=0)
            visual_start = 45 if options["motion_enabled"] else 40
            update("规划句意画面", visual_start, "保留原配音与字幕，按当前句意确定画面范围")
            try:
                previous_visual_plan = validate_visual_plan(read_json(directory/"visual_plan.json"), segments, text, directory)
            except (OSError, ValueError, KeyError, TypeError):
                previous_visual_plan = None
            visual_plan = plan_visual_units(segments, text, overview, gateway, directory, stop,
                progress=lambda p, message: update("规划句意画面", visual_start + p*2, message),
                reservations=reservations, motion_plan=motion_plan)
            visual_plan = revise_visual_requirements(visual_plan, previous_visual_plan, text, gateway, directory, stop,
                progress=lambda message: update("修订缺口画面需求", visual_start + 2, message))
            validate_visual_plan(visual_plan, segments, text, directory)
            assets = store.assets(selected_libraries) + motion_assets
            shots, boundaries, candidates_by_unit = _find_visual_shots(
                segments, text, visual_plan, assets, reservations, retriever, gateway, directory, options, stop, update,
                previous_plan=previous_visual_plan)
            used = []
            for shot in shots:
                if not shot.visual_unit_id:
                    continue
                selected = next(item for item in candidates_by_unit[shot.visual_unit_id] if item.get("pending_review") is False and item["asset_id"] == shot.asset_id
                    and item["scene_id"] == shot.scene_id and item["source_start"] <= shot.source_start+.000001
                    and item["source_end"] >= shot.source_end-.000001)
                used.append({key: selected[key] for key in ("asset_id", "scene_id", "fingerprint", "fps", "natural_start", "natural_end",
                    "source_start", "source_end", "usable", "content_kind", "visual_policy", "match_policy", "visual_unit_id",
                    "relation", "facts_match", "evidence", "conflicts", "subtitle_quotes", "visual_checked", "confidence",
                    "subtitle_ids", "visual_evidence", "evidence_protocol")})
                used[-1]["review_key"] = selected["_review_key"]
                for key in ("coverage", "review_protocol", "compatible_from"):
                    if key in selected:
                        used[-1][key] = selected[key]
            update("编排镜头", 70, "校验句意对应、全片来源去重与实际范围画面检查记录")
            policy = make_visual_policy(assets, boundaries, used, retriever.forbidden, visual_plan=visual_plan)
            validate_timeline(segments, shots, assets, policy=policy)
            timeline = timeline_dict(segments, shots)
            timeline["visual_policy"] = policy
            if motion_plan is not None:
                timeline["motion"] = {"version": 1, "plan_key": motion_plan["key"],
                                      "active_ids": list(dict.fromkeys(shot.asset_id for shot in reservations)), "timing_method": motion_plan["timing_method"]}
                if "manual_config_key" in motion_plan:
                    timeline["motion"].update(manual_config_key=motion_plan["manual_config_key"], placement=motion_plan["placement"])
            write_json(directory / "timeline.json", timeline)

        else:
            shots, timeline = checkpoint["shots"], checkpoint["timeline"]
            assets = store.assets(selected_libraries) + load_motion_assets(directory, shots, verify=True,
                plan_key=timeline.get("motion", {}).get("plan_key"))
            validate_timeline(segments, shots, assets, policy=timeline["visual_policy"])
        render_options = dict(options)
        render_start = 80 if level == 5 else 75
        if level <= 4 and options["bgm_mode"] == "auto" and options["bgm_volume"] > 0:
            update("自动配乐", 74, "根据大段主题和情绪选择已分析音乐，不重新上传音乐")
            music = plan_bgm(segments, overview, gateway, directory, options, stop,
                progress=lambda p, message: update("自动配乐", 74 + p * 2, message))
            state.update(bgm_music_count=music["music_count"], bgm_intervals=len(music["cues"]))
            update("制作配乐", 76, "编排音乐、交叉淡化，并根据旁白音量压低音乐")
            render_options["bgm_path"] = prepare_bgm(music, audio_path, directory, options, stop,
                progress=lambda p, message: update("制作配乐", 76 + p * 4, message))
            render_options["bgm_prepared"] = True
            timeline["bgm"] = music
            write_json(directory / "timeline.json", timeline)
            render_start = 80
        elif level == 5 and "music" in checkpoint:
            music = checkpoint["music"]
            render_options.update(bgm_path=str(directory / "bgm.wav"), bgm_prepared=True)
            state.update(bgm_music_count=music["music_count"], bgm_intervals=len(music["cues"]))
            render_start = 80
        video = render_timeline(segments, coalesce_motion_shots(shots), assets, audio_path, subtitle_path, directory, render_options, stop,
                                progress=lambda p, message: update("合成视频", render_start + p * (99 - render_start), message))
        weak_count = sum(shot.weak_match for shot in shots)
        state.update(state="complete", progress=100, phase="完成", message=f"生成完成，共 {len(shots)} 个镜头，{weak_count} 个镜头需关注匹配详情。",
                     video=video, weak_matches=weak_count, model_calls=model_calls)
    except (JobCancelled, JobPaused) as exc:
        state.update(state="paused", message=str(exc), model_calls=model_calls)
    except Exception as exc:
        # Gateway errors omit credentials and reference payloads.
        state.update(state="failed", message=str(exc), model_calls=model_calls)
    write_json(directory / "state.json", state)
