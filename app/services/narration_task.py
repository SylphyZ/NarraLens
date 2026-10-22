"""Persistent, resumable manuscript-to-video jobs; API credentials stay in memory."""

import json
import os
import shutil
import time
import re
import wave
from pathlib import Path
from copy import copy, deepcopy
from queue import Empty, SimpleQueue
from uuid import UUID, uuid4

from loguru import logger

from app.services.library.gateway import ModelGateway, ModelUsage
from app.services.library.retriever import MaterialRetriever
from app.models.library import RETRIEVAL_PROTOCOL
from app.services.library.runner import (
    DEFAULT_CALL_BUDGET, AnalysisCoordinator, JobCancelled, api_concurrency, check_cancel, library_operation, run_analysis_jobs,
)
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
from app.services.narration_agent import visual_backend, project_lock
from app.services.model_profiles import (
    catalog, default_bindings, get_profile, production_voice, project_bindings, project_references,
    public_values, resolve_project, validate_narration_voice,
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
        agent = policy.get("producer", "internal") == "agent"
        if agent and policy.get("agent", {}).get("project_id") != directory.name:
            raise ValueError("Agent时间轴属于其他项目，请在本项目重新创建并导入画面任务。")
        if not agent and policy.get("matching", {}).get("version") != MATCH_POLICY_VERSION:
            raise ValueError("此时间轴尚未执行2.7句意匹配，请从3. 寻找画面继续；配音与字幕可复用。")
        if not agent:
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
    with project_lock(identifier), library_operation(identifier, prefix="narration:"), library_operation(library_ids(settings) or library_ids(state) or identifier):
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


def _write_project_settings(directory, previous, updated):
    """Archive the old public identity before the settings commit point, once."""
    history = directory / "model-history.json"
    if previous.get("model_reference_schema") != 1 and not history.exists():
        write_json(history, {"schema": 1, "time": time.time(), "settings": public_values(previous)})
    write_json(directory / "settings.json", updated)


def update_project_bindings(task_id, bindings):
    """A stopped project's role IDs may change without unlocking production inputs."""
    identifier = str(UUID(str(task_id)))
    directory = task_directory(identifier)
    with project_lock(identifier), library_operation(identifier, prefix="narration:"):
        if runner.running("narration:" + identifier):
            raise ValueError("项目正在运行，请停止并等待结束后保存模型绑定。")
        _checked_task_directory(identifier)
        saved = read_json(directory / "settings.json")
        registry = catalog()
        old = project_bindings(saved, registry=registry)
        for role, value in bindings.items():
            if value and value != old.get(role):
                get_profile(value, role, registry=registry)
        updated = project_references(saved, bindings)
        _write_project_settings(directory, saved, updated)
        return updated


def _task_options(text, options, bindings, *, draft=False, reuse_audio=False):
    """Drafts check data; generation also checks bindings and local resources."""
    from app.services.narration_audio import manual_audio_ranges
    if not isinstance(options, dict):
        raise ValueError("项目设置应为对象。")
    result = motion_options(bgm_options(normalized_audio_options(options)))
    result["visual_backend"] = visual_backend(result)
    value = result.get("visual_no_gain_rounds", 2)
    if type(value) is not int or value < 1:
        raise ValueError("连续无改善轮数需为正整数。")
    result["visual_no_gain_rounds"] = value
    result["visual_review_concurrency"] = api_concurrency(result.get("visual_review_concurrency", 4))
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
            result.update(resolve_project({"model_reference_schema": 1, "model_bindings": bindings}, need_alignment=True)["alignment"])
            validate_alignment_options(result)
        manual_motion_entries(text, result)
    return result


def update_visual_backend(task_id, backend):
    """A stopped project can change its visual editor without unlocking narration settings."""
    backend = visual_backend({"visual_backend": backend})
    identifier = str(UUID(str(task_id)))
    directory = task_directory(identifier)
    with project_lock(identifier), library_operation(identifier, prefix="narration:"):
        if runner.running("narration:" + identifier):
            raise ValueError("请停止项目并等待结束后修改画面匹配方式。")
        _checked_task_directory(identifier)
        saved = read_json(directory / "settings.json")
        with library_operation(library_ids(saved)):
            updated = {**saved, "options": {**saved["options"], "visual_backend": backend}}
            _preserve_visual_result(directory)
            _write_project_settings(directory, saved, updated)
            return updated


def project_edit_status(task_id):
    """One conservative rule for editing new projects, including old pending records."""
    identifier = str(UUID(str(task_id)))
    directory = task_directory(identifier)
    settings = read_json(directory / "settings.json")
    state = read_task(identifier) or {}
    inputs = {"input.txt", "settings.json", "state.json", "source.json", "original.txt", "original.md", "original.markdown", "model-history.json"}
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
                      "项目已有成果：可单独保存模型绑定；修改音色、字幕和输出等制作参数请创建复用副本。"}


def update_project_settings(task_id, selected_libraries, options, voice_profile, model_profile, *, for_start=False):
    """Atomically replace a new project's config; never edit a generated project."""
    identifier = str(UUID(str(task_id)))
    directory = task_directory(identifier)
    with project_lock(identifier), library_operation(identifier, prefix="narration:"):
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
            options = _task_options(text, options, model_profile["_bindings"], draft=not for_start)
            if for_start:
                manual_motion_entries(text, options, complete=True)
            validate_narration_voice(voice_profile, require_credentials=for_start)
            updated = {**saved, "library_id": selected_libraries[0], "library_ids": selected_libraries,
                       "options": options,
                       "project_version": (TASK_ROOT.parent.parent / "project_version").read_text(encoding="utf-8").strip()}
            updated = project_references(updated, model_profile["_bindings"], voice_profile)
            state = read_task(identifier)
            summary = {**state, "library_id": selected_libraries[0], "library_ids": selected_libraries, "audio_mode": audio_mode(options)}
            # Config is the commit point. If its atomic write fails, restore the old progress summary.
            write_json(directory / "state.json", summary)
            try:
                _write_project_settings(directory, saved, updated)
            except Exception:
                write_json(directory / "state.json", state)
                raise
            return updated


def create_task(library_id, text, options, voice_profile, source=None, model_profile=None, *, draft=False):
    with library_operation(library_id), library_operation([] if draft else bgm_libraries(options), prefix="audio-library:"):
        return _create_task(library_id, text, options, voice_profile, source, model_profile, draft=draft)


def _create_task(library_id, text, options, voice_profile, source, model_profile, *, reuse_audio=False, draft=False):
    selected_libraries = library_ids(library_id)
    if not selected_libraries:
        raise ValueError("请至少选择一个素材库。")
    bindings = model_profile["_bindings"] if model_profile is not None else default_bindings()
    options = _task_options(text, options, bindings, draft=draft, reuse_audio=reuse_audio)
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
                "text_hash": stable_hash(text), "schema": 2, "generation_started": False,
                "project_version": (TASK_ROOT.parent.parent / "project_version").read_text(encoding="utf-8").strip()}
    settings = project_references(settings, bindings, voice_profile)
    write_json(directory / "settings.json", settings)
    state = {"id": task_id, "library_id": selected_libraries[0], "library_ids": selected_libraries,
             "state": "pending", "progress": 0, "phase": "待开始", "message": "任务已创建", "video": "", "audio_mode": audio_mode(options)}
    write_json(directory / "state.json", state)
    return task_id


def fork_task(task_id, options, voice_profile, model_profile, *, selected_libraries=None, reuse_audio=False):
    """Create a new task with the same manuscript and reusable, validated caches."""
    source_directory = task_directory(task_id)
    source_settings = read_json(source_directory / "settings.json")
    source_libraries = library_ids(source_settings)
    target_libraries = library_ids(selected_libraries) if selected_libraries is not None else source_libraries
    with project_lock(task_id), library_operation(task_id, prefix="narration:"), library_operation(source_libraries + target_libraries), \
            library_operation(bgm_libraries(options), prefix="audio-library:"):
        if runner.running("narration:" + str(UUID(str(task_id)))):
            raise ValueError("来源任务正在运行，请先停止并等待结束，再创建复用副本。")
        with (source_directory / "input.txt").open("r", encoding="utf-8", newline="") as source:
            text = source.read()
        if stable_hash(text) != source_settings["text_hash"]:
            raise ValueError("来源任务原文发生变化，无法创建复用缓存的副本。")
        if reuse_audio:
            original_options = {key: value for key, value in source_settings["options"].items() if not key.startswith("motion_")}
            requested_options = {key: value for key, value in options.items() if not key.startswith("motion_")}
            if original_options != requested_options or production_voice(voice_profile) != production_voice(source_settings["voice"]):
                raise ValueError("完整配音复用仅适用于保持原配音、字幕及输出设置的动画副本。")
            _resume_checkpoint(source_directory, source_settings, text, 3)
        bindings = dict(model_profile["_bindings"])
        if not voice_profile.get("engine"):
            bindings["tts"] = project_bindings(source_settings)["tts"]
        elif voice_profile.get("profile_id"):
            bindings["tts"] = voice_profile["profile_id"]
        models = {"_bindings": bindings}
        child_id = _create_task(target_libraries, text, options, voice_profile, None, models, reuse_audio=reuse_audio)
        child_directory = task_directory(child_id)
        settings = read_json(child_directory / "settings.json")
        settings["parent_task_id"] = str(UUID(str(task_id)))
        # Keep the source identity only as provenance for deterministic old audio keys.
        history = read_json(source_directory / "model-history.json", default={})
        if source_settings.get("model_reference_schema") != 1:
            history = {"schema": 1, "settings": public_values(source_settings)}
        if history:
            write_json(child_directory / "model-history.json", public_values(history))
        write_json(child_directory / "settings.json", settings)
        copied = 0
        try:
            for folder in ("planning", "matches", "alignment/cache", "bgm/planning", "motion/planning"):
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
                for name in ("audio_manifest.json", "long_audio.json", "audio_plan.json"):
                    if not (source_directory / name).is_file():
                        continue
                    record = read_json(source_directory / name)
                    for row in record.get("segments", []) + record.get("blocks", []):
                        if "path" not in row:
                            continue
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


def run_task(task_id, stop=None, *, start_stage="auto"):
    with project_lock(task_id), library_operation(task_id, prefix="narration:"):
        directory = task_directory(task_id)
        settings = read_json(directory / "settings.json")
        with library_operation(library_ids(settings)), library_operation(bgm_libraries(settings["options"]), prefix="audio-library:"):
            if not settings.get("generation_started"):
                settings["generation_started"] = True
                write_json(directory / "settings.json", settings)
            return _run_task(task_id, stop, start_stage=start_stage)


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


class _ReviewCoordinator(AnalysisCoordinator):
    """Workers own HTTP contexts; the task thread owns evidence, usage and candidates."""

    def __init__(self, retriever, gateway, directory, stop, concurrency, maximum_jobs, metrics):
        super().__init__(min(concurrency, max(1, maximum_jobs)), stop, gateway.before_request)
        self.configured = concurrency
        self.retriever, self.gateway, self.directory = retriever, gateway, directory
        self.settings = deepcopy(gateway.settings)
        self.events = SimpleQueue()
        self.metrics, self.inflight = metrics, []
        self.followups, self.progress = [], None
        self.last_reply = None
        self.sub_batch, self.wave_completed = 0, 0

    def client(self, job):
        client = copy(self.gateway)
        client.settings = self.settings
        client.context = {**self.gateway.context, "review_wave": self.metrics["waves"], "review_job": job["key"]}
        def before():
            self.before_request()  # Admission and the original call counter are protected by the existing lock.
            client.context["request_started_at"] = time.time()
        def diagnostic(record):
            now = time.time()
            record = {**record, "time": now, "elapsed_seconds": max(0, now-record.get("request_started_at", now))}
            self.events.put(("diagnostic", record))
        client.before_request, client.diagnostic, client.after_request = before, diagnostic, self.after_request
        return client

    def after_request(self, details):
        super().after_request(details)
        if self.gateway.after_request:
            self.events.put(("after", dict(details)))

    def execute(self, job):
        try:
            value = self.retriever.request_review(job, self.client(job), self.stop, media_slots=self.media_slots)
        except Exception as exc:
            self.events.put(("failed", job, self.gateway._redact_request(str(exc))[:500]))
            raise
        self.events.put(("result", job, value))

    def report(self, force=False):
        while True:
            try:
                event = self.events.get_nowait()
            except Empty:
                break
            kind = event[0]
            if kind == "diagnostic":
                record = event[1]
                self.metrics["api_requests"] += 1
                self.metrics["api_seconds"] += record["elapsed_seconds"]
                self.last_reply = record.get("time", time.time())
                if self.gateway.diagnostic:
                    self.gateway.diagnostic(record)
            elif kind == "after":
                self.gateway.after_request(event[1])
            else:
                job = event[1]
                descriptor = next(row for row in self.inflight if row["key"] == job["key"])
                if kind == "failed":
                    descriptor.update(state="interrupted", error=event[2])
                else:
                    approved, followups = self.retriever.commit_review(job, event[2], self.gateway, self.directory)
                    self.followups.extend(followups)
                    descriptor.update(state="returned", approved=len(approved), error=event[2]["error"])
                    self.metrics["completed_jobs"] += 1
                    self.metrics["approved_pairs"] += len(approved)
                    self.wave_completed += 1
        now = time.monotonic()
        if self.progress and (force or now-self.reported >= 1):
            self.reported = now
            with self.lock:
                active, limit = self.active, self.limit
            recent = time.strftime("%H:%M:%S", time.localtime(self.last_reply)) if self.last_reply else "尚无"
            done = sum(row["state"] != "queued" for row in self.inflight)
            phases = "、".join(dict.fromkeys(row["phase"] for row in self.inflight))
            status = "停止派发，保存已返回结果" if self.abort.is_set() or (self.stop is not None and self.stop.is_set()) else "审核"
            self.progress(f"{status}并发{active}/{limit}（设置{self.configured}）；第{self.metrics['waves']}波·第{self.sub_batch}批 "
                          f"{phases} {done}/{len(self.inflight)}（本波已完成{self.wave_completed}个请求）；"
                          f"单元{'、'.join(dict.fromkeys(r['unit_id'] for r in self.inflight))}；最近回包{recent}")

    def run(self, jobs, checkpoint, progress):
        self.progress = progress
        self.metrics["waves"] += 1
        self.sub_batch, self.wave_completed = 0, 0
        started = time.monotonic()
        pending = list(jobs)
        submitted = set()
        try:
            while pending:
                self.check()
                fresh = [job for job in pending if job["key"] not in submitted]
                self.metrics["duplicate_jobs_skipped"] += len(pending)-len(fresh)
                pending = fresh
                if not pending:
                    break
                selected, remaining, units = [], [], set()
                for job in pending:
                    if job["unit_id"] in units or len(selected) >= self.maximum:
                        remaining.append(job)
                    else:
                        selected.append(job)
                        units.add(job["unit_id"])
                submitted.update(job["key"] for job in selected)
                self.sub_batch += 1
                self.inflight = [{"key": j["key"], "unit_id": j["unit_id"], "file": j["file"], "state": "queued",
                    "review_keys": [c["_review_key"] for _, c in j["aliases"].values()],
                    "safety_only": j["safety_only"], "repair": j["repair"],
                    "phase": ("来源纠错" if j["repair"] else "来源检查") if j["safety_only"] else
                             ("局部纠错" if j["repair"] else "初审")} for j in selected]
                self.followups = []
                self.metrics["submitted_jobs"] += len(selected)
                self.metrics["review_candidates"] += sum(len(j["aliases"]) for j in selected)
                checkpoint()  # Persist frozen identities before starting any worker.
                try:
                    run_analysis_jobs(self, selected, self.execute)
                finally:
                    self.report(force=True)  # Drain paid replies even while cancellation is being reported.
                pending = self.followups+remaining
        finally:
            self.metrics["wave_seconds"] += time.monotonic()-started
            self.progress = None


def _find_visual_shots(segments, text, visual_plan, assets, reservations, retriever, gateway, directory, options, stop, update, *, previous_plan=None):
    """Search event pages; match once; let the allocator report precise shortages."""
    from app.services.timeline_planner import (
        _allocate_visual_timeline, visual_unit_intervals, unused_source, source_shot_key, candidate_facts,
    )
    units = [u for u in visual_plan["units"] if visual_unit_intervals(u,reservations)]
    target = float(options.get("shot_seconds",7))
    no_gain_limit = int(options.get("visual_no_gain_rounds",2))
    path = directory/"visual_search_state.json"
    saved = read_json(path,default={})
    signature = stable_hash((RETRIEVAL_PROTOCOL,retriever.index_key,visual_plan["key"],gateway.signature("text_fast")))
    same = saved.get("key") == signature
    cursors = {u["id"]:int(saved.get("cursors",{}).get(u["id"],0)) if same else 0 for u in units}
    exhausted, no_gain, pools = set(), {u["id"]:0 for u in units}, {u["id"]:[] for u in units}
    approved = {u["id"]:[] for u in units}
    metrics = {name:0 for name in ("waves","submitted_jobs","completed_jobs","review_candidates","approved_pairs",
        "api_requests","api_seconds","wave_seconds","duplicate_jobs_skipped")}
    reviews = _ReviewCoordinator(retriever,gateway,directory,stop,api_concurrency(options.get("visual_review_concurrency",4)),len(units),metrics)
    gaps, shots, round_number, completed = [], list(reservations), 0, False

    def save():
        write_json(path,{"version":RETRIEVAL_PROTOCOL,"key":signature,"index_key":retriever.index_key,
            "cursors":cursors,"exhausted":sorted(exhausted),"no_gain":no_gain,"review_metrics":metrics,
            "review_inflight":reviews.inflight,"gaps":gaps})
        write_json(directory/"visual_matches.json",{"version":RETRIEVAL_PROTOCOL,"plan_key":visual_plan["key"],
            "gaps":gaps,"issues":list(retriever.issues.values()),"state":"complete" if completed else "partial",
            "timing_method":visual_plan["timing_method"],"units":{
                u["id"]:{"requirement":u["requirement"],"approved":approved[u["id"]],
                          "examined":cursors[u["id"]],"exhausted":u["id"] in exhausted} for u in units}})

    def allocate():
        return _allocate_visual_timeline(segments,approved,target,visual_units=visual_plan["units"],
                                         reservations=reservations,used=())

    def capacity(unit):
        by_key = {c["_review_key"]:c for pool in approved.values() for c in pool}
        occupied = [by_key[s.review_key] for s in shots if s.visual_unit_id != unit["id"] and s.review_key in by_key]
        available = [c for c in approved[unit["id"]] if unused_source(c,occupied)]
        unique = {}
        for candidate in available:
            key = source_shot_key(candidate)
            unique[key] = max(unique.get(key,0),candidate["source_end"]-candidate["source_start"])
        return sum(unique.values()), {f for c in available for f in candidate_facts(c,unit["requirement"])}

    try:
        # Replay only the same fixed page IDs; pair receipts survive index changes independently.
        initial = []
        for unit in units:
            if cursors[unit["id"]]:
                page = retriever.search(unit,0,cursors[unit["id"]])
                pools[unit["id"]] = page.candidates
                initial.extend((unit,c) for c in page.candidates)
        jobs = retriever.prepare_reviews(initial,gateway,directory)
        if jobs:
            reviews.run(jobs,save,lambda msg:update("寻找画面",48,msg))
        for unit in units:
            approved[unit["id"]] = retriever.accepted(unit,pools[unit["id"]])
        shots,gaps = allocate()
        while gaps:
            check_cancel(stop)
            needed = {g["visual_unit_id"]:g for g in gaps}
            selected = [u for u in units if u["id"] in needed and u["id"] not in exhausted and no_gain[u["id"]] < no_gain_limit]
            if not selected:
                break
            round_number += 1
            pairs = []
            before = {u["id"]:capacity(u) for u in selected}
            for unit in selected:
                uid = unit["id"]
                page = retriever.search(unit,cursors[uid],12)
                cursors[uid] = page.cursor
                if page.exhausted:
                    exhausted.add(uid)
                pools[uid].extend(page.candidates)
                pairs.extend((unit,c) for c in page.candidates)
            jobs = retriever.prepare_reviews(pairs,gateway,directory,search_round=round_number)
            save()
            if jobs:
                reviews.run(jobs,save,lambda msg:update("寻找画面",50,msg))
            for unit in selected:
                uid = unit["id"]
                approved[uid] = retriever.accepted(unit,pools[uid])
            shots,gaps = allocate()
            for unit in selected:
                seconds,facts = capacity(unit)
                old_seconds,old_facts = before[unit["id"]]
                uid = unit["id"]
                no_gain[uid] = 0 if seconds > old_seconds+.01 or facts-old_facts else no_gain[uid]+1
            for gap in gaps:
                uid = gap["visual_unit_id"]
                if any(i["unit_id"] == uid for i in retriever.issues.values()):
                    gap["detail"] = "存在模型未确认项，查看匹配记录；不能当作内容不合适"
                elif approved.get(uid):
                    gap["detail"] = "有关联素材，但可用时长、具体事实或全片来源占用不足"
                elif retriever.missing_analysis:
                    gap["detail"] = "部分素材尚未完成拉片，请先补分析"
                else:
                    gap["detail"] = "已检索页未找到对应事件，可补充人物别名、时期资料或素材"
            save()
            update("寻找画面",55,f"第{round_number}轮事件检索；仍有{len(gaps)}处编排缺口；本轮只匹配保存的观察文字")
        if gaps:
            detail = "；".join(g["visual_unit_id"]+"："+g.get("detail",g["reason"]) for g in gaps[:5])
            raise JobPaused(f"仍有{len(gaps)}处画面缺口，共{sum(g['missing_seconds'] for g in gaps):.2f}秒。{detail}。"
                            "已保存付费匹配与检索游标，从3. 寻找画面继续；资料缺项请到视频素材库补拉片。")
        completed = True
        return shots,retriever.boundaries,approved
    finally:
        save()


def _run_task(task_id, stop, *, start_stage="auto"):
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
    agent = visual_backend(options) == "agent"
    # A completed Agent import resumes at production; its signed revision is checked here.
    if agent and level in {0, 3}:
        existing = read_json(directory / "timeline.json", default={})
        if existing.get("visual_policy", {}).get("producer") == "agent":
            try:
                _resume_checkpoint(directory, settings, text, 4)
                level, start_stage = 4, "bgm"
            except (OSError, ValueError, KeyError, TypeError, EOFError, wave.Error):
                # An explicit visual restart can replace an outdated import without deleting it.
                if level != 3:
                    raise
        elif level == 0:
            try:
                _resume_checkpoint(directory, settings, text, 3)
                level, start_stage = 3, "visual"
            except (OSError, ValueError, KeyError, TypeError, EOFError, wave.Error):
                pass
    checkpoint = _resume_checkpoint(directory, settings, text, level) if level else {}
    cached_motion = None
    if agent and level == 3:
        from app.services.narration_agent import _motion
        try:
            cached_motion = _motion(directory, settings, text, checkpoint["segments"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    required = ["text"] if (level == 0 or (level <= 3 and not agent) or
        (level <= 3 and options["motion_enabled"] and cached_motion is None) or
        (level == 4 and options["bgm_mode"] == "auto" and options["bgm_volume"] > 0)) else []
    if level == 0 or (level == 2 and mode == LONG_MODE):
        required.append("text_fast")
    if level <= 3 and not agent and "text_fast" not in required:
        required.append("text_fast")
    # The relevance switch does not disable the independent credits/usability check.
    if level <= 3 and not agent:
        required.append("vision")  # Used only when a saved range still needs local enrichment.
    resolved = resolve_project(settings, roles=required, need_voice=level <= 1,
        need_alignment=level <= 2 and mode == LONG_MODE and options["alignment_enabled"])
    model_profile, voice_profile = resolved["models"], resolved["voice"]
    options.update(resolved["alignment"])
    if level <= 2 and mode == LONG_MODE and options["tts_chunking"] == "manual":
        from app.services.narration_audio import manual_audio_ranges
        if options.get("tts_break_text_hash") != stable_hash(text):
            raise ValueError("任务保存的手动配音断点与原文不一致。")
        manual_audio_ranges(text, options.get("tts_break_after"))
    state["audio_mode"] = mode
    state["resume_stage"] = start_stage
    if stable_hash(text) != settings["text_hash"]:
        raise ValueError("任务原文发生变化，请创建新任务。")
    converted = project_references(settings, resolved["bindings"])
    if converted != settings:
        _write_project_settings(directory, settings, converted)
        settings = converted
    history = read_json(directory / "model-history.json", default={})
    legacy_voice = history.get("settings", {}).get("voice", {})
    legacy_profiles = [legacy_voice] if legacy_voice.get("engine") else []
    write_json(directory / "model-runs" / (str(time.time_ns()) + ".json"),
               {"schema": 1, "time": time.time(), "start_stage": start_stage, **public_values(resolved)})
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
        if level <= 4 and not (agent and level <= 3):
            validate_bgm(options)
        if level <= 3:
            if cached_motion is None:
                validate_motion(options)
            manual_motion_entries(text, options, complete=True)
        if level <= 2 and mode == LONG_MODE and options["alignment_enabled"]:
            validate_alignment_options(options)
        for kind in required:
            gateway.require(kind)
        store = LibraryStore()
        selected_libraries = library_ids(settings)
        if level <= 3 and not agent:
            update("校验字幕资料", 0 if level == 0 else 40, "本地检查正文资料；异常资料修复前不发起模型找片")
            subtitle_issues = [row for identifier in selected_libraries for row in audit_subtitles(store, identifier, stop)
                               if row["state"] in {"recovery_needed", "error"}]
            if subtitle_issues:
                write_json(directory/"subtitle_quality.json", {"state": "blocked", "issues": subtitle_issues})
                raise JobPaused("视频素材库字幕正文校验未通过：" + "；".join(
                    f"第{r['episode']}集：" + (r.get("error") or f"缺少{r.get('missing_body_events', 0)}条正文") for r in subtitle_issues) +
                    "。请到视频素材库→已登记素材→字幕资料校验与恢复，点“校验字幕”及“恢复正文与本地索引”，再从3. 寻找画面继续。配音、字幕、动画及旧成片保留。")
        retriever = MaterialRetriever(store, selected_libraries) if level <= 3 and not agent else None
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
                    break_after=options.get("tts_break_after") if options["tts_chunking"] == "manual" else None,
                    legacy_profiles=legacy_profiles)
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
                tts_budget=budget, subtitle_units=subtitle_units, cache_source=cache_source, legacy_profiles=legacy_profiles)
        else:
            audio_path = checkpoint["audio_path"]
            state.update(subtitle_timing=checkpoint["audio"].get("timing_method", "tts_units"),
                         subtitle_cues=len(checkpoint["audio"]["cues"]))
        subtitle_path = str(directory / "subtitles.srt")
        if level <= 2 or (options.get("subtitles", True) and not Path(subtitle_path).is_file()):
            subtitle_path = create_narration_subtitles(segments, subtitle_path)
        if level <= 3:
            motion_plan, motion_assets, reservations = None, [], []
            if cached_motion is not None:
                from app.models.timeline import VisualShot
                motion_plan, motion_assets = cached_motion["plan"], cached_motion["assets"]
                reservations = [VisualShot(**row) for row in cached_motion["reservations"]]
            elif options["motion_enabled"]:
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
            if agent:
                from app.services.narration_agent import create_task as create_agent_task
                update("准备外部Agent画面任务", 45, "固定全文、配音、字幕和动画占位；不调用内置画面匹配API")
                agent_task = create_agent_task(task_id, locked=True)
                state.update(agent_visual_task=agent_task["task_id"], phase="等待外部Agent画面方案")
                raise JobPaused("整篇画面任务已准备，请展开画面匹配设置，把TASK.md交给外部Agent。"
                    "完成后检查并导入画面方案，再从第4/5阶段推进；配音、字幕和旧成片保留。")
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
            if visual_plan.get("requirements_pending"):
                review = read_json(directory/"planning"/"requirement-review.json")
                reasons = "；".join(r["unit_id"]+"："+r["reason"]+("（已尝试局部纠正一次）" if r.get("corrected") else "")
                                   for r in review["deferred"][:8])
                raise JobPaused(f"画面需求已整理{review['reviewed']}项（复用{review['reused']}项），仍待处理{len(visual_plan['requirements_pending'])}项："
                                + reasons + "。已有配音、字幕、成功需求和旧成片保留；从3. 寻找画面继续仅补未完成项。")
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
                    "index_protocol", "index_key", "stream_index", "time_base", "origin_pts", "source_frame_count", "source_duration",
                    "shot_start_frame", "shot_end_frame", "start_frame", "end_frame",
                    "source_start", "source_end", "usable", "content_kind", "visual_policy", "match_policy", "visual_unit_id",
                    "relation", "facts_match", "evidence", "conflicts", "subtitle_quotes", "visual_checked", "confidence",
                    "subtitle_ids", "visual_evidence", "evidence_protocol")})
                used[-1]["review_key"] = selected["_review_key"]
                for key in ("coverage", "duplicate_group", "revision", "event_id"):
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
            if timeline["visual_policy"].get("producer") == "agent":
                for asset in assets:
                    source = timeline["visual_policy"]["sources"].get(asset["id"])
                    if source:
                        asset["duration"] = source["duration"]  # Actual indexed video duration, not container padding.
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
                                progress=lambda p, message: update("合成视频", render_start + p * (99 - render_start), message),
                                visual_policy=timeline["visual_policy"])
        weak_count = sum(shot.weak_match for shot in shots)
        state.update(state="complete", progress=100, phase="完成", message=f"生成完成，共 {len(shots)} 个镜头，{weak_count} 个镜头需关注匹配详情。",
                     video=video, weak_matches=weak_count, model_calls=model_calls)
    except (JobCancelled, JobPaused) as exc:
        state.update(state="paused", message=str(exc), model_calls=model_calls)
    except Exception as exc:
        # Gateway errors omit credentials and reference payloads.
        state.update(state="failed", message=str(exc), model_calls=model_calls)
    write_json(directory / "state.json", state)
