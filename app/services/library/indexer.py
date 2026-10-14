"""Path import, subtitle matching and resumable scene analysis."""

import hashlib
import json
import math
import re
import copy
import sqlite3
import time
from pathlib import Path
from subprocess import TimeoutExpired
from uuid import uuid4

import pysrt
from loguru import logger

from app.services.ass_subtitle import select_chinese_ass
from app.services.library.frame_extractor import extract_frames, inspect_video
from app.services.library.runner import (
    AnalysisCoordinator, DEFAULT_API_CONCURRENCY, JobCancelled, api_concurrency, check_cancel,
    library_operation, run_analysis_jobs as _run_build_jobs,
)
from app.services.library.gateway import ModelAPIError, ModelContentError, ModelFormatError, ModelInputBudgetError, ModelOutputLimitError
from app.services.library.store import stable_hash, write_json
from app.services.library.validation import cached_result
from app.services.subtitle_text import convert_ass_to_srt, decode_subtitle_bytes, has_timecodes


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".mpeg4"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa"}


def scan_directory(directory, extensions, recursive=True):
    root = Path(str(directory).strip().strip('"')).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("请输入运行本程序的电脑上的目录。")
    return sorted(str(item.resolve()) for item in (root.rglob("*") if recursive else root.glob("*")) if item.is_file() and item.suffix.lower() in extensions)


def episode_number(path):
    name = Path(path).stem
    for pattern in (r"[Ss]\d{1,2}[Ee](\d{1,3})", r"第\s*(\d{1,3})\s*[集话話]", r"[\[【（(](\d{1,3})[\]】）)]", r"(?:^|[ ._-])(?:[Ee][Pp]?|[Ee]pisode)?\s*(\d{1,3})(?:[ ._-]|$)"):
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def suggest_mappings(videos, subtitles):
    suggestions = []
    for index, video in enumerate(sorted(videos, key=lambda path: (episode_number(path) or 100000, path.casefold())), 1):
        episode = episode_number(video)
        stem = Path(video).stem.casefold()
        scores = []
        for subtitle in subtitles:
            subtitle_stem = Path(subtitle).stem.casefold()
            score = 100 if subtitle_stem == stem else 80 if subtitle_stem.startswith(stem + ".") or subtitle_stem.startswith(stem + "_") else 0
            if episode is not None and episode_number(subtitle) == episode:
                score = max(score, 50)
            if score:
                if (re.search(r"(?:^|[._ \-\[\]()])(?:chs|sc|zh-cn|zh-hans)(?:$|[._ \-\[\]()])", subtitle_stem)
                        or any(label in subtitle_stem for label in ("简体", "简中", "簡體", "簡中"))):
                    score += 5
                scores.append((score, subtitle))
        scores.sort(key=lambda item: (-item[0], item[1]))
        ambiguous = bool(len(scores) > 1 and scores[0][0] == scores[1][0])
        suggestions.append({"video": video, "episode": episode or index,
                            "subtitle": scores[0][1] if scores and not ambiguous else "",
                            "ambiguous": ambiguous, "candidates": [item[1] for item in scores[:8]]})
    return suggestions


def prepare_subtitle(path, settings):
    source = Path(path).resolve(strict=True)
    data = source.read_bytes()
    decoded = decode_subtitle_bytes(data)
    text = decoded.text
    uncertainty = 0
    if source.suffix.lower() in {".ass", ".ssa"}:
        if settings.get("chinese_only", True):
            filtered = select_chinese_ass(text, settings.get("styles", []), settings.get("line_mode", "auto"))
            text = filtered.text
            uncertainty = filtered.uncertain_events
        text = convert_ass_to_srt(text)
    if not has_timecodes(text):
        raise ValueError(f"字幕没有有效时间戳：{source.name}")
    cues = pysrt.from_string(text)
    if not cues or not any(item.text.strip() and item.end.ordinal > item.start.ordinal for item in cues):
        raise ValueError(f"字幕没有有效台词：{source.name}")
    return data, text, hashlib.sha256(data).hexdigest(), uncertainty


def import_materials(store, library_id, mappings, subtitle_settings, stop=None):
    with library_operation(library_id):
        return _import_materials(store, library_id, mappings, subtitle_settings, stop)


def _import_materials(store, library_id, mappings, subtitle_settings, stop=None):
    failures = []
    duplicates = []
    for index, mapping in enumerate(mappings):
        check_cancel(stop)
        store.job(library_id, "import", progress=index * 100 / len(mappings), message=f"登记素材 {index + 1}/{len(mappings)}")
        try:
            media = inspect_video(mapping.video_path, stop)
            subtitle = prepare_subtitle(mapping.subtitle_path, subtitle_settings) if mapping.subtitle_path else None
            duplicate = next((item for item in store.assets(library_id) if item["fingerprint"] == media.fingerprint and item["path"] != media.path), None)
            if duplicate:
                if subtitle and duplicate.get("content_hash") not in {None, subtitle[2]}:
                    raise ValueError("相同视频内容已登记，但字幕不同，请确认后只保留一组对应关系。")
                duplicates.append(Path(mapping.video_path).name)
                continue
            asset_id = store.upsert_asset(library_id, media, mapping.episode)
            if subtitle:
                data, text, content_hash, _ = subtitle
                directory = store.directory(library_id) / "subtitles" / asset_id / stable_hash((content_hash, subtitle_settings))[:20]
                directory.mkdir(parents=True, exist_ok=True)
                original = directory / ("original" + Path(mapping.subtitle_path).suffix.lower())
                processed = directory / "processed.srt"
                original.write_bytes(data)
                processed.write_text(text, encoding="utf-8")
                store.set_subtitle(library_id, asset_id, str(Path(mapping.subtitle_path).resolve()), original, processed, content_hash, subtitle_settings)
            elif not mapping.subtitle_path:
                with store.database(library_id) as db:
                    db.execute("UPDATE assets SET status=CASE WHEN active_subtitle IS NULL THEN status ELSE 'pending' END,active_subtitle=NULL WHERE id=?", (asset_id,))
        except JobCancelled:
            raise
        except Exception as exc:
            failures.append(f"{Path(mapping.video_path).name}: {exc}")
    store.job(library_id, "import", state="failed" if failures else "complete", progress=100,
              message=("\n".join(failures) if failures else "素材登记完成，视频仍位于原路径。") + (f"\n跳过 {len(duplicates)} 个相同内容的视频副本。" if duplicates else ""))


def _windows(asset, window_seconds):
    cues = pysrt.open(asset["processed_path"], encoding="utf-8") if asset.get("processed_path") else []
    windows = []

    def append(start, end):
        text = "\n".join(f"{item.start} --> {item.end} {item.text.replace(chr(10), ' ')}" for item in cues
                         if item.start.ordinal / 1000 < end and item.end.ordinal / 1000 > start)
        if len(text) > 7000:
            if end - start < .5:
                raise ValueError("字幕事件过于密集，请先整理重复字幕。")
            middle = (start + end) / 2
            append(start, middle)
            append(middle, end)
        else:
            windows.append((start, end, text))
    for index in range(math.ceil(asset["duration"] / window_seconds)):
        append(index * window_seconds, min(asset["duration"], (index + 1) * window_seconds))
    return windows


def build_library(store, library_id, gateway, settings, stop=None):
    previous_callback, previous_diagnostic = gateway.before_request, gateway.diagnostic
    try:
        with library_operation(library_id):
            return _build_library(store, library_id, gateway, settings, stop)
    finally:
        gateway.before_request, gateway.diagnostic = previous_callback, previous_diagnostic


def _episode_directory(store, library_id, asset, scenes, gateway, library_name, stop, retry_skipped):
    key = stable_hash(("episode-summary-v1", gateway.signature("text"), [scene["id"] for scene in scenes]))
    cached = cached_result(store.cache_get(library_id, key), {"sections": list, "characters": list})
    if cached and not cached.get("partial"):
        store.resolve_analysis_errors(library_id, asset["id"], stage="本集剧情目录")
        return cached
    outline = [{"start": scene["start"], "end": scene["end"],
                "summary": scene["analysis"]["summary"][:220],
                "characters": scene["analysis"].get("characters", [])[:6]} for scene in scenes]
    gaps = []

    def summarize(group):
        check_cancel(stop)
        group_key = stable_hash(("episode-group-v1", gateway.signature("text"), group, library_name))
        value = cached_result(store.cache_get(library_id, group_key), {"summary": str, "characters": list})
        if value:
            return [value]
        failure_key = "summary-failure:" + group_key
        failure = store.cache_get(library_id, failure_key)
        if failure and not retry_skipped:
            gaps.append(failure)
            return []
        gateway.context.update(stage="本集剧情目录", window=0, group=group_key[:16],
                               start=group[0]["start"], end=group[-1]["end"])
        try:
            value = gateway.json(
                "根据场景记录生成简短剧情目录，不添加记录中没有的剧情。"
                "返回 {summary:不超过400字,characters:[人物名称字符串],relationships:[关系变化字符串],"
                "themes:[主题字符串],events:[最多8条事件字符串]}。\n"
                f"作品：{library_name}，第{asset['episode']}集。场景：{json.dumps(group, ensure_ascii=False)}",
                max_output=4096, required={"summary": str, "characters": list})
            store.cache_put(library_id, group_key, value)
            store.cache_delete(library_id, failure_key)
            return [value]
        except ModelOutputLimitError as exc:
            if len(group) > 1:
                middle = len(group) // 2
                return summarize(group[:middle]) + summarize(group[middle:])
            error = exc
        except (ModelContentError, ModelFormatError) as exc:
            error = exc
        failure = {"start": group[0]["start"], "end": group[-1]["end"], "error": str(error)}
        store.cache_put(library_id, failure_key, failure)
        gaps.append(failure)
        return []

    sections = []
    for offset in range(0, len(outline), 10):
        sections.extend(summarize(outline[offset:offset + 10]))
    result = {"sections": sections, "characters": list(dict.fromkeys(name for group in sections for name in group["characters"])),
              "partial": bool(gaps), "gaps": gaps}
    if not gaps:
        store.cache_put(library_id, key, result)
        store.resolve_analysis_errors(library_id, asset["id"], stage="本集剧情目录")
    return result


class _BuildCoordinator(AnalysisCoordinator):
    """One build owns request admission, progress and diagnostics; contexts remain per worker."""

    def __init__(self, store, library_id, gateway, stop, concurrency, batch_size):
        super().__init__(concurrency, stop, gateway.before_request)
        self.store, self.library_id, self.gateway, self.stop = store, library_id, gateway, stop
        self.batch_limits = {"text": batch_size, "vision": batch_size}
        self.frames = self.media_slots
        self.total = 0
        self.phase = "scenes"
        self.directory_done = self.directory_total = 0
        self.settings = copy.deepcopy(gateway.settings)
        large_models = {"mimo-v2.6-flash", "mimo-v2.6-pro", "mimo-v2.6-pro-ultraspeed", "mimo-v2.5", "mimo-v2.5-pro"}
        for kind in ("text", "vision", "text_backup", "vision_backup"):
            profile = self.settings.get(kind)
            if not isinstance(profile, dict):
                continue
            model = str(profile.get("model", "")).lower().removeprefix("openai/")
            context = 1000000 if model in large_models else int(self.settings.get("context_tokens", 64000))
            profile.update(context_tokens=context, max_prompt_chars=max(16000, min(300000, context // 2)),
                           request_timeout=600, library_batch=True)

    def diagnostic(self, record):
        try:
            with self.lock:
                self.store.record_analysis_diagnostic(self.library_id, record, record)
        except (OSError, sqlite3.Error) as exc:
            logger.warning("建库诊断保存失败，已返回结果仍继续缓存：{}", exc)

    def record_request(self, record):
        path = self.store.directory(self.library_id) / "requests" / (uuid4().hex + ".json")
        write_json(path, record)
        return str(path)

    def client(self, context, *, batch=True):
        client = copy.copy(self.gateway)
        client.settings = self.settings if batch else copy.deepcopy(self.settings)
        if not batch:
            for kind in ("text", "vision", "text_backup", "vision_backup"):
                if isinstance(client.settings.get(kind), dict):
                    client.settings[kind]["library_batch"] = False
        client.context = dict(context)
        client.before_request = self.before_request
        client.after_request = self.after_request
        client.diagnostic = self.diagnostic
        client.request_recorder = self.gateway.request_recorder or self.record_request
        return client

    def report(self, force=False):
        with self.lock:
            now = time.monotonic()
            if not force and now - self.reported < 1:
                return
            self.reported = now
            if self.phase == "directories":
                progress = 95 + 5 * self.directory_done / max(1, self.directory_total)
                message = f"剧情目录 {self.directory_done}/{self.directory_total}"
            else:
                progress = 5 + 90 * self.done / max(1, self.total)
                message = f"场景 {self.done}/{self.total}；有效批量：字幕{self.batch_limits['text']}／视觉{self.batch_limits['vision']}"
            self.store.job(self.library_id, progress=progress,
                message=f"{message}；API并发 {self.active}/{self.limit}（设置上限{self.maximum}）；本次调用 {self.calls} 次")

    def smaller(self, kind, count):
        with self.lock:
            self.batch_limits[kind] = min(self.batch_limits[kind], max(1, count // 2))

    def done_entry(self, entry):
        with self.lock:
            if not entry.get("done"):
                entry["done"] = True
                self.done += 1


def _batch_context(kind, entries):
    asset = entries[0]["plan"]["asset"]
    indexes = [entry["window"] for entry in entries]
    return {"asset_id": asset["id"], "episode": asset["episode"],
            "stage": "视觉分析" if kind == "vision" else "字幕分析",
            "window": indexes[0] if len(indexes) == 1 else f"{min(indexes)}–{max(indexes)}",
            "start": entries[0]["start"], "end": entries[-1]["end"],
            "group": stable_hash((kind, [entry["id"] for entry in entries]))[:16],
            "scene_ids": [entry["id"] for entry in entries], "windows": indexes}


def _failed_window(coordinator, entry, kind, error):
    if entry.get("failed"):
        return
    entry["failed"] = True
    context = _batch_context(kind, [entry])
    store, library_id = coordinator.store, coordinator.library_id
    store.discard_scene(library_id, entry["id"])
    store.cache_put(library_id, entry["failure_key"], {**context, "error": str(error)})
    coordinator.diagnostic({**context, "state": "failed", "attempt": 1, "error": str(error),
                            "failure_category": getattr(error, "category", "format" if isinstance(error, ModelFormatError) else "local")})
    coordinator.done_entry(entry)


def _partial_batch_rows(message):
    """Only decode completed elements of the root scenes array, never nested frame fragments."""
    source = str(message).strip()
    opening = source.find("{")
    if opening < 0:
        return []
    source = source[opening:]
    match = re.match(r'\{\s*"scenes"\s*:\s*\[', source)
    if not match:
        return []
    position, rows = match.end(), []
    decoder = json.JSONDecoder(strict=False)
    while position < len(source):
        while position < len(source) and source[position] in " \r\n\t,":
            position += 1
        try:
            row, position = decoder.raw_decode(source, position)
        except ValueError:
            break
        if not isinstance(row, dict):
            break
        rows.append(row)
    return rows


def _fetch_batch(coordinator, entries, kind, library_name):
    field = "visual" if kind == "vision" else "analysis"
    key_field = "visual_key" if kind == "vision" else "text_key"
    required = {"summary": str, "frames": list} if kind == "vision" else {"summary": str, "characters": list}
    entries = [entry for entry in entries if not entry.get("failed") and entry.get(field) is None]
    if not entries:
        return
    coordinator.check()
    limit = coordinator.batch_limits[kind]
    if len(entries) > limit:
        for offset in range(0, len(entries), limit):
            _fetch_batch(coordinator, entries[offset:offset + limit], kind, library_name)
        return
    context = _batch_context(kind, entries)
    client = coordinator.client(context, batch=len(entries) > 1)
    if kind == "vision":
        instruction = ("按顺序描述视频帧，只记录可见动作、表情、环境和景别，人物身份不确定时不要编造。"
            "返回 {summary,actions:[字符串],emotions:[字符串],environment,shot_sizes:[字符串],"
            "frames:[{index:0至2,description:字符串}]}。")
    else:
        instruction = ("为素材检索记录字幕片段。只根据本场景台词描述事件，不推测表情或构图。人物未明确时写未知。"
            "返回 {title,summary,characters:[字符串],events:[字符串],themes:[字符串],emotions:[字符串],evidence:[关键台词字符串]}。")
    aliases = {f"W{entry['window']:04d}": entry for entry in entries}
    data = [{"scene_id": alias, "start": entry["start"], "end": entry["end"],
             **({"frame_ids": [f"{alias}:F{i}" for i in range(3)]} if kind == "vision" else {"subtitle": entry["transcript"]})}
            for alias, entry in aliases.items()]
    images = [frame for entry in entries for frame in entry.get("frames", [])] if kind == "vision" else None
    if len(entries) == 1:
        prompt = instruction if kind == "vision" else instruction + f"\n作品：{library_name}。第{context['episode']}集，范围{context['start']:.2f}至{context['end']:.2f}秒。字幕：\n{entries[0]['transcript']}"
    else:
        prompt = (instruction + "\n这是同一视频的独立场景批次。严格分别分析，不把其他场景的台词或画面混入当前场景。"
            "只返回 {\"scenes\":[{\"scene_id\":\"W0001\",\"result\":{上述字段}}]}，每个请求scene_id恰好一次，保持ID原样。"
            "图片顺序对应每项frame_ids，每场景恰好3张，其result.frames.index仍使用本场景0/1/2。"
            "每条summary不超过160字，每张frame.description不超过80字，列表各最多6项。\n"
            f"作品：{library_name}。第{context['episode']}集。场景：" + json.dumps(data, ensure_ascii=False))
    raw_bytes = len(prompt.encode("utf-8")) + sum((Path(frame).stat().st_size + 2) // 3 * 4 for frame in images or [])
    if raw_bytes > 32 * 1024 * 1024:
        error = ModelInputBudgetError("批量请求体超过32MB准备预算。")
        rows = []
    else:
        error = None
        try:
            result = client.json(prompt, images=images, model_kind=kind,
                max_output=2400 if len(entries) == 1 else min(65536, 1024 + len(entries) * (700 if kind == "vision" else 450)),
                required=required if len(entries) == 1 else {"scenes": list},
                request_metadata={"batch_size": len(entries), "scenes": [{"alias": alias, "scene_id": entry["id"], "window": entry["window"]} for alias, entry in aliases.items()]})
            rows = [{"scene_id": next(iter(aliases)), "result": result}] if len(entries) == 1 else result["scenes"]
        except (ModelContentError, ModelFormatError, ModelInputBudgetError) as exc:
            error = exc
            rows = _partial_batch_rows(getattr(exc, "response_text", "")) if len(entries) > 1 else []
        except ModelAPIError as exc:
            if len(entries) > 1 and "HTTP 400" in str(exc) and any(term in str(exc) for term in ("Request Error", "Invalid request parameters")):
                error, rows = exc, []
            else:
                raise
    grouped = {}
    for row in rows:
        if isinstance(row, dict):
            alias = str(row.get("scene_id", "")).strip().upper()
            if alias in aliases:
                grouped.setdefault(alias, []).append(row.get("result", row))
    for alias, values in grouped.items():
        value = cached_result(values[0], required) if len(values) == 1 else None
        if value:
            entry = aliases[alias]
            coordinator.store.cache_put(coordinator.library_id, entry[key_field], value)
            entry[field] = value
    missing = [entry for entry in entries if entry.get(field) is None]
    if missing:
        if len(entries) == 1:
            _failed_window(coordinator, missing[0], kind, error or ModelFormatError("单场景回复缺少有效字段。"))
        else:
            if isinstance(error, (ModelInputBudgetError, ModelOutputLimitError)) or not any(entry.get(field) is not None for entry in entries):
                if not isinstance(error, ModelContentError):
                    coordinator.smaller(kind, len(entries))
            middle = max(1, len(missing) // 2)
            _fetch_batch(coordinator, missing[:middle], kind, library_name)
            _fetch_batch(coordinator, missing[middle:], kind, library_name)
    if all(entry.get(field) is not None for entry in entries):
        coordinator.store.resolve_analysis_errors(coordinator.library_id, context["asset_id"], context["stage"], context["window"])


def _commit_ready(coordinator, entries):
    for entry in entries:
        if entry.get("done") or entry.get("failed") or entry.get("analysis") is None:
            continue
        if entry["plan"]["visual"] and (entry.get("visual") is None or not entry.get("frames")):
            continue
        analysis = entry["analysis"]
        if not entry["transcript"]:
            analysis = {**analysis, "summary": entry["visual"]["summary"]}
        scene = {"id": entry["id"], "start": entry["start"], "end": entry["end"],
                 "transcript": entry["transcript"], "analysis": analysis, "version": entry["plan"]["version"]}
        with coordinator.lock:
            store = coordinator.store
            store.save_scene(coordinator.library_id, entry["plan"]["asset"]["id"], scene, entry.get("visual") or {}, entry.get("frames") or [])
            store.cache_delete(coordinator.library_id, entry["failure_key"])
            store.resolve_analysis_errors(coordinator.library_id, entry["plan"]["asset"]["id"], window=entry["window"])
            entry["plan"]["scenes"][scene["id"]] = scene
            coordinator.done_entry(entry)


def _analyze_batch(coordinator, entries, library_name, retry_skipped):
    store, library_id = coordinator.store, coordinator.library_id
    try:
        for entry in entries:
            coordinator.check()
            if store.cache_get(library_id, entry["failure_key"]) and not retry_skipped:
                entry["failed"] = True
                store.discard_scene(library_id, entry["id"])
                coordinator.done_entry(entry)
                continue
            entry["analysis"] = cached_result(store.cache_get(library_id, entry["text_key"]), {"summary": str, "characters": list})
            if not entry["transcript"]:
                entry["analysis"] = {"title": "无字幕片段", "summary": "待视觉分析", "themes": [], "characters": []}
            entry["visual"] = cached_result(store.cache_get(library_id, entry["visual_key"]), {"summary": str, "frames": list}) if entry["plan"]["visual"] else {}
        _fetch_batch(coordinator, entries, "text", library_name)
        for entry in entries:
            if entry.get("failed") or not entry["plan"]["visual"]:
                continue
            acquired = False
            try:
                while not acquired:
                    coordinator.check()
                    acquired = coordinator.frames.acquire(timeout=.2)
                coordinator.check()
                frame_dir = store.directory(library_id) / "frames" / entry["plan"]["asset"]["id"] / entry["visual_key"][:20]
                entry["frames"] = extract_frames(entry["plan"]["asset"]["path"], entry["start"], entry["end"], 3, frame_dir)
            except JobCancelled:
                raise
            except (OSError, RuntimeError, TimeoutExpired) as exc:
                _failed_window(coordinator, entry, "vision", exc)
            finally:
                if acquired:
                    coordinator.frames.release()
        _commit_ready(coordinator, entries)
        if entries[0]["plan"]["visual"]:
            _fetch_batch(coordinator, entries, "vision", library_name)
    finally:
        _commit_ready(coordinator, entries)


def _build_library(store, library_id, gateway, settings, stop=None):
    """Plan once, run bounded cross-video batches, and preserve per-window checkpoints."""
    assets = store.assets(library_id)
    if not assets:
        raise ValueError("素材库为空，请先登记视频。")
    def setting(name, default, maximum):
        value = settings.get(name, default)
        try:
            valid = not isinstance(value, bool) and int(value) == float(value) and 1 <= int(value) <= maximum
        except (ValueError, TypeError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f"{name}需为1至{maximum}的整数。")
        return int(value)
    batch_size = setting("batch_size", 50, 128)
    concurrency = api_concurrency(settings.get("concurrency", DEFAULT_API_CONCURRENCY))
    coordinator = _BuildCoordinator(store, library_id, gateway, stop, concurrency, batch_size)
    visual = bool(settings.get("visual", True))
    window_seconds = max(10, min(120, float(settings.get("window_seconds", 15))))
    retry_skipped = bool(settings.get("retry_skipped", False))
    library_name = next(item["name"] for item in store.list_libraries() if item["id"] == library_id)
    version = stable_hash({"version": 2, "text": gateway.signature("text"), "vision": gateway.signature("vision"),
                           "visual": visual, "window": window_seconds})
    plans, warnings = [], []
    failed = 0
    for asset_index, asset in enumerate(assets):
        coordinator.check()
        context = {"asset_id": asset["id"], "episode": asset["episode"], "window": 0,
                   "start": 0, "end": asset["duration"], "stage": "读取素材"}
        store.job(library_id, progress=5 * asset_index / len(assets),
                  message=f"准备第{asset['episode']}集；每批最多{batch_size}场景，API并发上限{concurrency}")
        changed = False
        try:
            media = inspect_video(asset["path"], stop)
            if media.fingerprint != asset["fingerprint"]:
                changed = True
                raise ValueError("原视频已变化，请重新登记此文件。")
            if not visual and not asset.get("processed_path"):
                raise ValueError("此视频没有字幕，必须启用视觉分析。")
            windows = _windows(asset, window_seconds)
            identifiers = [stable_hash((asset["id"], asset["fingerprint"], asset.get("content_hash"), start, end, version))
                           for start, end, transcript in windows]
            retained = store.begin_analysis(library_id, asset["id"], identifiers)
            plan = {"asset": asset, "visual": visual, "version": version, "retained": retained,
                    "scenes": {}, "entries": [], "directory": None}
            for index, ((start, end, transcript), identifier) in enumerate(zip(windows, identifiers), 1):
                text_key = stable_hash(("scene-text-v2", gateway.signature("text"), library_name, transcript, asset["episode"], start, end))
                visual_key = stable_hash(("scene-vision-v1", gateway.signature("vision"), asset["fingerprint"], start, end))
                plan["entries"].append({"plan": plan, "id": identifier, "window": index, "start": start, "end": end,
                    "transcript": transcript, "text_key": text_key, "visual_key": visual_key,
                    "failure_key": "window-failure:" + stable_hash((identifier, text_key, visual_key))})
            coordinator.total += len(windows)
            plans.append(plan)
        except JobCancelled:
            raise
        except Exception as exc:
            failed += 1
            store.set_asset_status(library_id, asset["id"], "changed" if changed else "failed")
            coordinator.diagnostic({**context, "state": "failed", "attempt": 1, "error": str(exc)})
            warnings.append(f"第 {asset['episode']} 集：{exc}")
    jobs = [plan["entries"][offset:offset + batch_size] for plan in plans for offset in range(0, len(plan["entries"]), batch_size)]
    try:
        _run_build_jobs(coordinator, jobs, lambda entries: _analyze_batch(coordinator, entries, library_name, retry_skipped))
    except Exception:
        for plan in plans:
            if plan["scenes"] or plan["retained"]:
                store.set_asset_status(library_id, plan["asset"]["id"], "partial")
        raise
    directory_jobs = []
    for plan in plans:
        if not plan["scenes"]:
            failed += 1
            store.set_asset_status(library_id, plan["asset"]["id"], "failed")
            warnings.append(f"第 {plan['asset']['episode']} 集：没有可用场景，已跳过 {len(plan['entries'])} 个窗口。")
        else:
            store.set_asset_status(library_id, plan["asset"]["id"], "partial")
            directory_jobs.append(plan)
    coordinator.phase = "directories"
    coordinator.directory_total = len(directory_jobs)
    def finish_directory(plan):
        asset = plan["asset"]
        scenes = sorted(plan["scenes"].values(), key=lambda scene: (scene["start"], scene["end"]))
        client = coordinator.client({"asset_id": asset["id"], "episode": asset["episode"], "window": 0,
                                     "stage": "本集剧情目录", "start": 0, "end": asset["duration"]}, batch=False)
        try:
            directory = _episode_directory(store, library_id, asset, scenes, client, library_name, coordinator.abort, retry_skipped)
            store.cache_put(library_id, "episode-summary:" + asset["id"], directory)
            plan["directory"] = directory
            partial = any(entry.get("failed") for entry in plan["entries"]) or directory.get("partial")
            store.set_asset_status(library_id, asset["id"], "partial" if partial else "ready")
        finally:
            with coordinator.lock:
                coordinator.directory_done += 1
    _run_build_jobs(coordinator, directory_jobs, finish_directory)
    completed = partial = 0
    for plan in directory_jobs:
        skipped = sum(bool(entry.get("failed")) for entry in plan["entries"])
        directory = plan["directory"]
        if skipped or directory.get("partial"):
            partial += 1
            warnings.append(f"第 {plan['asset']['episode']} 集：可用 {len(plan['scenes'])}/{len(plan['entries'])} 个场景，跳过 {skipped} 个窗口，目录缺失 {len(directory.get('gaps', []))} 组。")
        else:
            completed += 1
    available = len(store.scenes(library_id))
    state = "complete" if not warnings else "partial" if available else "failed"
    store.job(library_id, state=state, progress=100,
        message=f"完成 {completed}/{len(assets)} 个视频，部分完成 {partial} 个，失败 {failed} 个，可用场景 {available} 个，本次调用 {coordinator.calls} 次；设置每批{batch_size}、并发{concurrency}。"
                + ("\n" + "\n".join(warnings) if warnings else ""))
