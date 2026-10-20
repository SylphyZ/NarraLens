"""Validate model replies and cached results before they reach indexing or editing."""

import math


class ModelFormatError(ValueError):
    """A model reply is incomplete or does not satisfy the expected data shape."""


def confidence(value):
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return 0.0
    return min(1.0, max(0.0, number)) if math.isfinite(number) else 0.0


def frame_index(value, default=1):
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
        return int(number) if math.isfinite(number) and number.is_integer() and 0 <= number <= 2 else default
    except (ValueError, TypeError, OverflowError):
        return default


def validate_result(value, required=None):
    if not isinstance(value, dict):
        raise ModelFormatError("模型需要返回顶层 JSON 对象。")
    result = dict(value)
    for field in ("summary", "claim"):
        if field in result and (not isinstance(result[field], str) or not result[field].strip()):
            raise ModelFormatError(f"模型 JSON 字段 {field} 应为非空字符串。")
    for field, expected in (required or {}).items():
        if field not in result:
            raise ModelFormatError(f"模型 JSON 缺少字段：{field}。")
        if result[field] is None and expected is list:
            result[field] = []
        if not isinstance(result[field], expected):
            raise ModelFormatError(f"模型 JSON 字段 {field} 的类型应为 {expected.__name__}。")
        if expected is str and not result[field].strip():
            raise ModelFormatError(f"模型 JSON 字段 {field} 不能为空。")
    for field in ("characters", "themes", "events", "emotions", "queries", "avoid", "arguments", "actions", "shot_sizes", "evidence"):
        if field not in result:
            continue
        items = result[field] or []
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            raise ModelFormatError(f"模型 JSON 字段 {field} 应为字符串列表。")
        result[field] = list(dict.fromkeys(item.strip() for item in items if item.strip()))[:32]
    if "frames" in result:
        frames = result["frames"] or []
        if not isinstance(frames, list) or any(not isinstance(item, dict) or not isinstance(item.get("description"), str) for item in frames):
            raise ModelFormatError("模型 frames 应为包含 description 的对象列表。")
        result["frames"] = [{**item, "index": frame_index(item.get("index"))} for item in frames[:3]]
    if "clips" in result:
        clips = result["clips"] or []
        if not isinstance(clips, list):
            raise ModelFormatError("模型 clips 应为列表。")
        normalized = []
        for item in clips:
            if not isinstance(item, dict) or not isinstance(item.get("scene_id"), str) or not item["scene_id"]:
                raise ModelFormatError("模型 clips 条目缺少有效的 scene_id。")
            selected_frame = frame_index(item.get("frame_index"), None)
            normalized.append({**item, "confidence": confidence(item.get("confidence")),
                               "frame_index": selected_frame if selected_frame is not None else 1,
                               "frame_index_valid": selected_frame is not None and item.get("frame_index_valid", True) is not False,
                               "reason": item.get("reason") if isinstance(item.get("reason"), str) else ""})
        result["clips"] = normalized
    return result


def cached_result(value, required=None):
    try:
        return validate_result(value, required)
    except ModelFormatError:
        return None


def selected_ids(value, allowed, field, *, required=False):
    if not isinstance(value, list) or any(not isinstance(v, str) or v not in allowed for v in value):
        raise ModelFormatError(f"{field}须使用本次给定的固定ID列表。")
    if len(value) != len(set(value)) or (required and not value):
        raise ModelFormatError(f"{field}不能重复或缺失。")
    return value


def observation_result(row, item):
    """Validate only this range; a malformed peer cannot discard a paid valid result."""
    from app.models.library import ANALYSIS_PROTOCOL, media_id
    from app.services.library.store import stable_hash
    if not isinstance(row, dict) or row.get("range_id") != item["id"]:
        raise ModelFormatError("range_id必须等于当前给定ID。")
    if not isinstance(row.get("content_kind"), str) or row.get("content_kind") not in {"story", "credits", "opening", "black", "unknown"}:
        raise ModelFormatError("content_kind应为story/credits/opening/black/unknown。")
    if not isinstance(row.get("summary"), str) or not row["summary"].strip():
        raise ModelFormatError("summary须说明实际内容。")
    frames = {r["id"]: r for r in item["frames"]}
    cues = {r["id"]: r for r in item["cues"]}
    observations = []
    raw = row.get("observations")
    if not isinstance(raw, list) or not raw:
        raise ModelFormatError("observations须包含实际观察及证据ID。")
    for index, evidence in enumerate(raw):
        if not isinstance(evidence, dict) or not isinstance(evidence.get("kind"), str) or evidence.get("kind") not in {"visual", "subtitle", "context"}:
            raise ModelFormatError("观察种类应为visual/subtitle/context。")
        if not isinstance(evidence.get("text"), str) or not evidence["text"].strip():
            raise ModelFormatError("观察text不能为空。")
        fids = selected_ids(evidence.get("frame_ids", []), frames, "frame_ids")
        cids = selected_ids(evidence.get("cue_ids", []), cues, "cue_ids")
        if evidence["kind"] == "visual" and not fids or evidence["kind"] == "subtitle" and not cids:
            raise ModelFormatError("可见事实必须有当前范围帧ID，字幕说明必须有当前字幕ID。")
        if evidence["kind"] == "subtitle" and any(cues[c].get("partial") for c in cids):
            raise ModelFormatError("跨界字幕只可作为context，不能充当当前完整台词证据。")
        times = [frames[f]["time"] for f in fids]
        spans = [(cues[c]["start"], cues[c]["end"]) for c in cids]
        if evidence["kind"] == "context":
            a, b = item["start"], item["end"]
        else:
            a = min(times+[x[0] for x in spans])
            b = max(times+[x[1] for x in spans])
        observations.append({"id": media_id("O", item["id"], evidence), "kind": evidence["kind"],
                             "text": evidence["text"].strip(), "frame_ids": fids, "cue_ids": cids,
                             "start": max(item["start"], a), "end": min(item["end"], b)})
    people = row.get("visible_people", [])
    if not isinstance(people, list):
        raise ModelFormatError("visible_people应为对象列表。")
    for person in people:
        if not isinstance(person, dict) or not isinstance(person.get("name"), str) or not person["name"].strip():
            raise ModelFormatError("出镜人物须有name，不能确认的写未知并描述外观。")
        selected_ids(person.get("frame_ids"), frames, "visible_people.frame_ids", required=True)
        if not isinstance(person.get("identity_basis"), str) or not person["identity_basis"].strip():
            raise ModelFormatError("人物身份须说明依据，不能只由台词提及确定出镜。")
    result = {**item, "protocol": ANALYSIS_PROTOCOL, "summary": row["summary"].strip(),
              "content_kind": row["content_kind"], "visible_people": people,
              "observations": observations, "coverage": "sampled" if frames else "subtitle_only",
              "analysis_state": "ready" if frames and row["content_kind"] != "unknown" else "uncertain"}
    for field in ("tags", "mentioned_people", "uncertainties"):
        entries = row.get(field, [])
        if not isinstance(entries, list) or any(not isinstance(v, str) for v in entries):
            raise ModelFormatError(f"{field}应为字符串列表。")
        result[field] = entries
    for field in ("period", "location"):
        if not isinstance(row.get(field, ""), str):
            raise ModelFormatError(f"{field}应为字符串，未知可留空。")
        result[field] = row.get(field, "")
    result["revision"] = stable_hash(result)
    return result


def correction_result(item, changes):
    """One validator for form drafts, JSON imports and committed corrections."""
    allowed = {"visible_people", "period", "location", "summary", "tags", "content_kind",
               "duplicate_group", "uncertainties", "observations"}
    if not isinstance(changes, dict) or set(changes) - allowed:
        raise ModelFormatError("仅可编辑观察字段，镜头编号、时间和来源由程序保留。")
    for field, fields in (("observations", {"kind", "text", "frame_ids", "cue_ids"}),
                          ("visible_people", {"name", "period", "identity_basis", "frame_ids"})):
        if field in changes and (not isinstance(changes[field], list) or any(
                not isinstance(row, dict) or set(row)-fields for row in changes[field])):
            raise ModelFormatError(f"{field}仅可编辑说明和选择当前证据，不能手工修改证据编号或时间。")
    if not isinstance(changes.get("duplicate_group", ""), str):
        raise ModelFormatError("重复画面组须为文本。")
    result = observation_result({**item, **changes, "range_id": item["id"]}, item)
    result["duplicate_group"] = changes.get("duplicate_group", item.get("duplicate_group", ""))
    return result


def content_result(value, ranges, context_ids=()):
    """Semantic groups cover current ranges once, and may explicitly continue a prior group."""
    from app.models.library import media_id
    rows = value.get("segments") if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ModelFormatError("segments须为内容段列表。")
    order = {r["id"]: i for i, r in enumerate(ranges)}
    by_id = {r["id"]: r for r in ranges}
    covered, result = [], []
    for row in rows:
        ids = selected_ids(row.get("range_ids") if isinstance(row, dict) else None, order, "range_ids", required=True)
        indices = [order[i] for i in ids]
        if indices != list(range(indices[0], indices[-1]+1)):
            raise ModelFormatError("内容段须按原片连续范围组成，不能跳跃镜头。")
        if not isinstance(row.get("summary"), str) or not row["summary"].strip():
            raise ModelFormatError("内容段须有summary。")
        previous = row.get("continues_from", "")
        if previous and (previous not in context_ids or indices[0] != 0):
            raise ModelFormatError("continues_from只可引用紧邻上批的范围ID。")
        result.append({"id": media_id("E", ids), "range_ids": ids,
                       "start": by_id[ids[0]]["start"], "end": by_id[ids[-1]]["end"],
                       "summary": row["summary"], "title": str(row.get("title", "")),
                       "continues_from": previous})
        covered.extend(ids)
    if covered != list(order):
        raise ModelFormatError("内容段须按顺序恰好覆盖本批全部range_id。")
    return result
