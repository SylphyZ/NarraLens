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
        for item in clips[:6]:
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
