"""Arrange independent visual cuts under continuous narration."""

import math
import re
from dataclasses import replace

from app.models.timeline import VisualShot


def _visual_intervals(segment, reservations):
    intervals, position = [], segment.audio_start
    for shot in sorted((row for row in reservations if row.segment_id == segment.id), key=lambda row: row.timeline_start):
        if shot.timeline_start < position - .001 or shot.timeline_end > segment.audio_end + .001 or shot.timeline_end <= shot.timeline_start:
            raise ValueError("解释动画区间重叠或超出段落配音范围。")
        if shot.timeline_start > position + .001:
            intervals.append((position, shot.timeline_start, None))
        intervals.append((shot.timeline_start, shot.timeline_end, shot))
        position = shot.timeline_end
    if position < segment.audio_end - .001:
        intervals.append((position, segment.audio_end, None))
    return intervals


def plan_visual_timeline(segments, candidates_by_segment, target_shot_seconds=7.0, scene_boundaries=None, *, reservations=()):
    shots = []
    uses = {}
    used_ranges = {}
    last_key = None

    def key(item):
        return item.get("_shot_key", item["scene_id"])

    for segment in segments:
        intervals = _visual_intervals(segment, reservations)
        candidates = candidates_by_segment.get(segment.id, [])
        if scene_boundaries is not None:
            stable = []
            for candidate in candidates:
                boundaries = scene_boundaries.get(candidate["asset_id"], [])
                for left, right in zip(boundaries, boundaries[1:]):
                    start = max(left, candidate["source_start"]) + .12
                    end = min(right, candidate["source_end"]) - .12
                    if end - start >= 2:
                        stable.append({**candidate, "source_start": start, "source_end": end,
                                       "_shot_key": (candidate["scene_id"], left, right),
                                       "_anchor_distance": abs((start + end) / 2 - candidate.get("anchor", (start + end) / 2))})
            candidates = stable
        if not candidates and any(reserved is None for _, _, reserved in intervals):
            raise ValueError(f"段落 {segment.id} 缺少至少两秒的连续原片镜头，请增加可用素材。")
        for start, end, reserved in intervals:
            if reserved is not None:
                shots.append(reserved)
                last_key = reserved.scene_id
                continue
            capacity = min(max(2, target_shot_seconds), max(item["source_end"] - item["source_start"] for item in candidates))
            position = start
            while position < end - .001:
                remaining = end - position
                desired = remaining / max(1, math.ceil(remaining / capacity))
                eligible = [item for item in candidates if item["source_end"] - item["source_start"] >= min(desired, remaining) - .001]
                pool = eligible or candidates
                candidate = max(pool, key=lambda item: item.get("confidence", 0) - uses.get(key(item), 0) * .22
                                - (.5 if key(item) == last_key else 0) - min(.3, item.get("_anchor_distance", 0) / 400))
                available = candidate["source_end"] - candidate["source_start"]
                length = min(desired, remaining, available)
                if length < .02:
                    raise ValueError(f"段落 {segment.id} 的候选画面太短。")
                anchor = candidate.get("anchor", (candidate["source_start"] + candidate["source_end"]) / 2)
                preferred = max(candidate["source_start"], min(anchor - length / 2, candidate["source_end"] - length))
                ranges = used_ranges.setdefault(candidate["asset_id"], [])
                starts = {preferred, candidate["source_start"], candidate["source_end"] - length}
                for offset in range(math.ceil(available / length)):
                    starts.add(min(candidate["source_end"] - length, candidate["source_start"] + offset * length))
                def reuse_cost(start):
                    overlap = sum(max(0, min(start + length, previous_end) - max(start, previous_start)) for previous_start, previous_end in ranges)
                    return overlap * 100 + abs(start - preferred)
                source_start = min(starts, key=reuse_cost)
                repeated = uses.get(key(candidate), 0) > 0
                shots.append(VisualShot(
                    candidate["asset_id"], candidate["scene_id"], candidate["video_path"], source_start, source_start + length,
                    position, position + length, segment.id,
                    candidate.get("reason", "") + ("；该场景被再次使用。" if repeated else ""),
                    bool(candidate.get("weak_match", True) or repeated),
                ))
                uses[key(candidate)] = uses.get(key(candidate), 0) + 1
                last_key = key(candidate)
                ranges.append((source_start, source_start + length))
                position += length
    return shots


def coalesce_motion_shots(shots):
    """Render adjacent slices of one animation as a continuous clip."""
    result = []
    for shot in shots:
        previous = result[-1] if result else None
        if previous and re.fullmatch(r"motion-[a-f0-9]{16}", shot.asset_id) and previous.asset_id == shot.asset_id and \
                previous.video_path == shot.video_path and abs(previous.timeline_end-shot.timeline_start) < .001 and \
                abs(previous.source_end-shot.source_start) < .001:
            result[-1] = replace(previous, source_end=shot.source_end, timeline_end=shot.timeline_end)
        else:
            result.append(shot)
    return result


def validate_timeline(segments, shots, assets, scene_boundaries=None):
    if not segments or not shots:
        raise ValueError("时间线为空。")
    asset_map = {item["id"]: item for item in assets}
    segment_map = {item.id: item for item in segments}
    motion_groups = [group for group in coalesce_motion_shots(shots) if re.fullmatch(r"motion-[a-f0-9]{16}", group.asset_id)
                     and group.timeline_end-group.timeline_start >= 4-.01] if scene_boundaries is not None else []
    position = 0.0
    for shot in shots:
        asset = asset_map.get(shot.asset_id)
        segment = segment_map.get(shot.segment_id)
        if not asset or not segment or shot.video_path != asset["path"]:
            raise ValueError("时间线引用了无效素材。")
        if abs(shot.timeline_start - position) > .01 or shot.timeline_end <= shot.timeline_start:
            raise ValueError("画面时间线存在缺口或重叠。")
        if not (0 <= shot.source_start < shot.source_end <= asset["duration"] + .01):
            raise ValueError("裁剪时间超出来源视频范围。")
        if abs((shot.source_end - shot.source_start) - (shot.timeline_end - shot.timeline_start)) > .01:
            raise ValueError("来源片段与成片片段时长不一致。")
        if shot.timeline_start < segment.audio_start - .01 or shot.timeline_end > segment.audio_end + .01:
            owners = [s for s in segments if s.audio_end > shot.timeline_start and s.audio_start < shot.timeline_end]
            if not re.fullmatch(r"motion-[a-f0-9]{16}", shot.asset_id) or not owners or owners[0].id != shot.segment_id:
                raise ValueError("镜头超出对应配音段落。")
            covered = shot.timeline_start
            for owner in owners:
                left, right = max(covered, owner.audio_start), min(shot.timeline_end, owner.audio_end)
                if abs(left-covered) > .01 or right <= left:
                    raise ValueError("连续动画未被配音段落完整覆盖。")
                covered = right
            if abs(covered-shot.timeline_end) > .01:
                raise ValueError("连续动画超出配音范围。")
        if scene_boundaries is not None:
            continuous_motion = any(group.asset_id == shot.asset_id and group.timeline_start <= shot.timeline_start + .001
                                    and group.timeline_end >= shot.timeline_end - .001 for group in motion_groups)
            if not continuous_motion and shot.timeline_end - shot.timeline_start < min(1.0, segment.audio_end - segment.audio_start) - .01:
                raise ValueError("镜头过短，时间线需要重新均衡。")
            if any(shot.source_start + .001 < boundary < shot.source_end - .001
                   for boundary in scene_boundaries.get(shot.asset_id, [])):
                raise ValueError("裁剪范围跨越原片切镜边界。")
        position = shot.timeline_end
    if abs(position - segments[-1].audio_end) > .01:
        raise ValueError("画面未完整覆盖配音。")
