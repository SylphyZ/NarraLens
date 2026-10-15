"""Arrange independent visual cuts under continuous narration."""

import math
import re
from bisect import bisect_right
from dataclasses import replace

from app.models.timeline import VisualShot


VISUAL_POLICY_VERSION = "unique-source-credits-v1"


class VisualCapacityError(ValueError):
    def __init__(self, segment_id, missing_seconds):
        self.missing_seconds = missing_seconds
        super().__init__(f"段落 {segment_id} 缺少 {missing_seconds:.2f} 秒合格且未用的自然镜头；请增加素材后从3. 寻找画面继续。")


def source_shot_key(item):
    """Analysis windows and library registrations do not define physical shots."""
    return (item["fingerprint"], round(item["natural_start"] * item["fps"]), round(item["natural_end"] * item["fps"]))


def unused_source(item, used):
    return not any(previous["fingerprint"] == item["fingerprint"] and (
        source_shot_key(previous) == source_shot_key(item) or
        min(previous["natural_end"], item["natural_end"]) - max(previous["natural_start"], item["natural_start"]) > .000001 or
        min(previous["source_end"], item["source_end"]) - max(previous["source_start"], item["source_start"]) > .000001
    ) for previous in used)


def natural_candidates(candidates, scene_boundaries, assets):
    """Split windows at real cuts, retaining the full cut identity for deduplication."""
    asset_map = {asset["id"]: asset for asset in assets}
    result = []
    for candidate in candidates:
        asset = asset_map[candidate["asset_id"]]
        if not asset.get("fingerprint") or not math.isfinite(asset["fps"]) or asset["fps"] <= 0:
            raise ValueError("素材指纹或帧率缺失，请更新素材登记后继续。")
        for left, right in zip(scene_boundaries[asset["id"]], scene_boundaries[asset["id"]][1:]):
            start = max(left, candidate["source_start"]) + .12
            end = min(right, candidate["source_end"]) - .12
            if end - start >= 2:
                result.append({**candidate, "source_start": start, "source_end": end,
                               "fingerprint": asset["fingerprint"], "fps": asset["fps"],
                               "natural_start": left, "natural_end": right})
    return result


def make_visual_policy(assets, boundaries, approved, forbidden=()):
    used_ids = {item["asset_id"] for item in approved}
    fingerprints = {item["fingerprint"] for item in approved}
    return {"version": VISUAL_POLICY_VERSION, "approved": approved,
            "forbidden": [item for item in forbidden if item["fingerprint"] in fingerprints],
            "sources": {asset["id"]: {**{key: asset[key] for key in ("fingerprint", "fps", "duration")},
                                      "boundaries": boundaries[asset["id"]]}
                        for asset in assets if asset["id"] in used_ids}}


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


def plan_visual_timeline(segments, candidates_by_segment, target_shot_seconds=7.0, scene_boundaries=None, *, reservations=(), used=()):
    shots = []
    # A failed attempt must not consume the caller's ledger.
    allocated = list(used)
    if not math.isfinite(target_shot_seconds) or target_shot_seconds <= 0:
        raise ValueError("目标镜头时长必须是有效正数。")

    for segment in segments:
        intervals = _visual_intervals(segment, reservations)
        candidates = candidates_by_segment.get(segment.id, [])
        candidates = [item for item in candidates if item.get("visual_policy") == VISUAL_POLICY_VERSION
                      and item.get("usable") is True and item.get("content_kind") == "story"]
        for start, end, reserved in intervals:
            if reserved is not None:
                shots.append(reserved)
                continue
            if end - start < min(1.0, segment.audio_end - segment.audio_start) - .01:
                raise ValueError(f"段落 {segment.id} 仅剩 {end-start:.3f} 秒原片残段，请扩大相邻动画范围后重选。")
            position = start
            while position < end - .001:
                remaining = end - position
                pool = [item for item in candidates if unused_source(item, allocated)]
                if not pool:
                    missing = remaining + sum(right-left for left, right, row in intervals if left >= end and row is None)
                    raise VisualCapacityError(segment.id, missing)
                capacity = min(max(2, target_shot_seconds), max(item["source_end"] - item["source_start"] for item in pool))
                desired = remaining / max(1, math.ceil(remaining / capacity))
                eligible = [item for item in pool if item["source_end"] - item["source_start"] >= desired - .001]
                candidate = max(eligible, key=lambda item: item.get("confidence", 0)) if eligible else \
                    max(pool, key=lambda item: (item["source_end"] - item["source_start"], item.get("confidence", 0)))
                available = candidate["source_end"] - candidate["source_start"]
                length = min(desired, remaining, available)
                if .001 < remaining - length < 1:
                    length = remaining - 1
                if length < .02:
                    raise ValueError(f"段落 {segment.id} 的原片区间太短，请扩大相邻动画范围，避免几帧残段。")
                anchor = candidate.get("anchor", (candidate["source_start"] + candidate["source_end"]) / 2)
                source_start = max(candidate["source_start"], min(anchor - length / 2, candidate["source_end"] - length))
                shots.append(VisualShot(
                    candidate["asset_id"], candidate["scene_id"], candidate["video_path"], source_start, source_start + length,
                    position, position + length, segment.id,
                    candidate.get("reason", ""), bool(candidate.get("weak_match", True)),
                ))
                allocated.append(candidate)
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


def validate_timeline(segments, shots, assets, scene_boundaries=None, *, policy=None):
    if not segments or not shots:
        raise ValueError("时间线为空。")
    asset_map = {item["id"]: item for item in assets}
    segment_map = {item.id: item for item in segments}
    if policy is not None:
        if not isinstance(policy, dict) or policy.get("version") != VISUAL_POLICY_VERSION:
            raise ValueError("镜头策略已更新，请从3. 寻找画面继续；已有配音与字幕可复用。")
        if not isinstance(policy.get("sources"), dict) or not isinstance(policy.get("approved"), list) or not isinstance(policy.get("forbidden", []), list):
            raise ValueError("镜头审核记录缺失，请从3. 寻找画面继续。")
        scene_boundaries = {identifier: source["boundaries"] for identifier, source in policy["sources"].items()}
    used = []
    source_ranges = {}
    motion_groups = [group for group in coalesce_motion_shots(shots) if re.fullmatch(r"motion-[a-f0-9]{16}", group.asset_id)
                     and group.timeline_end-group.timeline_start >= 4-.01] if scene_boundaries is not None else []
    position = 0.0
    for shot in shots:
        asset = asset_map.get(shot.asset_id)
        segment = segment_map.get(shot.segment_id)
        if any(not math.isfinite(number) for number in (shot.source_start, shot.source_end, shot.timeline_start, shot.timeline_end)):
            raise ValueError("镜头时间不是有效数字。")
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
        is_motion = bool(re.fullmatch(r"motion-[a-f0-9]{16}", shot.asset_id))
        if not is_motion:
            identity = asset.get("fingerprint") or shot.video_path
            ranges = source_ranges.setdefault(identity, [])
            if any(min(shot.source_end, right) - max(shot.source_start, left) > .000001 for left, right in ranges):
                raise ValueError(f"段落 {shot.segment_id} 重复使用同一来源画面，请从3. 寻找画面重新编排。")
            ranges.append((shot.source_start, shot.source_end))
            if policy is not None:
                source = policy["sources"].get(shot.asset_id, {})
                if not asset.get("fingerprint") or source.get("fingerprint") != asset["fingerprint"] or \
                        source.get("fps") != asset.get("fps") or abs(source.get("duration", 0) - asset["duration"]) > .01:
                    raise ValueError("素材身份与镜头审核记录不一致，请从3. 寻找画面继续。")
                approved = next((item for item in policy["approved"] if item.get("asset_id") == shot.asset_id
                                 and item.get("scene_id") == shot.scene_id and item.get("usable") is True
                                 and item.get("content_kind") == "story" and item.get("visual_policy") == VISUAL_POLICY_VERSION
                                 and item["source_start"] <= shot.source_start + .000001
                                 and item["source_end"] >= shot.source_end - .000001), None)
                if approved is None or approved["fingerprint"] != asset["fingerprint"]:
                    raise ValueError("镜头缺少实际来源范围的合格画面记录，请从3. 寻找画面继续。")
                if approved["fps"] != asset["fps"] or not (0 <= approved["natural_start"] <= approved["source_start"] <
                        approved["source_end"] <= approved["natural_end"] <= asset["duration"] + .01):
                    raise ValueError("镜头审核来源范围无效，请从3. 寻找画面继续。")
                if any(item["fingerprint"] == identity and min(item["source_end"], shot.source_end) -
                       max(item["source_start"], shot.source_start) > .000001 for item in policy.get("forbidden", [])):
                    raise ValueError("镜头与已排除的名单或不明画面范围相交，请从3. 寻找画面继续。")
            boundaries = (scene_boundaries or {}).get(shot.asset_id, [])
            if boundaries:
                index = bisect_right(boundaries, shot.source_start) - 1
                if index < 0 or index + 1 >= len(boundaries) or shot.source_end > boundaries[index + 1] + .000001:
                    raise ValueError("裁剪范围跨越原片切镜边界。")
                if policy is not None and (abs(approved["natural_start"] - boundaries[index]) > .000001 or
                                           abs(approved["natural_end"] - boundaries[index + 1]) > .000001):
                    raise ValueError("镜头审核与真实切镜范围不一致，请从3. 寻找画面继续。")
                physical = {"fingerprint": identity, "fps": asset.get("fps", 30),
                            "natural_start": boundaries[index], "natural_end": boundaries[index + 1],
                            "source_start": shot.source_start, "source_end": shot.source_end}
                if not unused_source(physical, used):
                    raise ValueError(f"段落 {shot.segment_id} 再次使用同一自然镜头，请从3. 寻找画面重新编排。")
                used.append(physical)
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
