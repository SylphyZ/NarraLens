"""Arrange independent visual cuts under continuous narration."""

import math
import re
from bisect import bisect_right
from dataclasses import replace

from app.models.timeline import VisualShot


VISUAL_POLICY_VERSION = "unique-source-credits-v1"
MATCH_POLICY_VERSION = "semantic-visual-units-v1"


class VisualCapacityError(ValueError):
    def __init__(self, segment_id, missing_seconds, *, gaps=()):
        self.missing_seconds = missing_seconds
        self.gaps = list(gaps)
        super().__init__(f"画面单元 {segment_id} 缺少 {missing_seconds:.2f} 秒内容匹配且未用的自然镜头；请增加素材或调整动画后从3. 寻找画面继续。")


def match_allowed(candidate, requirement):
    """Content approval is independent of source usability and ranking scores."""
    relation = candidate.get("relation")
    evidence, conflicts = candidate.get("evidence"), candidate.get("conflicts")
    score = candidate.get("confidence")
    return (candidate.get("match_policy") == MATCH_POLICY_VERSION and
            candidate.get("facts_match") is True and isinstance(conflicts, list) and not conflicts and
            isinstance(evidence, list) and bool(evidence) and all(isinstance(x, str) and x.strip() for x in evidence) and
            type(candidate.get("visual_checked")) is bool and type(score) in (int, float) and math.isfinite(score) and 0 < score <= 1 and
            (relation == "direct" or (relation == "supporting" and requirement.get("kind") in {"state", "abstract", "transition"})))


def matching_policy(plan):
    return {"version": MATCH_POLICY_VERSION, "plan_key": plan["key"], "text_hash": plan["text_hash"],
            "audio_key": plan["audio_key"], "timing_method": plan["timing_method"],
            "units": [{**{k: unit[k] for k in ("id", "segment_id", "text_start", "text_end", "start", "end")},
                       "requirement": {k: unit["requirement"][k] for k in ("kind", "required_facts")}} for unit in plan["units"]]}


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


def make_visual_policy(assets, boundaries, approved, forbidden=(), *, visual_plan):
    used_ids = {item["asset_id"] for item in approved}
    fingerprints = {item["fingerprint"] for item in approved}
    return {"version": VISUAL_POLICY_VERSION, "approved": approved, "matching": matching_policy(visual_plan),
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


def visual_unit_intervals(unit, reservations):
    intervals, position = [], unit["start"]
    for shot in sorted(reservations, key=lambda row: row.timeline_start):
        if shot.timeline_end <= position or shot.timeline_start >= unit["end"]:
            continue
        if shot.timeline_start > position:
            intervals.append((position, min(shot.timeline_start, unit["end"])))
        position = max(position, min(shot.timeline_end, unit["end"]))
    if position < unit["end"] - .001:
        intervals.append((position, unit["end"]))
    return intervals


def plan_visual_timeline(segments, candidates_by_segment, target_shot_seconds=7.0, scene_boundaries=None, *, visual_units=(), reservations=(), used=()):
    """Allocate scarce factual footage first; only a complete attempt is committed."""
    shots, gaps = list(reservations), []
    allocated = list(used)
    if not math.isfinite(target_shot_seconds) or target_shot_seconds <= 0:
        raise ValueError("目标镜头时长必须是有效正数。")
    if not visual_units:
        raise ValueError("缺少句意画面单元，请从3. 寻找画面继续。")
    owners = {s.id: s for s in segments}
    for segment in segments:
        _visual_intervals(segment, reservations)
    pools = {unit["id"]: [item for item in candidates_by_segment.get(unit["id"], [])
             if item.get("visual_policy") == VISUAL_POLICY_VERSION and item.get("usable") is True
             and item.get("content_kind") == "story" and item.get("visual_unit_id") == unit["id"]
             and match_allowed(item, unit["requirement"])] for unit in visual_units}

    def priority(unit):
        needed = sum(right-left for left, right in visual_unit_intervals(unit, reservations))
        unique = {source_shot_key(item): item for item in pools[unit["id"]]}
        capacity = sum(min(target_shot_seconds, item["source_end"]-item["source_start"]) for item in unique.values())
        return ({"evidence": 0, "state": 1, "abstract": 2, "transition": 3}[unit["requirement"]["kind"]],
                capacity / max(.001, needed), unit["start"])

    for unit in sorted(visual_units, key=priority):
        segment = owners[unit["segment_id"]]
        intervals = visual_unit_intervals(unit, reservations)
        candidates = pools[unit["id"]]
        trial, trial_used, failure = [], list(allocated), None
        for interval_index, (start, end) in enumerate(intervals):
            position = start
            while position < end - .001:
                remaining = end - position
                minimum = min(1.0, segment.audio_end - segment.audio_start)
                pool = sorted((item for item in candidates if unused_source(item, trial_used)),
                              key=lambda item: (item["relation"] == "direct", item["confidence"]), reverse=True)
                chosen = None
                for candidate in pool:
                    available = candidate["source_end"] - candidate["source_start"]
                    other_capacity = sum(item["source_end"]-item["source_start"] for item in pool
                                         if source_shot_key(item) != source_shot_key(candidate))
                    # A rhythm target must not discard the only matching shot's usable seconds.
                    minimum_length = max(minimum, remaining-other_capacity)
                    length = min(max(minimum_length, target_shot_seconds), available, remaining)
                    nearby = [point-position for point in unit.get("cut_times", []) if
                              minimum_length <= point-position <= min(available, remaining) and abs(point-position-length) <= 1 and
                              (end-point >= minimum-.001 or abs(end-point) <= .001)]
                    if nearby:
                        length = min(nearby, key=lambda value: abs(value-length))
                    if .001 < remaining-length < minimum:
                        length = remaining if available >= remaining else remaining-minimum
                    if length >= minimum-.001:
                        chosen = candidate, length
                        break
                if chosen is None:
                    failure = {"visual_unit_id": unit["id"], "segment_id": unit["segment_id"], "start": position,
                               "category": "short_interval" if remaining < minimum-.001 else "material",
                               "end": unit["end"], "missing_seconds": round(remaining +
                               sum(b-a for a, b in intervals[interval_index+1:]), 3),
                               "reason": "原片残段不足1秒，请合并相邻句意范围或扩大相邻动画范围。" if remaining < minimum-.001 else
                                         "缺少通过当前句意检查、事实对应且未使用的自然镜头。"}
                    break
                candidate, length = chosen
                anchor = candidate.get("anchor", (candidate["source_start"] + candidate["source_end"]) / 2)
                source_start = max(candidate["source_start"], min(anchor - length / 2, candidate["source_end"] - length))
                trial.append(VisualShot(
                    candidate["asset_id"], candidate["scene_id"], candidate["video_path"], source_start, source_start + length,
                    position, position + length, segment.id,
                    candidate.get("reason", ""), bool(candidate.get("weak_match", True)), unit["id"],
                ))
                trial_used.append(candidate)
                position += length
            if failure:
                break
        if failure:
            gaps.append(failure)
        else:
            shots.extend(trial)
            allocated = trial_used
    if gaps:
        raise VisualCapacityError(gaps[0]["visual_unit_id"], sum(row["missing_seconds"] for row in gaps), gaps=gaps)
    return sorted(shots, key=lambda row: row.timeline_start)


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
        matching = policy.get("matching", {})
        if not isinstance(matching, dict) or matching.get("version") != MATCH_POLICY_VERSION or not isinstance(matching.get("units"), list):
            raise ValueError("此时间轴尚未执行2.7句意匹配，请从3. 寻找画面继续；配音与字幕可复用。")
        unit_map = {unit["id"]: unit for unit in matching["units"]}
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
                unit = unit_map.get(shot.visual_unit_id)
                if not unit or unit["segment_id"] != shot.segment_id or shot.timeline_start < unit["start"]-.001 or shot.timeline_end > unit["end"]+.001:
                    raise ValueError("镜头未绑定对应句意时间，请从3. 寻找画面继续。")
                source = policy["sources"].get(shot.asset_id, {})
                if not asset.get("fingerprint") or source.get("fingerprint") != asset["fingerprint"] or \
                        source.get("fps") != asset.get("fps") or abs(source.get("duration", 0) - asset["duration"]) > .01:
                    raise ValueError("素材身份与镜头审核记录不一致，请从3. 寻找画面继续。")
                approved = next((item for item in policy["approved"] if item.get("asset_id") == shot.asset_id
                                 and item.get("scene_id") == shot.scene_id and item.get("usable") is True
                                 and item.get("visual_unit_id") == shot.visual_unit_id and match_allowed(item, unit["requirement"])
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
