"""Arrange independent visual cuts under continuous narration."""

import math
import re
from bisect import bisect_right
from dataclasses import replace

from app.models.timeline import VisualShot
from app.services.library.store import stable_hash


VISUAL_POLICY_VERSION = "unique-source-credits-v1"
MATCH_POLICY_VERSION = "semantic-visual-units-v1"
PROPOSAL_VERSION = "token-shortlist-v1"
COVERAGE_VERSION = "atomic-fact-coverage-v1"


def candidate_facts(candidate, requirement, start=None, end=None):
    """Count only evidence lying inside the source range that will actually be used."""
    coverage = candidate.get("coverage", {})
    if requirement.get("coverage_version") != COVERAGE_VERSION or coverage.get("version") != COVERAGE_VERSION or coverage.get("requirement_hash") != stable_hash(requirement):
        return []
    start = candidate["source_start"] if start is None else start
    end = candidate["source_end"] if end is None else end
    valid = {fact["id"] for fact in requirement.get("facts", [])}
    return list(dict.fromkeys(row["fact_id"] for row in coverage.get("supports", []) if row.get("fact_id") in valid
                and type(row.get("start")) in (int,float) and type(row.get("end")) in (int,float)
                and math.isfinite(row["start"]) and math.isfinite(row["end"])
                and start-.000001 <= row["start"] <= row["end"] <= end+.000001))


def _evidence_window(candidate, length, missing=()):
    coverage = candidate.get("coverage", {})
    rows = [r for r in coverage.get("supports", []) if r.get("fact_id") in missing]
    rows.sort(key=lambda r: list(missing).index(r["fact_id"]))
    if not rows:
        rows = list(coverage.get("supports", []))[:1]
    if not rows and coverage.get("motif_match") is True:
        rows = [coverage["motif_range"]] if coverage.get("motif_range") else []
    anchor = candidate.get("anchor", (candidate["source_start"]+candidate["source_end"])/2)
    if rows:
        left, right = min(row["start"] for row in rows), max(row["end"] for row in rows)
        length = max(length, right-left)
        start = max(candidate["source_start"], min(anchor-length/2, left, candidate["source_end"]-length))
        start = max(start, right-length)
    else:
        start = max(candidate["source_start"], min(anchor-length/2, candidate["source_end"]-length))
    return start, length


def fact_sequence(shots, approved):
    """First actual evidence occurrence, measured in the used source/timeline range."""
    by_key = {c.get("review_key",c.get("_review_key")):c for c in approved}
    first = {}
    for shot in shots:
        candidate = by_key.get(shot.review_key,{})
        for support in candidate.get("coverage",{}).get("supports",[]):
            identifier = support["fact_id"]
            if identifier in shot.fact_ids:
                point = shot.timeline_start + support["start"]-shot.source_start
                first[identifier] = min(first.get(identifier,point),point)
    return sorted(first,key=lambda identifier:(first[identifier],identifier))


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
    common = (candidate.get("pending_review") is not True and candidate.get("match_policy") == MATCH_POLICY_VERSION and
            isinstance(conflicts, list) and not conflicts and
            isinstance(evidence, list) and bool(evidence) and all(isinstance(x, str) and x.strip() for x in evidence) and
            type(candidate.get("visual_checked")) is bool and type(score) in (int, float) and math.isfinite(score) and 0 < score <= 1)
    if requirement.get("coverage_version") == COVERAGE_VERSION:
        coverage = candidate.get("coverage", {})
        if not common or coverage.get("version") != COVERAGE_VERSION or coverage.get("requirement_hash") != stable_hash(requirement) or coverage.get("constraints_match") is not True:
            return False
        if requirement.get("facts"):
            return relation == "direct" and bool(candidate_facts(candidate, requirement))
        return (requirement.get("kind") in {"state", "abstract", "transition"} and relation in {"direct", "supporting"}
                and coverage.get("motif_match") is True and isinstance(coverage.get("motif_range"), dict))
    return common and candidate.get("facts_match") is True and (relation == "direct" or (relation == "supporting" and requirement.get("kind") in {"state", "abstract", "transition"}))


def matching_policy(plan):
    return {"version": MATCH_POLICY_VERSION, "plan_key": plan["key"], "text_hash": plan["text_hash"],
            "audio_key": plan["audio_key"], "timing_method": plan["timing_method"],
            "units": [{**{k: unit[k] for k in ("id", "segment_id", "text_start", "text_end", "start", "end")},
                       "requirement": ({**unit["requirement"]} if unit["requirement"].get("coverage_version") == COVERAGE_VERSION else
                                       {k: unit["requirement"][k] for k in ("kind", "required_facts")})} for unit in plan["units"]]}


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


def _allocate_visual_timeline(segments, candidates_by_segment, target_shot_seconds, *, visual_units, reservations, used, preview=False):
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
    def eligible(item, unit):
        if item.get("visual_unit_id") != unit["id"]:
            return False
        approved = (item.get("visual_policy") == VISUAL_POLICY_VERSION and item.get("usable") is True
                    and item.get("content_kind") == "story" and match_allowed(item, unit["requirement"]))
        return approved or (preview and item.get("proposal_version") == PROPOSAL_VERSION
                            and item.get("pending_review") is True and item.get("usable") is not False)

    pools = {unit["id"]: [item for item in candidates_by_segment.get(unit["id"], []) if eligible(item, unit)] for unit in visual_units}

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
        required = [f["id"] for f in unit["requirement"].get("facts", [])] if unit["requirement"].get("coverage_version") == COVERAGE_VERSION else []
        covered = set()
        for interval_index, (start, end) in enumerate(intervals):
            position = start
            while position < end - .001:
                remaining = end - position
                minimum = min(1.0, segment.audio_end - segment.audio_start)
                if remaining < minimum-.001 and trial and abs(trial[-1].timeline_end-position) < .001:
                    previous, source = trial[-1], trial_used[-1]
                    length = previous.timeline_end-previous.timeline_start+remaining
                    if length <= source["source_end"]-source["source_start"]+.000001:
                        left = max(source["source_start"], min(previous.source_start, source["source_end"]-length))
                        trial[-1] = replace(previous, source_start=left, source_end=left+length, timeline_end=end,
                                            fact_ids=candidate_facts(source,unit["requirement"],left,left+length))
                        covered = {identifier for shot in trial for identifier in shot.fact_ids}
                        position = end
                        continue
                missing = [identifier for identifier in required if identifier not in covered]
                def coverage_rank(item):
                    facts = candidate_facts(item, unit["requirement"])
                    return (3 if missing and missing[0] in facts else 2 if missing and item.get("pending_review") else 1 if not missing else 0)
                pool = sorted((item for item in candidates if unused_source(item, trial_used)),
                              key=lambda item: (coverage_rank(item), not item.get("pending_review", False), item.get("relation") == "direct",
                                                item.get("confidence", item.get("proposal_score", 0))), reverse=True)
                chosen = None
                for candidate in pool if remaining >= minimum-.001 else []:
                    available = candidate["source_end"] - candidate["source_start"]
                    prefix = []
                    possible = set(candidate_facts(candidate,unit["requirement"]))
                    for identifier in missing:
                        if identifier not in possible:
                            break
                        prefix.append(identifier)
                    if missing and not prefix and candidate.get("pending_review") is not True:
                        continue
                    total_remaining = remaining+sum(b-a for a,b in intervals[interval_index+1:])
                    reserve = minimum*(len(missing)-len(prefix)) if missing and candidate.get("pending_review") is not True else 0
                    maximum_length = min(available,remaining,total_remaining-reserve)
                    other_capacity = sum(item["source_end"]-item["source_start"] for item in pool
                                         if source_shot_key(item) != source_shot_key(candidate))
                    # A rhythm target must not discard the only matching shot's usable seconds.
                    minimum_length = max(minimum, remaining-other_capacity)
                    rhythm = min(target_shot_seconds,total_remaining/max(1,len(missing))*max(1,len(prefix))) if missing and prefix else target_shot_seconds
                    length = min(max(minimum_length, rhythm), maximum_length)
                    nearby = [point-position for point in unit.get("cut_times", []) if
                              minimum_length <= point-position <= maximum_length and abs(point-position-length) <= 1 and
                              (end-point >= minimum-.001 or abs(end-point) <= .001)]
                    if nearby:
                        length = min(nearby, key=lambda value: abs(value-length))
                    if .001 < remaining-length < minimum:
                        length = remaining if available >= remaining else remaining-minimum
                    if length >= minimum-.001:
                        source_left, proposed_length = _evidence_window(candidate, length, prefix)
                        if proposed_length > maximum_length+.000001 or source_left < candidate["source_start"]-.000001 or source_left+proposed_length > candidate["source_end"]+.000001:
                            continue
                        if .001 < remaining-proposed_length < minimum:
                            proposed_length = remaining if available >= remaining else remaining-minimum
                            source_left, proposed_length = _evidence_window(candidate, proposed_length, prefix)
                            if proposed_length > maximum_length+.000001 or .001 < remaining-proposed_length < minimum:
                                continue
                        chosen = candidate, proposed_length, source_left
                        break
                if chosen is None:
                    failure = {"visual_unit_id": unit["id"], "segment_id": unit["segment_id"], "start": position,
                               "category": "short_interval" if end-start < minimum-.001 else "material",
                               "end": unit["end"], "missing_seconds": round(remaining +
                               sum(b-a for a, b in intervals[interval_index+1:]), 3),
                               "reason": "原片残段不足1秒，请合并相邻句意范围或扩大相邻动画范围。" if end-start < minimum-.001 else
                                         "缺少通过当前句意检查、事实对应且未使用的自然镜头。"}
                    if missing and failure["category"] != "short_interval":
                        failure.update(category="fact_coverage",missing_facts=missing,
                                       reason="当前可分配范围不能按原文顺序覆盖事实/最短镜头时长："+"、".join(missing))
                    break
                candidate, length, source_start = chosen
                fact_ids = candidate_facts(candidate, unit["requirement"], source_start, source_start+length)
                trial.append(VisualShot(
                    candidate["asset_id"], candidate["scene_id"], candidate["video_path"], source_start, source_start + length,
                    position, position + length, segment.id,
                    candidate.get("reason", ""), bool(candidate.get("weak_match", True)), unit["id"], fact_ids, candidate.get("_review_key", ""),
                ))
                trial_used.append(candidate)
                covered.update(fact_ids)
                position += length
            if failure:
                break
        if not failure and intervals and required and set(required)-covered:
            missing = [identifier for identifier in required if identifier not in covered]
            failure = {"visual_unit_id":unit["id"], "segment_id":unit["segment_id"], "start":unit["start"], "end":unit["end"],
                       "category":"fact_coverage", "missing_seconds":round(sum(b-a for a,b in intervals),3),
                       "missing_facts":missing, "reason":"实际采用镜头尚未覆盖具体事实："+"、".join(missing)}
        if not failure and intervals and required and fact_sequence(trial,trial_used) != required:
            failure = {"visual_unit_id":unit["id"],"segment_id":unit["segment_id"],"start":unit["start"],"end":unit["end"],
                       "category":"fact_order","missing_seconds":round(sum(b-a for a,b in intervals),3),"reason":"具体事实首次出现顺序与原文不符，需调整镜头编排。"}
        if failure:
            gaps.append(failure)
        if not failure or preview:
            shots.extend(trial)
            allocated = trial_used
    return sorted(shots, key=lambda row: row.timeline_start), gaps


def propose_visual_timeline(segments, candidates_by_unit, target_shot_seconds=7.0, *, visual_units, reservations=()):
    """A partial, unapproved proposal may choose what to inspect, never what to render."""
    return _allocate_visual_timeline(segments, candidates_by_unit, target_shot_seconds,
                                    visual_units=visual_units, reservations=reservations, used=(), preview=True)


def plan_visual_timeline(segments, candidates_by_segment, target_shot_seconds=7.0, scene_boundaries=None, *, visual_units=(), reservations=(), used=()):
    shots, gaps = _allocate_visual_timeline(segments, candidates_by_segment, target_shot_seconds,
        visual_units=visual_units, reservations=reservations, used=used)
    if gaps:
        raise VisualCapacityError(gaps[0]["visual_unit_id"], sum(row["missing_seconds"] for row in gaps), gaps=gaps)
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
        matching = policy.get("matching", {})
        if not isinstance(matching, dict) or matching.get("version") != MATCH_POLICY_VERSION or not isinstance(matching.get("units"), list):
            raise ValueError("此时间轴尚未执行2.7句意匹配，请从3. 寻找画面继续；配音与字幕可复用。")
        unit_map = {unit["id"]: unit for unit in matching["units"]}
        scene_boundaries = {identifier: source["boundaries"] for identifier, source in policy["sources"].items()}
    used = []
    fact_coverage = {identifier:set() for identifier in unit_map} if policy is not None else {}
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
                if unit["requirement"].get("coverage_version") == COVERAGE_VERSION:
                    facts = candidate_facts(approved, unit["requirement"], shot.source_start, shot.source_end)
                    if shot.fact_ids != facts or not shot.review_key or shot.review_key != approved.get("review_key"):
                        raise ValueError("镜头事实记录与实际采用范围不一致，请从3. 寻找画面继续。")
                    if unit["requirement"].get("facts") and not facts:
                        raise ValueError("当前具体事实镜头的实际采用范围不含已核对的事实依据，请从3. 寻找画面继续。")
                    if not unit["requirement"].get("facts"):
                        motif = approved["coverage"]["motif_range"]
                        if not shot.source_start-.000001 <= motif["start"] <= motif["end"] <= shot.source_end+.000001:
                            raise ValueError("配图依据未包含在实际采用范围内，请从3. 寻找画面继续。")
                    fact_coverage[unit["id"]].update(facts)
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
    if policy is not None:
        for identifier, unit in unit_map.items():
            # Fully animated units have no original-footage evidence obligation.
            original = [s for s in shots if s.visual_unit_id == identifier and not re.fullmatch(r"motion-[a-f0-9]{16}",s.asset_id)]
            if not original:
                continue
            missing = {f["id"] for f in unit["requirement"].get("facts",[])}-fact_coverage[identifier]
            if unit["requirement"].get("coverage_version") == COVERAGE_VERSION and missing:
                raise ValueError(f"画面单元{identifier}的实际镜头未覆盖事实{sorted(missing)}，请从3. 寻找画面继续。")
            if unit["requirement"].get("coverage_version") == COVERAGE_VERSION and unit["requirement"].get("facts") and fact_sequence(original,policy["approved"]) != [f["id"] for f in unit["requirement"]["facts"]]:
                raise ValueError(f"画面单元{identifier}的事实出现顺序与原文不符，请从3. 寻找画面继续。")
