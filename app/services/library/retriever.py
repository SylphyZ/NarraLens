"""Local lexical/semantic-label retrieval followed by bounded model review."""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pysrt

from app.services.library.frame_extractor import extract_frames
from app.services.library.runner import check_cancel
from app.services.library.store import library_ids, stable_hash, write_json, read_json
from app.services.subtitle_segmentation import _tokenizer
from app.services.library.validation import confidence, frame_index, ModelFormatError
from app.services.timeline_planner import (
    VISUAL_POLICY_VERSION, MATCH_POLICY_VERSION, PROPOSAL_VERSION, match_allowed, natural_candidates, source_shot_key, unused_source,
)


SUBTITLE_EVIDENCE_VERSION = "subtitle-evidence-ids-v1"
REVIEW_SCOPE_VERSION = "source-usability-v1"


def _credits_description(scene):
    """Only definite positive descriptions exclude a whole window without rechecking."""
    def positive(text):
        for clause in re.split(r"[，。；\n.!;]", str(text)):
            if re.search(r"没有|未见|不是|不含|并非|无(?:滚动|演职员|片尾)|\b(?:no|not|without)\b", clause, re.I):
                continue
            if re.search(r"演职员|职员表|制作人员名单|片尾名单|スタッフ|エンディングクレジット|\b(?:credits|credit roll|staff list)\b", clause, re.I) or \
                    ("黑底" in clause and "名单" in clause):
                return True
        return False
    visual = scene.get("visual") or {}
    frames = visual.get("frames") or []
    if frames and len(frames) >= 3 and all(positive(item.get("description", "")) for item in frames):
        return "已有逐帧描述均明确为演职员名单画面。"
    if not frames and positive(visual.get("summary", "")):
        return "已有视觉描述明确为演职员名单，缺少可单独放行的剧情范围。"
    return ""


def _review_rows(value, expected):
    rows = value.get("clips") if isinstance(value, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get("scene_id"), str) for row in rows):
        raise ModelFormatError("clips须为对象列表，每条包含字符串scene_id；允许ID：" + str(sorted(expected)))
    identifiers = [row["scene_id"] for row in rows]
    duplicates = [identifier for identifier, count in Counter(identifiers).items() if count > 1]
    missing, unknown = sorted(set(expected) - set(identifiers)), sorted(set(identifiers) - set(expected))
    if len(rows) != len(expected) or duplicates or missing or unknown:
        raise ModelFormatError(f"每个scene_id须返回一次。缺失：{missing}；重复：{duplicates}；未知：{unknown}；允许ID：{sorted(expected)}")
    return rows


def _review_result(value, expected):
    for row in _review_rows(value, expected):
        if type(row.get("usable")) is not bool or row.get("content_kind") not in ("story", "credits", "mixed", "unknown") or \
                not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ModelFormatError("每个镜头须含 usable 布尔值、content_kind（story/credits/mixed/unknown）及非空 reason。")
        if row["usable"] and row["content_kind"] != "story":
            raise ModelFormatError("只有明确的 story 画面可标记 usable=true，名单、混合及不明画面必须为 false。")
        if not row["usable"] and row["content_kind"] == "story":
            raise ModelFormatError("story剧情画面应为usable=true；与原文无关只影响内容关系，不能标为全局不可用。")
    return value


def _match_result(value, expected, with_images, *, relevance=True):
    """Retry unidentified batches; reject identified invalid rows independently."""
    result = []
    all_ids = {entry["id"] for candidate in expected.values() for entry in candidate["subtitle_entries"]}
    for original in _review_rows(value, expected):
        row, issues = dict(original), []
        for name in ("_validation_issues", "_usability_valid", "_raw_response"):
            row.pop(name, None)

        def issue(field, category, message):
            issues.append({"field": field, "category": category, "message": message})

        usability_valid = True
        try:
            _review_result({"clips": [row]}, {row["scene_id"]})
        except ModelFormatError as exc:
            usability_valid = False
            issue("usable/content_kind/reason", "invalid_usability", str(exc))
        if with_images and row.get("frame_index_valid") is not True:
            usability_valid = False
            issue("frame_index", "invalid_frame", "视觉检查须返回合法frame_index，范围0至2。")
        if row.get("content_kind") == "unknown" and usability_valid:
            issue("content_kind", "unknown_visual", "画面尚不能可靠辨认，保留为未确认，不能作为跨需求禁用依据。")
        if relevance:
            if row.get("relation") not in ("direct", "supporting", "unrelated", "unknown"):
                issue("relation", "invalid_relation", "relation须为direct/supporting/unrelated/unknown。")
            if type(row.get("facts_match")) is not bool:
                issue("facts_match", "invalid_type", "facts_match须为布尔值。")
            for name in ("visual_evidence", "conflicts", "subtitle_ids"):
                items = row.get(name)
                if not isinstance(items, list) or len(items) > 3 or any(not isinstance(x, str) or not x.strip() for x in items):
                    issue(name, "invalid_list", f"{name}须为最多3项的非空字符串列表，可为空。")
            entries = {entry["id"]: entry["text"] for entry in expected[row["scene_id"]]["subtitle_entries"]}
            ids = row.get("subtitle_ids")
            valid_ids = isinstance(ids, list) and all(isinstance(x, str) for x in ids)
            if valid_ids:
                if len(ids) != len(set(ids)):
                    issue("subtitle_ids", "duplicate_id", "字幕编号不得重复。")
                foreign = [identifier for identifier in ids if identifier in all_ids and identifier not in entries]
                if foreign:
                    issue("subtitle_ids", "cross_candidate_id", "引用了其他候选的字幕编号：" + str(foreign))
                unknown = [identifier for identifier in ids if identifier not in all_ids]
                if unknown:
                    issue("subtitle_ids", "unknown_id", "字幕编号未在当前目录中给出：" + str(unknown))
            quotes = [entries[identifier] for identifier in ids] if valid_ids and all(x in entries for x in ids) else []
            visual = row.get("visual_evidence")
            evidence = list(visual) if isinstance(visual, list) and all(isinstance(x, str) and x.strip() for x in visual) else []
            evidence.extend("字幕：" + quote for quote in quotes)
            if row.get("relation") in ("direct", "supporting") and not evidence:
                issue("visual_evidence/subtitle_ids", "missing_evidence", "direct/supporting须有可核对的画面依据或本候选字幕编号。")
            # Free-form quotes/evidence from the model never become trusted evidence.
            row.update(evidence_protocol=SUBTITLE_EVIDENCE_VERSION, subtitle_quotes=quotes, evidence=evidence)
        row.update(_validation_issues=issues, _usability_valid=usability_valid, _raw_response=original)
        result.append(row)
    return {"clips": result}


class MaterialRetriever:
    def __init__(self, store, library_id):
        self.store = store
        self.library_ids = sorted(library_ids(library_id))
        self.library_names = {item["id"]: item["name"] for item in store.list_libraries()}
        self.scenes = store.scenes(self.library_ids)
        self.scene_map = {scene["id"]: scene for scene in self.scenes}
        self.scene_indices = {scene["id"]: index for index, scene in enumerate(self.scenes)}
        if not self.scenes:
            raise ValueError("素材库尚无可用的成功场景，请先建立或继续分析素材库。")
        self.version = stable_hash((MATCH_POLICY_VERSION, self.library_ids,
            [(s["id"], s["version"], s["analysis"], s["visual"], s["transcript"]) for s in self.scenes]))
        self.documents = []
        self.subtitles = {}
        self.statistics = {}
        self.unconfirmed = set()
        self.recorded = set()
        self.rejected_matches = set()
        self.rejected_scenes = set()
        self.rejected_ranges = set()
        self.forbidden = []
        self.exclusion_keys = set()
        self.source_libraries = defaultdict(set)
        for scene in self.scenes:
            self.source_libraries[scene["fingerprint"]].add(scene["library_id"])
            if _credits_description(scene):
                self.rejected_scenes.add(scene["id"])
                self.forbidden.append({"fingerprint": scene["fingerprint"], "source_start": scene["start"],
                                       "source_end": scene["end"], "content_kind": "credits"})
        self.inverted = defaultdict(set)
        for index, scene in enumerate(self.scenes):
            analysis, visual = scene["analysis"] or {}, scene["visual"] or {}
            # Whole-episode character lists are not evidence that a person appears in this shot.
            # Retrieval uses full labels below; review background excludes window dialogue and plot summaries.
            description = json.dumps({"window_characters": analysis.get("characters", []),
                                      "window_actions": visual.get("actions", []),
                                      "window_environment": visual.get("environment", "")}, ensure_ascii=False)
            fields = [(4, scene["transcript"]), (3, analysis.get("characters", [])), (3, analysis.get("events", [])),
                      (2, visual.get("actions", [])), (2, visual.get("environment", "")),
                      (1, analysis.get("summary", "")), (1, visual.get("summary", "")), (.5, analysis.get("themes", []))]
            terms = [(weight, Counter(self.tokens(json.dumps(value, ensure_ascii=False)))) for weight, value in fields]
            self.documents.append((description, terms))
            for term in {term for _, counts in terms for term in counts}:
                self.inverted[term].add(index)

    @staticmethod
    def tokens(text):
        words = [word.casefold().strip() for word in _tokenizer().cut(str(text)) if re.search(r"[\w\u4e00-\u9fff]", word)]
        return [word for word in words if len(word) > 1]

    @staticmethod
    def _request_sources(candidates, with_images=False):
        sources, offset = [], 0
        for candidate in candidates:
            source = {key: candidate[key] for key in ("library_id", "scene_id", "asset_id", "episode", "video_path", "source_start", "source_end")}
            if "_review_id" in candidate:
                source.update(review_id=candidate["_review_id"], fingerprint=candidate["fingerprint"],
                              natural_start=candidate["natural_start"], natural_end=candidate["natural_end"])
            if with_images:
                frames = candidate["frame_paths"][:3]
                source.update(frame_paths=frames, image_indices=list(range(offset, offset + len(frames))))
                offset += len(frames)
            sources.append(source)
        return sources

    def range_allowed(self, candidate):
        return not any(row["fingerprint"] == candidate["fingerprint"] and
                       min(row["source_end"], candidate["source_end"]) - max(row["source_start"], candidate["source_start"]) > .000001
                       for row in self.forbidden)

    def _load_exclusions(self, candidate, gateway):
        key = "visual-excluded:" + stable_hash((VISUAL_POLICY_VERSION, candidate["fingerprint"], gateway.signature("vision")))
        if key not in self.exclusion_keys:
            for identifier in self.source_libraries[candidate["fingerprint"]]:
                rows = self.store.cache_get(identifier, key)
                if isinstance(rows, list) and any(isinstance(row, dict) and row.get("content_kind") not in {"credits", "mixed"} for row in rows):
                    self._archive_cache(identifier, key, rows)
                for row in rows if isinstance(rows, list) else []:
                    if isinstance(row, dict) and row.get("fingerprint") == candidate["fingerprint"] and \
                            row.get("content_kind") in {"credits", "mixed"} and \
                            all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) for k in ("source_start", "source_end")) and \
                            0 <= row["source_start"] < row["source_end"]:
                        self.forbidden.append(row)
            self.exclusion_keys.add(key)
        return key

    def _archive_cache(self, library_id, key, value):
        """Keep the legacy value before changing the scope of a cached judgment."""
        archive = key + ":before-" + REVIEW_SCOPE_VERSION
        if self.store.cache_get(library_id, archive) is None:
            self.store.cache_put(library_id, archive, {"key": key, "rule": REVIEW_SCOPE_VERSION, "value": value})

    def _reject(self, candidate, judgment, gateway):
        if judgment["content_kind"] not in {"credits", "mixed"}:
            return  # Unknown/irrelevant to one manuscript is not a permanent source exclusion.
        key = self._load_exclusions(candidate, gateway)
        row = {name: candidate[name] for name in ("fingerprint", "source_start", "source_end")}
        row["content_kind"] = judgment["content_kind"]
        row.update(scope=REVIEW_SCOPE_VERSION, reason=judgment["reason"])
        if not any(all(old.get(k) == row[k] for k in ("fingerprint", "source_start", "source_end", "content_kind")) for old in self.forbidden):
            self.forbidden.append(row)
            self.store.cache_put(candidate["library_id"], key, [item for item in self.forbidden if item["fingerprint"] == candidate["fingerprint"]])
        self.rejected_ranges.add(candidate["_safety_key"])

    def retrieve(self, requirement, limit=12, *, excluded=()):
        queries = [requirement.get("claim", ""), *requirement.get("queries", []), *requirement.get("events", [])]
        specifics = " ".join(requirement.get("characters", []) + requirement.get("required_facts", []))
        rankings = []
        for query in dict.fromkeys(queries):
            tokens = set(self.tokens(str(query) + " " + specifics))
            pool = set().union(*(self.inverted.get(term, set()) for term in tokens)) if tokens else set()
            scored = []
            for index in pool:
                scene = self.scenes[index]
                if scene["id"] in excluded or scene["id"] in self.rejected_scenes or scene["end"]-scene["start"] < 2.24:
                    continue
                score = sum(weight * sum(math.log(1 + len(self.scenes)/(1+len(self.inverted.get(term, ())))) *
                            min(3, counts.get(term, 0)) for term in tokens) / math.sqrt(max(1, sum(counts.values())))
                            for weight, counts in self.documents[index][1])
                if score > 0:
                    scored.append((score, index))
            rankings.append(sorted(scored, reverse=True))
        # Round-robin across queries so one broad theme cannot occupy the whole batch.
        selected, seen = [], set()
        for rank in range(max((len(rows) for rows in rankings), default=0)):
            for rows in rankings:
                if rank < len(rows) and rows[rank][1] not in seen:
                    selected.append(rows[rank])
                    seen.add(rows[rank][1])
                    if len(selected) >= limit:
                        break
            if len(selected) >= limit:
                break
        results = []
        for score, index in selected:
            scene = self.scenes[index]
            results.append({"scene_id": scene["id"], "asset_id": scene["asset_id"], "video_path": scene["path"],
                            "library_id": scene["library_id"], "library_name": self.library_names.get(scene["library_id"], ""),
                            "source_start": scene["start"], "source_end": scene["end"], "episode": scene["episode"],
                            "description": self.documents[index][0][:1800], "score": score,
                            "frame_paths": scene["frame_paths"], "anchor": (scene["start"] + scene["end"]) / 2})
        return results

    def _range_subtitle(self, candidate, asset):
        identifier = asset["id"]
        if identifier not in self.subtitles:
            path = Path(asset["processed_path"]) if asset.get("processed_path") else None
            try:
                text = path.read_text(encoding="utf-8-sig") if path else ""
                cues = pysrt.from_string(text) if text else []
                rows = [(cue.start.ordinal/1000, cue.end.ordinal/1000, cue.text) for cue in cues]
                self.subtitles[identifier] = (stable_hash((str(path), text)), rows, "processed" if rows else "missing")
            except (OSError, ValueError):
                self.subtitles[identifier] = (stable_hash("missing"), [], "missing")
        identity, rows, state = self.subtitles[identifier]
        entries = [{"id": f"{candidate['_review_id']}:L{index:04d}", "text": text, "start": start, "end": end,
                    "overlap_start": max(start, candidate["source_start"]), "overlap_end": min(end, candidate["source_end"])}
                   for index, (start, end, text) in enumerate(rows, 1)
                   if text.strip() and start < end and end > candidate["source_start"] and start < candidate["source_end"]]
        return entries, identity, state

    @staticmethod
    def unit_payload(segment):
        requirement = segment.requirement
        result = {"text": segment.text, "kind": requirement["kind"]}
        for name in ("claim", "characters", "events", "required_facts", "avoid"):
            if requirement.get(name) and requirement[name] != segment.text:
                result[name] = requirement[name]
        return result

    def _remember(self, segment, candidate, judgment, *, cached=False):
        key = candidate["_review_key"]
        stats = self.statistics.setdefault(segment.id, {"checked": 0, "approved": 0, "unconfirmed": 0,
                                                       "cache_hits": 0, "reasons": {}, "examples": []})
        issues = judgment["_validation_issues"]
        if not issues:
            candidate.update({name: judgment[name] for name in ("relation", "facts_match", "evidence", "conflicts",
                "subtitle_quotes", "subtitle_ids", "visual_evidence", "evidence_protocol")})
            candidate.update(usable=judgment["usable"], content_kind=judgment["content_kind"],
                visual_policy=VISUAL_POLICY_VERSION, match_policy=MATCH_POLICY_VERSION, visual_unit_id=segment.id,
                visual_checked=candidate["_verify_visual"], confidence=confidence(judgment.get("confidence")),
                reason=judgment["reason"], weak_match=judgment["relation"] != "direct" or not candidate["_verify_visual"])
        reviewed = {**candidate, "pending_review": False}
        accepted = not issues and candidate["usable"] and self.range_allowed(candidate) and match_allowed(reviewed, segment.requirement)
        if key not in self.recorded:
            self.recorded.add(key)
            stats["checked"] += 1
            stats["cache_hits"] += int(cached)
            if accepted:
                stats["approved"] += 1
            else:
                if issues:
                    stats["unconfirmed"] += 1
                    category = ("画面内容未确认" if any(item["category"] == "unknown_visual" for item in issues) else
                                "字幕引用未确认" if any(item["field"] == "subtitle_ids" for item in issues) else "候选格式未确认")
                    reason = "; ".join(item["message"] for item in issues)
                else:
                    category = "画面不可用" if not candidate["usable"] else judgment["relation"] if judgment["facts_match"] else "事实未对应"
                    reason = "; ".join(judgment["conflicts"]) or judgment["reason"]
                stats["reasons"][category] = stats["reasons"].get(category, 0) + 1
                if len(stats["examples"]) < 12:
                    stats["examples"].append({"scene_id": candidate["scene_id"], "review_id": candidate["_review_id"],
                        "source_start": candidate["source_start"], "source_end": candidate["source_end"],
                        "relation": "unknown" if issues else judgment["relation"], "reason": reason, "validation_issues": issues})
        if accepted:
            candidate.update(pending_review=False, _screened=True)
            candidate.pop("_unconfirmed", None)
            candidate.pop("_rejected", None)
            self.unconfirmed.discard(key)
            self.rejected_matches.discard(key)
            selected = frame_index(judgment.get("frame_index", 1)) if candidate["_verify_visual"] else 1
            candidate["anchor"] = candidate["source_start"] + (candidate["source_end"]-candidate["source_start"]) * (selected+.5)/3
        elif issues:
            self.unconfirmed.add(key)
            candidate.update(pending_review=True, _unconfirmed=True)
            candidate.pop("proposal_version", None)
        else:
            self.rejected_matches.add(key)
            candidate["_rejected"] = True
        return accepted

    def prepare(self, segment, candidates, assets, gateway, verify_visual=True, *, scene_boundaries=None):
        """Read exact-range caches and rank locally; this method makes no model calls."""
        asset_map = {asset["id"]: asset for asset in assets}
        if scene_boundaries is not None:
            candidates = natural_candidates(candidates, scene_boundaries, assets)
        unique = {}
        for original in candidates:
            candidate = dict(original)
            asset, scene = asset_map.get(candidate.get("asset_id")), self.scene_map.get(candidate.get("scene_id"))
            if not asset or not scene or scene["asset_id"] != asset["id"] or asset["library_id"] not in self.library_ids:
                continue
            if candidate.get("fingerprint") != asset["fingerprint"] or candidate.get("fps") != asset["fps"] or \
                    any(type(candidate.get(k)) not in (int, float) or not math.isfinite(candidate[k]) for k in
                        ("source_start", "source_end", "natural_start", "natural_end")) or \
                    not 0 <= candidate["natural_start"] <= candidate["source_start"] < candidate["source_end"] <= candidate["natural_end"] <= asset["duration"]+.01:
                continue
            candidate.update(video_path=asset["path"], library_id=asset["library_id"], description=self.documents[self.scene_indices[scene["id"]]][0][:1800])
            self._load_exclusions(candidate, gateway)
            if not self.range_allowed(candidate):
                continue
            safety_key = "visual-usability:" + stable_hash((VISUAL_POLICY_VERSION, candidate["fingerprint"],
                candidate["source_start"], candidate["source_end"], gateway.signature("vision")))
            candidate.update(_safety_key=safety_key, _review_id=stable_hash(safety_key)[:24], _verify_visual=verify_visual,
                             visual_unit_id=segment.id)
            entries, subtitle_key, state = self._range_subtitle(candidate, asset)
            legacy = "visual-unit-match:" + stable_hash((MATCH_POLICY_VERSION, SUBTITLE_EVIDENCE_VERSION, self.version,
                safety_key, segment.id, segment.text, segment.requirement, subtitle_key, entries, verify_visual,
                gateway.signature("text"), gateway.signature("vision")))
            key = "visual-unit-match:" + stable_hash((MATCH_POLICY_VERSION, SUBTITLE_EVIDENCE_VERSION, "range-context-v2",
                safety_key, segment.id, segment.text, segment.requirement, candidate["description"], entries, verify_visual,
                gateway.signature("text"), gateway.signature("vision")))
            if candidate.get("_review_key") != key:
                candidate.pop("proposal_version", None)
                candidate.pop("_screened", None)
            candidate.update(_review_key=key, subtitle_entries=entries, subtitle_state=state)
            candidate.pop("_rejected", None)
            if candidate.pop("_unconfirmed", False):
                candidate.pop("_screened", None)
            if key in self.rejected_matches:
                continue
            safety = self.store.cache_get(candidate["library_id"], safety_key)
            try:
                if safety is not None:
                    _review_result({"clips": [safety]}, {candidate["_review_id"]})
                    if safety["content_kind"] == "unknown":
                        self._archive_cache(candidate["library_id"], safety_key, safety)
                        safety = None
                    elif not safety["usable"]:
                        self._reject(candidate, safety, gateway)
                        continue
            except (ModelFormatError, TypeError, KeyError):
                if safety is not None:
                    self._archive_cache(candidate["library_id"], safety_key, safety)
                safety = None
            candidate["_safety"] = safety
            judgment = self.store.cache_get(candidate["library_id"], key)
            migrated = judgment is None
            if judgment is None:
                judgment = self.store.cache_get(candidate["library_id"], legacy)
            if isinstance(judgment, dict) and judgment.get("usable") is False and judgment.get("content_kind") == "unknown" and \
                    judgment.get("review_scope") != REVIEW_SCOPE_VERSION:
                self._archive_cache(candidate["library_id"], legacy if migrated else key, judgment)
                judgment = None  # Recheck this legacy source/relationship mixture without approving it.
            valid = False
            if isinstance(judgment, dict) and judgment.get("evidence_protocol") == SUBTITLE_EVIDENCE_VERSION:
                try:
                    checked = _match_result({"clips": [judgment]}, {candidate["_review_id"]: candidate}, verify_visual)["clips"][0]
                except (ModelFormatError, TypeError, KeyError):
                    checked = None
                if checked is not None and not checked["_validation_issues"] and (verify_visual or safety is not None):
                    valid = self._remember(segment, candidate, checked, cached=True)
                    if migrated:
                        self.store.cache_put(candidate["library_id"], key, {k: v for k, v in checked.items() if not k.startswith("_")})
                    if not valid:
                        continue
            if not valid:
                for name in ("match_policy", "visual_policy", "facts_match", "relation", "confidence", "usable", "content_kind"):
                    candidate.pop(name, None)
                candidate["pending_review"] = True
                query = set(self.tokens(segment.text + " " + json.dumps(segment.requirement, ensure_ascii=False)))
                observation = self.store.cache_get(candidate["library_id"], "visual-observation:" + stable_hash(safety_key)) or {}
                facts = observation.get("facts", []) if isinstance(observation, dict) else []
                candidate["observed_facts"] = facts if isinstance(facts, list) and all(isinstance(x, str) for x in facts) else []
                scores = [(len(query & set(self.tokens(row["text"]))), row) for row in entries]
                best = max(scores, key=lambda item: item[0], default=(0, None))
                score = 4*sum(value for value, _ in scores) + 2*len(query & set(self.tokens(" ".join(candidate["observed_facts"]))))
                lexical = candidate.get("score", 0)
                lexical = lexical if type(lexical) in (int, float) and math.isfinite(lexical) else 0
                score += len(query & set(self.tokens(candidate["description"]))) + math.log1p(max(0, lexical))
                candidate["proposal_score"] = score/(1+score)
                if best[0]:
                    row = best[1]
                    candidate["anchor"] = (row["overlap_start"]+row["overlap_end"])/2
            identity = source_shot_key(candidate)
            previous = unique.get(identity)
            rank = lambda item: (not item.get("pending_review", True), item.get("confidence", item.get("proposal_score", 0)), item["source_end"]-item["source_start"])
            if previous is None or rank(candidate) > rank(previous):
                unique[identity] = candidate
        return sorted(unique.values(), key=lambda item: (not item.get("pending_review", True), item.get("confidence", item.get("proposal_score", 0))), reverse=True)

    def restore_candidates(self, directory, segments, assets, gateway, verify_visual, *, stop=None):
        """Recover paid ranges, including replies that the old pending-state bug did not approve."""
        result = {identifier: [] for identifier in segments}
        directory = Path(directory)
        asset_map, requests, seen = {a["id"]: a for a in assets}, {}, set()
        for path in (Path(directory)/"candidates").glob("*.json"):
            check_cancel(stop)
            try:
                saved = read_json(path)
                if not isinstance(saved, dict):
                    continue
                candidates = list(saved["value"]) if isinstance(saved.get("value"), list) else []
                reviews = {(r["unit_id"], r["review_id"], r["source_start"], r["source_end"])
                           for r in saved.get("reviews", []) if isinstance(r, dict) and
                           all(k in r for k in ("unit_id", "review_id", "source_start", "source_end"))}
                records = list(saved.get("requests", []))
                for group in saved.get("reviews", []):
                    if not isinstance(group, dict) or saved.get("visual_unit_id") not in segments:
                        continue
                    records.extend(group.get("requests", []))
                    reviews.update((saved["visual_unit_id"], clip["scene_id"], clip["source_start"], clip["source_end"])
                        for clip in group.get("clips", []) if isinstance(clip, dict) and
                        all(k in clip for k in ("scene_id", "source_start", "source_end")))
                for record in records if reviews else ():
                    if not isinstance(record, dict):
                        continue
                    relative = Path(record.get("request_file", ""))
                    if relative.parent != Path("model-requests") or not re.fullmatch(r"[a-f0-9]{32}\.json", relative.name):
                        continue
                    if relative.name not in requests:
                        try:
                            requests[relative.name] = read_json(directory/relative).get("metadata", {})
                        except (OSError, ValueError, TypeError, AttributeError):
                            requests[relative.name] = {}
                    metadata = requests[relative.name]
                    if not isinstance(metadata, dict) or not isinstance(metadata.get("canonical_pairs", {}), dict) or \
                            not isinstance(metadata.get("candidates", []), list):
                        continue
                    pairs = list(metadata.get("canonical_pairs", {}).values())
                    if not pairs and metadata.get("visual_unit_id") == saved.get("visual_unit_id") in segments:
                        pairs = [{"unit_id": metadata["visual_unit_id"], "review_id": c.get("review_id")}
                                 for c in metadata.get("candidates", []) if isinstance(c, dict)]
                    for pair in pairs:
                        if not isinstance(pair, dict):
                            continue
                        for source in metadata.get("candidates", []):
                            if not isinstance(source, dict):
                                continue
                            owner = pair.get("unit_id")
                            asset = asset_map.get(source.get("asset_id"))
                            if owner not in segments or not asset or pair.get("review_id") != source.get("review_id") or \
                                    asset.get("fingerprint") != source.get("fingerprint") or \
                                    (owner, source["review_id"], source["source_start"], source["source_end"]) not in reviews:
                                continue
                            candidates.append({**source, "visual_unit_id": owner, "fps": asset["fps"]})
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    unit = candidate.get("visual_unit_id")
                    identity = (unit, candidate.get("asset_id"), candidate.get("scene_id"), candidate.get("source_start"), candidate.get("source_end"))
                    if unit in segments and identity not in seen:
                        seen.add(identity)
                        rows = self.prepare(segments[unit], [candidate], assets, gateway, verify_visual)
                        result[unit].extend(rows)  # Pending ranges still need the normal approval gate.
            except (OSError, ValueError, TypeError, KeyError):
                continue
        return result

    def shortlist(self, groups, gateway, directory, stop=None, *, search_round=0):
        """Batch text-only choices; all choices remain provisional until explicit review."""
        pairs = []
        for segment, candidates, needed in groups:
            capacity = 0
            for candidate in candidates:
                if candidate.pop("_force_review", False):
                    candidate.update(proposal_version=PROPOSAL_VERSION, _screened=True)
                    continue
                if candidate.get("pending_review") and not candidate.get("_screened") and candidate["_review_key"] not in self.unconfirmed:
                    pairs.append((segment, candidate))
                    capacity += min(needed, candidate["source_end"]-candidate["source_start"])
                    if capacity >= max(2, needed*2) or sum(s.id == segment.id for s, _ in pairs) >= 8:
                        break
        def submit(batch):
            if not batch:
                return
            check_cancel(stop)
            units = {s.id: s for s, _ in batch}
            clips = {c["_safety_key"]: c for _, c in batch}
            ua = {name: f"U{i}" for i, name in enumerate(units, 1)}
            ca = {key: f"C{i}" for i, key in enumerate(clips, 1)}
            allowed = {ua[name]: [ca[c["_safety_key"]] for s, c in batch if s.id == name] for name in units}
            identity = stable_hash((PROPOSAL_VERSION, gateway.signature("text_fast"),
                [(s.id, self.unit_payload(s), c["_review_key"], c["subtitle_entries"], c.get("observed_facts")) for s, c in batch]))
            path = Path(directory)/"planning"/f"preselect-{identity[:24]}.json"
            try:
                saved = read_json(path, default={})
            except (OSError, ValueError):
                saved = {}
            chosen = saved.get("chosen") if isinstance(saved, dict) and saved.get("key") == identity and not saved.get("incomplete") else None
            if not isinstance(chosen, list) or any(not isinstance(row, list) or len(row) != 2 or not all(isinstance(x, str) for x in row) for row in chosen):
                chosen = None
            if chosen is None:
                reply = None
                if len(clips) <= 3:
                    chosen = [[s.id, c["_review_key"]] for s, c in batch]
                else:
                    payload = {"units": [{"id": ua[name], **self.unit_payload(s), "candidates": allowed[ua[name]]} for name, s in units.items()],
                        "clips": [{"id": ca[key], "background": c["description"], "observed": c.get("observed_facts", []), "observation_partial": True,
                                   "subtitles": [row["text"] for row in c["subtitle_entries"]]} for key, c in clips.items()]}
                    def validate(value):
                        rows = value.get("selections")
                        if not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get("unit_id"), str) for r in rows) or \
                                len(rows) != len(allowed) or {r["unit_id"] for r in rows} != set(allowed):
                            raise ModelFormatError("selections须逐一返回所有unit_id且各一次："+str(allowed))
                        invalid = []
                        for row in rows:
                            ids = row.get("clip_ids")
                            if not isinstance(ids, list) or any(not isinstance(x, str) or x not in allowed[row["unit_id"]] for x in ids) or len(ids) != len(set(ids)):
                                invalid.append(row["unit_id"])
                        return {"selections": [{**row, "clip_ids": []} if row["unit_id"] in invalid else row for row in rows],
                                "invalid_units": invalid, "raw_response": value}
                    prompt = ("为文案预选值得进一步核对的少数镜头，只返回编号；预选不等于事实确认。"
                        "背景与字幕不证明实际画面必然出现某人或事件；observed只是部分线索，未列不等于不存在。具体人物/事件要对应，抽象论述可用有具体同向联系的情境。"
                        "遵守required_facts和avoid，否认关系不能支持建立关系。按相关性排序，证据不足可空，不按时长硬凑。"
                        "返回{selections:[{unit_id,clip_ids:[]}]}，每个单元一次，不写解释。\n"+json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                    limit = int(gateway.settings["text_fast"].get("max_prompt_chars", 16000))
                    if len(prompt) > limit or len(prompt)*2+1024 >= int(gateway.settings["text_fast"].get("context_tokens", 64000)):
                        if len(batch) > 1:
                            submit(batch[:len(batch)//2]); submit(batch[len(batch)//2:])
                        else:
                            segment, candidate = batch[0]
                            candidate["_screened"] = True
                            self._remember(segment, candidate, {"_validation_issues": [{"field": "input", "category": "input_budget",
                                "message": "文字预选单镜头输入超过模型预算，保留为未确认，调整上下文设置后可继续。"}]})
                        return
                    gateway.context.update(stage="候选文字预选", segment="、".join(units))
                    value = gateway.json(prompt, model_kind="text_fast", max_output=1200, required={"selections": list}, result_validator=validate,
                        request_metadata={"usage_stage": "text_preselect", "search_round": search_round,
                                          "pairs": [{"unit_id": s.id, "review_key": c["_review_key"]} for s, c in batch]})
                    reply = value["raw_response"]
                    selected = {(r["unit_id"], x) for r in value["selections"] for x in r["clip_ids"]}
                    for segment, candidate in batch:
                        if ua[segment.id] in value.get("invalid_units", []):
                            self._remember(segment, candidate, {"_validation_issues": [{"field": "preselection", "category": "invalid_selection",
                                "message": "文字预选返回的候选编号类型、归属或唯一性无效，本次未确认。"}]})
                    chosen = [[s.id, c["_review_key"]] for s, c in batch if (ua[s.id], ca[c["_safety_key"]]) in selected]
                write_json(path, {"key": identity, "chosen": chosen, "response": reply,
                                  "incomplete": any(c["_review_key"] in self.unconfirmed for _, c in batch)})
            for segment, candidate in batch:
                candidate["_screened"] = True
                if [segment.id, candidate["_review_key"]] in chosen:
                    candidate["proposal_version"] = PROPOSAL_VERSION
        batch, unique, unit_ids = [], set(), set()
        for pair in pairs:
            source = pair[1]["_safety_key"]
            if batch and (len(unique | {source}) > 20 or len(unit_ids | {pair[0].id}) > 4):
                submit(batch)
                batch, unique, unit_ids = [], set(), set()
            batch.append(pair); unique.add(source); unit_ids.add(pair[0].id)
        submit(batch)

    def review_pairs(self, pairs, gateway, directory, stop=None, *, search_round=0, _safety_only=False, _repair=None):
        """Only proposed pairs reach this final review; images/catalogues are shared by range."""
        from app.services.library.gateway import ModelAPIError, ModelInputBudgetError
        accepted = []
        for offset in range(0, len(pairs), 6):
            batch = pairs[offset:offset+6]
            check_cancel(stop)
            units = {s.id: s for s, _ in batch}
            clips = {c["_safety_key"]: c for _, c in batch}
            ua = {name: f"U{i}" for i, name in enumerate(units, 1)}
            ca = {key: f"C{i}" for i, key in enumerate(clips, 1)}
            aliases = {f"P{i}": pair for i, pair in enumerate(batch, 1)}
            references = {f"{ca[key]}:L{i}": row["id"] for key, c in clips.items() for i, row in enumerate(c["subtitle_entries"], 1)}
            with_images = _safety_only or batch[0][1]["_verify_visual"]
            # With confirmation disabled, the existing independent visual safety check still runs on cache misses.
            safety_only = not with_images and any(c.get("_safety") is None for c in clips.values())
            if safety_only:
                safety_batch = [(s, c) for s, c in batch if c.get("_safety") is None]
                self.review_pairs(safety_batch, gateway, directory, stop, search_round=search_round, _safety_only=True, _repair=_repair)
                batch = [(s, c) for s, c in batch if c.get("_safety") and c["_safety"]["usable"] and c["_review_key"] not in self.unconfirmed]
                if batch:
                    accepted.extend(self.review_pairs(batch, gateway, directory, stop, search_round=search_round, _repair=_repair))
                continue
            images = []
            if with_images:
                for c in clips.values():
                    folder = self.store.directory(c["library_id"])/"frames"/c["asset_id"]/c["_review_id"]
                    c["frame_paths"] = extract_frames(c["video_path"], c["source_start"], c["source_end"], 3, folder)
                    images.extend(c["frame_paths"])
            payload = {"units": [{"id": ua[name], **self.unit_payload(s)} for name, s in units.items()],
                "clips": [{"id": ca[key], "background": c["description"],
                    "subtitles": [{"id": f"{ca[key]}:L{i}", "text": row["text"]} for i, row in enumerate(c["subtitle_entries"], 1)]}
                    for key, c in clips.items()],
                "pairs": [{"scene_id": alias, "unit": ua[s.id], "clip": ca[c["_safety_key"]]} for alias, (s, c) in aliases.items()]}
            def validate(value):
                rows = _review_rows(value, aliases)
                normalized = []
                for original in rows:
                    segment, candidate = aliases[original["scene_id"]]
                    row = {**original, "scene_id": candidate["_review_id"]}
                    if isinstance(row.get("subtitle_ids"), list):
                        row["subtitle_ids"] = [references.get(x, "unknown:"+x) if isinstance(x, str) else x for x in row["subtitle_ids"]]
                    checked = _match_result({"clips": [row]}, {candidate["_review_id"]: candidate}, with_images, relevance=not _safety_only)["clips"][0]
                    checked.update(_pair=original["scene_id"], _raw_response=original)
                    normalized.append(checked)
                return {"clips": normalized}
            prompt = ("逐一检查pairs，仅返回每个P编号一次。clips按顺序各附3图，frame_index是该素材内0至2。"
                "usable仅判断画面：剧情story为true，职员表/演职员名单credits、混合mixed、不明unknown为false；普通对白/歌词不因此禁用。"
                "relation为direct/supporting/unrelated/unknown，facts_match只用true或false；具体事实仅部分满足时用false，禁止partial。"
                "confidence必须为0至1的有限数字。人物/事件不符只影响relation/facts_match/conflicts，不得因此将story标为不可用。"
                "evidence类须直接证据；abstract/state/transition可用有具体同向联系的辅助画面，不要求单镜头证明整套理论。"
                "遵守人物/时期/事件/关系及avoid，提及人物不等于出现，否认不能支持建立关系；背景仅作线索，不把窗口外剧情当本镜头。"
                "visual_evidence仅写可见动作/状态/环境，不写对白；subtitle_ids只选本素材的C编号字幕，不跨素材、不抄引文。"
                "字幕可能仅部分重叠，不能声称整句在范围内说完。给具体短依据，拒绝理由简短，每种依据最多3条。"
                "返回{clips:[{scene_id,usable,content_kind,reason,confidence,frame_index,relation,facts_match,visual_evidence:[],conflicts:[],subtitle_ids:[]}]}。\n")
            if not with_images:
                prompt = prompt.replace("clips按顺序各附3图，frame_index是该素材内0至2。", "本次无新图，已独立检查画面可用性；据范围字幕和已有描述核对内容，不明细节标unknown。")
            if _safety_only:
                prompt = ("仅检查实际画面可用性，每个P编号一次。每个C按顺序附3张图，frame_index为该C内0至2。"
                    "story剧情为usable=true；职员表/演职员名单credits、混合mixed、不明unknown为false。普通对白、歌词不因此禁用。"
                    "返回{clips:[{scene_id,usable,content_kind,reason,confidence,frame_index}]}，不判断文案关系。\n")
                payload = {"clips": [{"id": ca[key]} for key in clips],
                           "pairs": [{"scene_id": a, "clip": ca[c["_safety_key"]]} for a, (_, c) in aliases.items()]}
            if _repair:
                reverse = {value: key for key, value in references.items()}
                feedback = []
                for alias, (_, candidate) in aliases.items():
                    old = _repair[candidate["_review_key"]]
                    previous = {k: v for k, v in old["previous"].items() if k in (
                        "usable", "content_kind", "reason", "confidence", "frame_index", "relation", "facts_match",
                        "visual_evidence", "conflicts", "subtitle_ids")}
                    if isinstance(previous.get("subtitle_ids"), list):
                        previous["subtitle_ids"] = [reverse.get(x, "invalid:"+str(x)) for x in previous["subtitle_ids"]]
                    feedback.append({"scene_id": alias, "previous": previous, "issues": old["issues"]})
                payload["repair"] = feedback
                prompt += "这是一次失败条目的局部纠错。原回复只作待修数据，按当前P/C目录修正字段；证据不足应明确不匹配，不为了通过而改成true。\n"
            text = prompt+json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            identity = stable_hash((PROPOSAL_VERSION, [(s.id, c["_review_key"]) for s, c in batch], with_images))
            if _repair:
                identity = stable_hash((identity, "row-repair-v1"))
            path = Path(directory)/"candidates"/f"review-{identity[:24]}.json"
            limit = int(gateway.settings["vision" if with_images else "text"].get("max_prompt_chars", 16000))
            too_large = len(text) > limit or len(text)*2+len(images)*1600+1024 >= int(gateway.settings["vision" if with_images else "text"].get("context_tokens", 64000))
            if too_large:
                if len(batch) > 1:
                    accepted.extend(self.review_pairs(batch[:len(batch)//2], gateway, directory, stop, search_round=search_round, _safety_only=_safety_only, _repair=_repair))
                    accepted.extend(self.review_pairs(batch[len(batch)//2:], gateway, directory, stop, search_round=search_round, _safety_only=_safety_only, _repair=_repair))
                else:
                    segment, candidate = batch[0]
                    self._remember(segment, candidate, {"_validation_issues": [{"field": "input", "category": "input_budget", "message": "单镜头输入超过本模型预算，请调整模型上下文设置。"}]})
                continue
            gateway.context.update(stage="检查句意与实际镜头", segment="、".join(units))
            requests, previous_after = [], gateway.after_request
            def after(details):
                requests.append({k: details[k] for k in ("request_file", "usage", "http_status") if k in details})
                if previous_after:
                    previous_after(details)
            gateway.after_request = after
            try:
                result = gateway.json(text, images=images or None, model_kind="vision" if with_images else "text", max_output=2400,
                    required={"clips": list}, result_validator=validate, attempts=1 if _repair else 3,
                    request_metadata={"usage_stage": "visual_safety" if _safety_only else "visual_review", "search_round": search_round,
                        "candidate_file": path.relative_to(directory).as_posix(), "canonical_pairs": {
                            alias: {"unit_id": s.id, "review_id": c["_review_id"]} for alias, (s, c) in aliases.items()},
                        "subtitle_aliases": references, "local_repair": bool(_repair),
                        "candidates": self._request_sources(list(clips.values()), with_images=with_images)})
            finally:
                gateway.after_request = previous_after
            approved, diagnostics, deferred = [], [], []
            for row in result["clips"]:
                segment, candidate = aliases[row["_pair"]]
                candidate.pop("proposal_version", None)
                row["review_scope"] = REVIEW_SCOPE_VERSION
                if with_images and row["_usability_valid"]:
                    safety = {k: row[k] for k in ("scene_id", "usable", "content_kind", "reason")}
                    safety["scope"] = REVIEW_SCOPE_VERSION
                    if safety["content_kind"] != "unknown" or _safety_only:
                        self.store.cache_put(candidate["library_id"], candidate["_safety_key"], safety)
                    candidate["_safety"] = safety
                    if not safety["usable"]:
                        self._reject(candidate, safety, gateway)
                    elif isinstance(row.get("visual_evidence"), list) and all(isinstance(x, str) for x in row["visual_evidence"]):
                        self.store.cache_put(candidate["library_id"], "visual-observation:"+stable_hash(candidate["_safety_key"]),
                                             {"partial": True, "facts": row["visual_evidence"]})
                if row["_validation_issues"] and not _repair and not any(
                        issue["category"] == "unknown_visual" for issue in row["_validation_issues"]):
                    deferred.append((segment, candidate, row))
                else:
                    if not _safety_only and not row["_validation_issues"]:
                        self.store.cache_put(candidate["library_id"], candidate["_review_key"], {k: v for k, v in row.items() if not k.startswith("_")})
                    if (not _safety_only or row["_validation_issues"]) and self._remember(segment, candidate, row) and not _safety_only:
                        accepted.append(candidate); approved.append(candidate)
                diagnostics.append({"unit_id": segment.id, "review_id": candidate["_review_id"],
                    "source_start": candidate["source_start"], "source_end": candidate["source_end"],
                    "response": row["_raw_response"], "response_stage": "normalized_model_reply",
                    "validation_issues": row["_validation_issues"]})
            try:
                previous = read_json(path, default={})
            except (OSError, ValueError):
                previous = {}
            history = previous.get("history", []) if isinstance(previous, dict) else []
            if not isinstance(history, list):
                history = []
            if isinstance(previous, dict) and previous.get("reviews"):
                history.append({k: previous[k] for k in ("reviews", "requests", "subtitle_aliases") if k in previous})
            audit = {"key": identity, "version": MATCH_POLICY_VERSION, "value": approved, "reviews": diagnostics,
                     "requests": requests, "subtitle_aliases": references, "history": history, "local_repair": bool(_repair)}
            write_json(path, audit)
            if deferred:
                repair = {c["_review_key"]: {"previous": row, "issues": row["_validation_issues"]} for _, c, row in deferred}
                try:
                    accepted.extend(self.review_pairs([(s, c) for s, c, _ in deferred], gateway, directory, stop,
                        search_round=search_round, _safety_only=_safety_only, _repair=repair))
                except (ModelFormatError, ModelAPIError, ModelInputBudgetError) as exc:
                    for segment, candidate, row in deferred:
                        self._remember(segment, candidate, row)
                    audit["repair_error"] = gateway._redact_request(str(exc))[:1000]
                    write_json(path, audit)
        return accepted

    def select(self, segment, overview, gateway, directory, used=(), verify_visual=True, stop=None, *, candidates, scene_boundaries, assets):
        prepared = self.prepare(segment, candidates, assets, gateway, verify_visual, scene_boundaries=scene_boundaries)
        prepared = [c for c in prepared if unused_source(c, used)]
        return [c for c in prepared if not c.get("pending_review")] + self.review_pairs(
            [(segment, c) for c in prepared if c.get("pending_review") and unused_source(c, used)], gateway, directory, stop)
