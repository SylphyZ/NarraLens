"""Local lexical/semantic-label retrieval followed by bounded model review."""

import json
import math
import re
from collections import Counter, defaultdict
from contextlib import nullcontext
from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path

from app.services.library.frame_extractor import extract_frames
from app.services.library.indexer import plot_cues
from app.services.library.runner import check_cancel
from app.services.library.store import library_ids, stable_hash, write_json, read_json
from app.services.subtitle_segmentation import _tokenizer
from app.services.library.validation import confidence, frame_index, ModelFormatError
from app.services.timeline_planner import (
    VISUAL_POLICY_VERSION, MATCH_POLICY_VERSION, PROPOSAL_VERSION, COVERAGE_VERSION, match_allowed, natural_candidates, source_shot_key, unused_source,
)


SUBTITLE_EVIDENCE_VERSION = "subtitle-evidence-ids-v1"
REVIEW_SCOPE_VERSION = "source-usability-v1"
RETRIEVAL_REPAIR_VERSION = "preselection-compatibility-v1"
CANDIDATE_QUEUE_VERSION = "candidate-progress-v1"
REVIEW_PROTOCOL = "single-unit-review-v2"


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
                if not isinstance(items, list) or (name != "visual_evidence" and len(items) > 3) or \
                        any(not isinstance(x, str) or not x.strip() for x in items):
                    issue(name, "invalid_list", f"{name}须为非空字符串列表，可为空" + ("。" if name == "visual_evidence" else "，最多3项。"))
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
            candidate = expected[row["scene_id"]]
            requirement = candidate.get("_requirement", {})
            if requirement.get("coverage_version") == COVERAGE_VERSION:
                allowed = {fact["id"] for fact in requirement.get("facts", [])}
                supports, raw_supports = [], row.get("supports")
                if type(row.get("constraints_match")) is not bool or type(row.get("motif_match")) is not bool:
                    issue("constraints_match/motif_match", "invalid_type", "约束满足和配图符合须为布尔值。")
                if not isinstance(raw_supports, list) or len(raw_supports) > len(allowed):
                    issue("supports", "invalid_list", "supports须逐条引用当前F编号，最多每事实一次。")
                    raw_supports = []
                seen_facts = set()
                for support in raw_supports:
                    if not isinstance(support, dict) or not isinstance(support.get("fact_id"),str) or support["fact_id"] not in allowed or support["fact_id"] in seen_facts:
                        issue("supports", "invalid_fact", "事实编号须属于当前单元且不得重复。")
                        continue
                    seen_facts.add(support["fact_id"])
                    indices, citations = support.get("visual_indices"), support.get("subtitle_ids")
                    if not isinstance(indices,list) or len(indices)>3 or any(type(i) is not int or not 0<=i<=2 for i in indices) or len(indices)!=len(set(indices)) or (indices and not with_images):
                        issue("supports", "invalid_frame", "事实可见依据须为本候选0至2图片编号；无图时不得引用图片。")
                        continue
                    if not isinstance(citations,list) or len(citations)>3 or any(not isinstance(i,str) or i not in entries for i in citations) or len(citations)!=len(set(citations)):
                        issue("supports", "invalid_citation", "事实引用须使用本候选实际范围内的字幕编号。")
                        continue
                    if (not indices and not citations) or not isinstance(support.get("reason"),str) or not support["reason"].strip():
                        issue("supports", "missing_evidence", "事实须有当前可见帧或合法范围字幕的依据及理由。")
                        continue
                    points = [candidate["source_start"]+(candidate["source_end"]-candidate["source_start"])*(i+.5)/3 for i in indices]
                    cited = [e for e in candidate["subtitle_entries"] if e["id"] in citations]
                    left = min(points+[e["overlap_start"] for e in cited])
                    right = max(points+[e["overlap_end"] for e in cited])
                    supports.append({"fact_id":support["fact_id"],"start":left,"end":right,"visual_indices":indices,"subtitle_ids":citations,"reason":support["reason"]})
                    evidence.append("事实"+support["fact_id"]+"："+support["reason"])
                motif = None
                if row.get("motif_match") is True:
                    anchors = [candidate["source_start"]+(candidate["source_end"]-candidate["source_start"])*(row.get("frame_index",1)+.5)/3] if with_images and row.get("visual_evidence") else []
                    cited = [e for e in candidate["subtitle_entries"] if valid_ids and e["id"] in ids]
                    if anchors or cited:
                        motif = {"start":min(anchors+[e["overlap_start"] for e in cited]),"end":max(anchors+[e["overlap_end"] for e in cited])}
                    else:
                        issue("motif_match","missing_evidence","配图符合须有本候选的实际画面或字幕依据。")
                row["coverage"] = {"version":COVERAGE_VERSION,"requirement_hash":stable_hash(requirement),"mode":"model-fact-evidence",
                                   "supports":supports,"constraints_match":row.get("constraints_match"),"motif_match":row.get("motif_match"),"motif_range":motif}
            if row.get("relation") in ("direct", "supporting") and not evidence:
                issue("visual_evidence/subtitle_ids", "missing_evidence", "direct/supporting须有可核对的画面依据或本候选字幕编号。")
            # Free-form quotes/evidence from the model never become trusted evidence.
            row.update(evidence_protocol=SUBTITLE_EVIDENCE_VERSION, subtitle_quotes=quotes, evidence=evidence)
        row.update(_validation_issues=issues, _usability_valid=usability_valid, _raw_response=original)
        result.append(row)
    return {"clips": result}


def _match_pairs(value, aliases, references, with_images, *, relevance=True, require_owner=False):
    """Resolve only a unique clip-to-pair alias; conflicting replies remain unconfirmed."""
    rows = value.get("clips") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise ModelFormatError("clips须为对象列表，scene_id仅使用本次P编号。")
    groups, discarded = defaultdict(list), []
    clips = list(dict.fromkeys(c["_safety_key"] for _, c in aliases.values()))
    clip_pairs = {f"C{i}": [a for a, (_, c) in aliases.items() if c["_safety_key"] == key]
                  for i, key in enumerate(clips, 1)}
    for row in rows:
        alias = row.get("scene_id") if isinstance(row, dict) else None
        if require_owner and len({s.id for s, _ in aliases.values()}) == 1 and \
                isinstance(alias, str) and alias not in aliases and row.get("unit_id") == "U1":
            matches = clip_pairs.get(alias, [])
            if len(matches) == 1:
                alias = matches[0]
        if not isinstance(alias, str) or alias not in aliases:
            discarded.append(row)
        else:
            groups[alias].append(row)
    result = []
    for alias, (segment, candidate) in aliases.items():
        items = groups[alias]
        if len(items) != 1:
            result.append({"scene_id": candidate["_review_id"], "_pair": alias, "_usability_valid": False,
                "_raw_response": {"replies": items, "unassigned_replies": discarded},
                "_validation_issues": [{"field": "scene_id", "category": "pair_identity",
                    "message": f"当前P需要恰好一条回复，实际{len(items)}条；逐帧或冲突回复不能合并猜测。"}]})
            continue
        original = items[0]
        if require_owner and original.get("unit_id") != "U1":
            result.append({"scene_id":candidate["_review_id"],"_pair":alias,"_usability_valid":False,"_raw_response":original,
                           "_validation_issues":[{"field":"unit_id","category":"pair_identity","message":"本次仅审核U1，回复必须绑定当前单元。"}]})
            continue
        row = {**original, "scene_id": candidate["_review_id"]}
        if require_owner:
            row["unit_id"] = segment.id
        row.pop("normalization", None)  # Only the local negative-only branch may issue this proof.
        if isinstance(row.get("subtitle_ids"), list):
            row["subtitle_ids"] = [references.get(x, "unknown:"+x) if isinstance(x, str) else x for x in row["subtitle_ids"]]
        if isinstance(row.get("supports"), list):
            row["supports"] = [{**s, "subtitle_ids":[references.get(x,"unknown:"+x) if isinstance(x,str) else x for x in s["subtitle_ids"]]}
                               if isinstance(s,dict) and isinstance(s.get("subtitle_ids"),list) else s for s in row["supports"]]
        expected = {candidate["_review_id"]: candidate}
        checked = _match_result({"clips": [row]}, expected, with_images, relevance=relevance)["clips"][0]
        ids = row.get("subtitle_ids")
        # Only a known negative may shed redundant, entirely valid citations; positive evidence stays strict.
        if relevance and candidate.get("_requirement",{}).get("coverage_version") != COVERAGE_VERSION and \
                row.get("facts_match") is False and row.get("usable") is True and row.get("content_kind") == "story" and \
                checked["_validation_issues"] and all(i["field"] == "subtitle_ids" and i["category"] == "invalid_list"
                    for i in checked["_validation_issues"]) and isinstance(ids, list) and len(ids) > 3 and \
                all(isinstance(x, str) and x in {e["id"] for e in candidate["subtitle_entries"]} for x in ids) and len(ids) == len(set(ids)):
            trimmed = _match_result({"clips": [{**row, "subtitle_ids": ids[:3]}]}, expected, with_images)["clips"][0]
            if not trimmed["_validation_issues"]:
                checked = trimmed
                checked["normalization"] = {"rule": "negative-citation-limit-v1", "original_count": len(ids)}
        checked.update(_pair=alias, _raw_response=original)
        if original["scene_id"] != alias:
            checked["normalization"] = {"rule": "unique-clip-pair-v1", "from": original["scene_id"], "to": alias}
        result.append(checked)
    return {"clips": result, "discarded_replies": discarded}


def _repair_feedback(aliases, references, repair):
    """Carry actual bad rows into the reduced batch using its frozen alias catalog."""
    current_pairs = {c["_review_key"]: a for a, (_, c) in aliases.items()}
    current_clips = {key: f"C{i}" for i, key in enumerate(dict.fromkeys(c["_safety_key"] for _, c in aliases.values()), 1)}
    current_refs = {value: key for key, value in references.items()}
    feedback = []
    for alias, (segment, candidate) in aliases.items():
        old = repair[candidate["_review_key"]]
        previous_aliases = old["aliases"]
        pair_ids = {a: current_pairs.get(c["_review_key"], "outside:" + a) for a, (_, c) in previous_aliases.items()}
        clip_ids = {f"C{i}": current_clips.get(key, f"outside:C{i}") for i, key in
                    enumerate(dict.fromkeys(c["_safety_key"] for _, c in previous_aliases.values()), 1)}
        subtitle_ids = {a: current_refs.get(real, "outside:" + a) for a, real in old["references"].items()}
        raw = old["previous"]
        rows = raw.get("replies", []) + raw.get("unassigned_replies", []) if isinstance(raw, dict) and "replies" in raw else [raw]

        def remap(value, catalog):
            return catalog.get(value, "invalid:" + value) if isinstance(value, str) else value

        previous = []
        for original in rows:
            if not isinstance(original, dict):
                previous.append(original)
                continue
            row = dict(original)
            if "scene_id" in row:
                row["scene_id"] = remap(row["scene_id"], {**pair_ids, **clip_ids})
            if "unit_id" in row:
                row["unit_id"] = remap(row["unit_id"], {"U1": "U1", segment.id: "U1"})
            if isinstance(row.get("subtitle_ids"), list):
                row["subtitle_ids"] = [remap(x, subtitle_ids) for x in row["subtitle_ids"]]
            if isinstance(row.get("supports"), list):
                row["supports"] = [{**s, "subtitle_ids": [remap(x, subtitle_ids) for x in s["subtitle_ids"]]}
                    if isinstance(s, dict) and isinstance(s.get("subtitle_ids"), list) else s for s in row["supports"]]
            previous.append(row)
        feedback.append({"expected_scene_id": alias, "expected_unit_id": "U1",
            "returned_scene_ids": [r.get("scene_id") for r in rows if isinstance(r, dict)],
            "previous": previous, "issues": old["issues"]})
    return feedback


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
        self.evaluated = defaultdict(set)
        self.preselection = {}
        self.requirement_history = {}
        self.candidate_progress = {}
        self.queue_signature = None
        self.legacy_quarantine = {}
        self.previous_requirements = {}
        self.scene_aliases, self.previous_versions = {}, {}
        migrations = sorted(store.subtitle_migrations(self.library_ids), key=lambda m: m.get("created", 0), reverse=True)
        previous_scenes, affected_indices = list(self.scenes), set()
        for migration in migrations:
            if migration.get("state") != "committed":
                continue
            old = {row["id"]: row for row in migration.get("old_scenes", [])}
            for old_id, new_id in migration.get("scene_map", {}).items():
                current = self.scene_map.get(new_id)
                prior = old.get(old_id)
                if not current or not prior or current["asset_id"] != migration.get("asset_id") or current["fingerprint"] != migration.get("fingerprint") or any(current[k] != prior[k] for k in ("start", "end")):
                    continue
                index = self.scene_indices[new_id]
                affected_indices.add(index)
                self.scene_aliases[old_id] = new_id
                previous_scenes[index] = {**current, **prior, "analysis":json.loads(prior["analysis"]),"visual":json.loads(prior["visual"])}
            prior_version = stable_hash((MATCH_POLICY_VERSION,self.library_ids,[(s["id"],s["version"],s["analysis"],s["visual"],s["transcript"]) for s in previous_scenes]))
            self.previous_versions[prior_version] = sorted(affected_indices)
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
                cues = plot_cues(asset) if text else []
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
        if requirement.get("coverage_version") == COVERAGE_VERSION:
            return {**result,"claim":requirement["claim"],
                    "facts":[{k:f[k] for k in ("id","quote","description","constraints")} for f in requirement.get("facts",[])],
                    "constraints":[{k:c[k] for k in ("quote","description")} for c in requirement.get("constraints",[])],"motifs":requirement.get("motifs",[])}
        for name in ("claim", "characters", "events", "required_facts", "avoid"):
            if requirement.get(name) and requirement[name] != segment.text:
                result[name] = requirement[name]
        return result

    def configure_requirements(self, segments, revisions, gateway, previous_plan=None):
        """Accept only reproducible revisions from the current, validated manuscript plan."""
        self.requirement_history = {}
        self.previous_requirements = {u["id"]:u["requirement"] for u in previous_plan.get("units",[])} if isinstance(previous_plan,dict) else {}
        if not isinstance(revisions, dict) or revisions.get("version") != "gap-facts-review-v1" or \
                revisions.get("model_signature") != gateway.signature("text"):
            return
        rows = revisions.get("entries", [])
        if not isinstance(rows, list):
            return
        counts = Counter(row.get("unit_id") for row in rows if isinstance(row, dict) and isinstance(row.get("unit_id"), str))
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("unit_id"), str) or counts.get(row["unit_id"]) != 1:
                continue
            segment = segments.get(row["unit_id"])
            before, after = row.get("before"), row.get("after")
            if not segment or not isinstance(before, dict) or before.get("kind") not in {"evidence", "state", "abstract", "transition"} or \
                    after != segment.requirement or before == after:
                continue
            old_segment = replace(segment, requirement=before)
            equivalent = self.unit_payload(old_segment) == self.unit_payload(segment) and before.get("fact_sources") == after.get("fact_sources")
            if equivalent:
                # No field understood by the review may change through the metadata-only path.
                if {k: v for k, v in before.items() if k != "themes"} != {k: v for k, v in after.items() if k != "themes"}:
                    continue
                rule = "equivalent-review-input"
            else:
                decision, facts = row.get("decision"), before.get("fact_sources")
                if not isinstance(decision, dict) or not isinstance(facts, list) or not facts or \
                        decision.get("unit_id") != segment.id or decision.get("kind") not in {"evidence", "state", "abstract", "transition"} or \
                        not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
                    continue
                roles = decision.get("facts")
                if not isinstance(roles, list) or len(roles) != len(facts) or any(not isinstance(r, dict) or \
                        type(r.get("index")) is not int or r.get("role") not in {"concrete", "commentary"} or \
                        not isinstance(r.get("reason"), str) or not r["reason"].strip() for r in roles) or \
                        sorted(r["index"] for r in roles) != list(range(1, len(facts)+1)):
                    continue
                if any(not isinstance(f, dict) or not isinstance(f.get("quote"), str) or not f["quote"] or \
                       f["quote"] not in segment.text for f in facts):
                    continue
                roles = sorted(roles, key=lambda r: r["index"])
                hard = [facts[r["index"]-1] for r in roles if r["role"] == "concrete"]
                commentary = [facts[r["index"]-1]["quote"] for r in roles if r["role"] == "commentary"]
                expected = {**before, "kind": before["kind"] if hard else decision["kind"], "fact_sources": hard,
                    "required_facts": list(dict.fromkeys(f["quote"] for f in hard)),
                    "themes": list(dict.fromkeys([*before.get("themes", []), *commentary]))[:8]}
                if expected != after:
                    continue  # Characters, events, claim, avoid and bound facts cannot be added or rewritten.
                rule = "removed-commentary-only"
            self.requirement_history[segment.id] = {"segment": old_segment, "proof": {
                "rule": rule, "revision": stable_hash(row), "before": stable_hash(before), "after": stable_hash(after)}}

    def _match_keys(self, segment, candidate, subtitle_key, gateway, *, legacy_protocol=False):
        """Keep the original two cache identities shared by normal and compatibility lookup."""
        common = (MATCH_POLICY_VERSION, SUBTITLE_EVIDENCE_VERSION)
        tail = (candidate["subtitle_entries"], candidate["_verify_visual"], gateway.signature("text"), gateway.signature("vision"))
        legacy = "visual-unit-match:" + stable_hash((*common, self.version, candidate["_safety_key"], segment.id,
            segment.text, segment.requirement, subtitle_key, *tail))
        key = "visual-unit-match:" + stable_hash((*common, "range-context-v2", candidate["_safety_key"], segment.id,
            segment.text, segment.requirement, candidate["description"], *tail))
        if not legacy_protocol:
            key = "visual-unit-match:" + stable_hash((REVIEW_PROTOCOL, key))
            legacy = "visual-unit-match:" + stable_hash((REVIEW_PROTOCOL, legacy))
        return key, legacy

    def _compatible_judgment(self, segment, candidate, subtitle_key, gateway):
        # An unchanged complete legacy positive keeps its original full-contract provenance.
        # Negatives from the old mixed-unit protocol are never silently promoted or reused as new reviews.
        prior_requirement = self.previous_requirements.get(segment.id, segment.requirement)
        prior_segment = replace(segment, requirement=prior_requirement)
        for old_key in self._match_keys(prior_segment,candidate,subtitle_key,gateway,legacy_protocol=True):
            old = self.store.cache_get(candidate["library_id"],old_key)
            if not isinstance(old,dict) or old.get("evidence_protocol") != SUBTITLE_EVIDENCE_VERSION:
                continue
            if old.get("facts_match") is False:
                self.legacy_quarantine[old_key] = {"unit_id":segment.id,"asset_id":candidate["asset_id"],"source_start":candidate["source_start"],"source_end":candidate["source_end"],
                    "reason":"旧混合单元协议的否定未按当前单单元合同确认；保留旧结果，必要时重新审核。","old_response":old}
                continue
            if prior_requirement != segment.requirement or segment.requirement.get("coverage_version") == COVERAGE_VERSION:
                continue
            try:
                checked = _match_result({"clips":[old]},{candidate["_review_id"]:candidate},candidate["_verify_visual"])["clips"][0]
            except (ModelFormatError, KeyError, TypeError):
                continue
            view = {**checked,"pending_review":False,"match_policy":MATCH_POLICY_VERSION,"visual_checked":candidate["_verify_visual"]}
            if not checked["_validation_issues"] and old.get("usable") is True and old.get("content_kind") == "story" and self.range_allowed(candidate) and match_allowed(view,segment.requirement):
                return {**old,"compatible_from":{"version":REVIEW_PROTOCOL,"key":old_key,"mode":"legacy-full-contract","requirement_hash":stable_hash(segment.requirement)}}
        history = self.requirement_history.get(segment.id)
        if not history or not (candidate["_verify_visual"] or candidate.get("_safety")):
            return None
        for old_key in self._match_keys(history["segment"], candidate, subtitle_key, gateway, legacy_protocol=True):
            old = self.store.cache_get(candidate["library_id"], old_key)
            if not isinstance(old, dict) or old.get("evidence_protocol") != SUBTITLE_EVIDENCE_VERSION:
                continue
            try:
                checked = _match_result({"clips": [old]}, {candidate["_review_id"]: candidate}, candidate["_verify_visual"])["clips"][0]
            except (ModelFormatError, TypeError, KeyError):
                continue
            if checked["_validation_issues"] or checked.get("usable") is not True or checked.get("content_kind") != "story":
                continue
            view = {**checked, "pending_review": False, "match_policy": MATCH_POLICY_VERSION,
                    "visual_checked": candidate["_verify_visual"]}
            if not self.range_allowed(candidate) or not match_allowed(view, history["segment"].requirement) or \
                    not match_allowed(view, segment.requirement):
                continue
            # Preserve the canonical reply, including any saved provenance; no new model response is invented.
            return {**old, "compatible_from": {"version": RETRIEVAL_REPAIR_VERSION, "key": old_key,
                **history["proof"], "model_identity": "existing-cache-contract"}}
        return None

    def configure_queue(self, progress, gateway):
        """Restore per-pair scheduling only for the same review rules and configured models."""
        kinds = [k for k in ("text", "vision", "text_backup", "vision_backup") if k in gateway.settings]
        self.queue_signature = stable_hash((CANDIDATE_QUEUE_VERSION, REVIEW_PROTOCOL, [(k, gateway.signature(k),
            gateway.settings[k].get("enabled"), {name: gateway.settings[k].get(name) for name in
                ("thinking_level", "max_tokens", "max_prompt_chars", "context_tokens")}) for k in kinds]))
        self.candidate_progress = {k: dict(v) for k, v in progress.items() if isinstance(k, str) and
            isinstance(v, dict) and v.get("version") == CANDIDATE_QUEUE_VERSION and
            v.get("model_signature") == self.queue_signature and isinstance(v.get("unit_id"), str)} if isinstance(progress, dict) else {}

    def _remember(self, segment, candidate, judgment, *, cached=False, request_file=None):
        key = candidate["_review_key"]
        stats = self.statistics.setdefault(segment.id, {"checked": 0, "approved": 0, "unconfirmed": 0,
                                                       "cache_hits": 0, "reasons": {}, "examples": []})
        issues = judgment["_validation_issues"]
        if self.queue_signature:
            previous = self.candidate_progress.get(key, {})
            self.candidate_progress[key] = {"version": CANDIDATE_QUEUE_VERSION, "model_signature": self.queue_signature,
                "unit_id": segment.id, "state": "unconfirmed" if issues else "reviewed", "stage": "actual_review",
                "request_file": request_file or previous.get("request_file"),
                **({"judgment": {**{k: v for k, v in judgment.items() if not k.startswith("_")}, "scene_id": candidate["_review_id"]},
                    "reason": "; ".join(i["message"] for i in issues)} if issues else {})}
        if not issues:
            self.unconfirmed.discard(key)
            self.evaluated[segment.id].add(key)
            candidate.update({name: judgment[name] for name in ("relation", "facts_match", "evidence", "conflicts",
                "subtitle_quotes", "subtitle_ids", "visual_evidence", "evidence_protocol")})
            for name in ("coverage", "review_protocol", "compatible_from"):
                if name in judgment:
                    candidate[name] = judgment[name]
            candidate.update(usable=judgment["usable"], content_kind=judgment["content_kind"],
                visual_policy=VISUAL_POLICY_VERSION, match_policy=MATCH_POLICY_VERSION, visual_unit_id=segment.id,
                visual_checked=candidate["_verify_visual"], confidence=confidence(judgment.get("confidence")),
                reason=judgment["reason"], weak_match=judgment["relation"] != "direct" or not candidate["_verify_visual"])
        reviewed = {**candidate, "pending_review": False}
        accepted = not issues and candidate["usable"] and self.range_allowed(candidate) and match_allowed(reviewed, segment.requirement)
        if self.queue_signature:
            self.candidate_progress[key]["state"] = "approved" if accepted else "unconfirmed" if issues else "rejected"
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

    def prepare(self, segment, candidates, assets, gateway, verify_visual=True, *, scene_boundaries=None, _identity_only=False, diagnostic=None):
        """Read exact-range caches and rank locally; this method makes no model calls."""
        asset_map = {asset["id"]: asset for asset in assets}
        if scene_boundaries is not None:
            windows = {c["scene_id"] for c in candidates}
            candidates = natural_candidates(candidates, scene_boundaries, assets)
            if diagnostic is not None:
                diagnostic["natural_ranges"] = len(candidates)
                diagnostic["windows_without_natural_ranges"] = len(windows-{c["scene_id"] for c in candidates})
        def count(reason):
            if diagnostic is not None:
                diagnostic[reason] = diagnostic.get(reason, 0)+1
        unique = {}
        for original in candidates:
            candidate = dict(original)
            if candidate.get("scene_id") in self.scene_aliases:
                candidate["scene_id"] = self.scene_aliases[candidate["scene_id"]]
            asset, scene = asset_map.get(candidate.get("asset_id")), self.scene_map.get(candidate.get("scene_id"))
            if not asset or not scene or scene["asset_id"] != asset["id"] or asset["library_id"] not in self.library_ids:
                count("invalid_source")
                continue
            if candidate.get("fingerprint") != asset["fingerprint"] or candidate.get("fps") != asset["fps"] or \
                    any(type(candidate.get(k)) not in (int, float) or not math.isfinite(candidate[k]) for k in
                        ("source_start", "source_end", "natural_start", "natural_end")) or \
                    not 0 <= candidate["natural_start"] <= candidate["source_start"] < candidate["source_end"] <= candidate["natural_end"] <= asset["duration"]+.01:
                count("invalid_source")
                continue
            candidate.update(video_path=asset["path"], library_id=asset["library_id"], description=self.documents[self.scene_indices[scene["id"]]][0][:1800])
            if not _identity_only:
                self._load_exclusions(candidate, gateway)
                if not self.range_allowed(candidate):
                    count("source_unavailable")
                    continue
            safety_key = "visual-usability:" + stable_hash((VISUAL_POLICY_VERSION, candidate["fingerprint"],
                candidate["source_start"], candidate["source_end"], gateway.signature("vision")))
            candidate.update(_safety_key=safety_key, _review_id=stable_hash(safety_key)[:24], _verify_visual=verify_visual,
                             visual_unit_id=segment.id, _requirement=segment.requirement)
            entries, subtitle_key, state = self._range_subtitle(candidate, asset)
            candidate["subtitle_entries"] = entries
            key, legacy = self._match_keys(segment, candidate, subtitle_key, gateway)
            if candidate.get("_review_key") != key:
                for name in ("proposal_version", "_screened", "_preselection", "_preselection_rank", "compatible_from"):
                    candidate.pop(name, None)
            candidate.update(_review_key=key, subtitle_entries=entries, subtitle_state=state)
            if _identity_only:
                candidate["pending_review"] = True
                unique[source_shot_key(candidate)] = candidate
                continue
            candidate.pop("_rejected", None)
            if candidate.pop("_unconfirmed", False):
                candidate.pop("_screened", None)
            if key in self.rejected_matches:
                count("cached_rejections")
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
                        count("source_unavailable")
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
            compatibility = False
            if judgment is None:
                judgment = self._compatible_judgment(segment, candidate, subtitle_key, gateway)
                compatibility = judgment is not None
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
                    if valid and (compatibility or checked.get("compatible_from")):
                        candidate["compatible_from"] = checked["compatible_from"]
                    if not valid:
                        count("cached_rejections")
                        continue
            if not valid:
                for name in ("match_policy", "visual_policy", "facts_match", "relation", "confidence", "usable", "content_kind", "coverage", "review_protocol"):
                    candidate.pop(name, None)
                candidate["pending_review"] = True
                progress = self.candidate_progress.get(key, {})
                if progress.get("state") == "unconfirmed" and progress.get("unit_id") == segment.id and isinstance(progress.get("judgment"), dict):
                    try:
                        checked = _match_result({"clips": [progress["judgment"]]}, {candidate["_review_id"]: candidate}, verify_visual)["clips"][0]
                        if checked["_validation_issues"]:
                            self._remember(segment, candidate, checked, cached=True)
                    except (ModelFormatError, TypeError, KeyError):
                        pass  # A missing identity is not proof of an old visual failure or an approval.
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
            if previous is not None:
                count("duplicate_ranges")
        return sorted(unique.values(), key=lambda item: (not item.get("pending_review", True), item.get("confidence", item.get("proposal_score", 0))), reverse=True)

    def _restore_reply(self, saved, metadata, segments, assets, gateway, verify_visual, request_file, *, directory=None):
        """Recover incomplete replies only after reconstructing their exact original input key."""
        pairs = metadata.get("canonical_pairs")
        receipt = (directory is not None and saved.get("receipt_version") == "frozen-review-receipt-v1" and
                   saved.get("application_complete") is False and not saved.get("error") and
                   isinstance(saved.get("parsed_response"), dict) and metadata.get("review_protocol") == REVIEW_PROTOCOL)
        if not self.queue_signature or (not receipt and not any(isinstance(r, dict) and r.get("validation_issues") for r in saved.get("reviews", []))) or \
                metadata.get("usage_stage") not in ({"visual_review", "visual_safety"} if receipt else {"visual_review"}) or not isinstance(pairs, dict) or not pairs or \
                any(not isinstance(alias, str) or not re.fullmatch(r"P[1-9]\d*", alias) for alias in pairs):
            return
        aliases = {}
        for alias in sorted(pairs, key=lambda name: int(name[1:])):
            pair = pairs[alias]
            if not isinstance(pair, dict) or pair.get("unit_id") not in segments:
                return
            segment = segments[pair["unit_id"]]
            sources = [c for c in metadata.get("candidates", []) if isinstance(c, dict) and c.get("review_id") == pair.get("review_id")]
            if len(sources) != 1:
                return
            source = sources[0]
            asset = next((a for a in assets if a["id"] == source.get("asset_id")), None)
            if not asset or asset["fingerprint"] != source.get("fingerprint"):
                return
            candidates = self.prepare(segment, [{**source, "fps": asset["fps"]}], assets, gateway, verify_visual, _identity_only=not receipt)
            if len(candidates) != 1 or candidates[0]["_review_id"] != pair.get("review_id"):
                return
            aliases[alias] = (segment, candidates[0])
        safety_only = receipt and metadata.get("usage_stage") == "visual_safety"
        with_images = verify_visual or safety_only
        identity = stable_hash((PROPOSAL_VERSION, [(s.id, c["_review_key"]) for s, c in aliases.values()], with_images))
        if metadata.get("local_repair"):
            identity = stable_hash((identity, "row-repair-v1"))
        references = saved.get("subtitle_aliases")
        if saved.get("key") != identity or not isinstance(references, dict):
            return
        if metadata.get("review_protocol") == REVIEW_PROTOCOL:
            clips = {c["_safety_key"]: c for _, c in aliases.values()}
            expected_refs = {f"C{i}:L{j}": row["id"] for i, c in enumerate(clips.values(), 1)
                             for j, row in enumerate(c["subtitle_entries"], 1)}
            if len({s.id for s, _ in aliases.values()}) != 1 or references != expected_refs or metadata.get("subtitle_aliases") != references:
                return
        if receipt:
            relative = f"candidates/review-{identity[:24]}.json"
            if metadata.get("candidate_file") != relative or len({s.id for s, _ in aliases.values()}) != 1:
                return
            if not with_images and any(not (c.get("_safety") or {}).get("usable") for _, c in aliases.values()):
                return  # Text-only receipts still require the existing independent source check.
            result = _match_pairs(saved["parsed_response"], aliases, references, with_images,
                                  relevance=not safety_only, require_owner=not safety_only)
            skipped = {c["_review_key"] for _, c in aliases.values() if self.store.cache_get(c["library_id"],
                       c["_safety_key"] if safety_only else c["_review_key"]) is not None}
            job = {"key": identity, "file": relative, "unit_id": next(iter(aliases.values()))[0].id,
                   "aliases": aliases, "frozen": deepcopy(aliases), "references": references, "with_images": with_images,
                   "safety_only": safety_only, "repair": bool(metadata.get("local_repair")),
                   "search_round": metadata.get("search_round", 0), "skip_keys": skipped}
            self.commit_review(job, {"result": result, "requests": saved.get("requests", []),
                "parsed_response": saved["parsed_response"], "error": ""}, gateway, directory)
            for row in result["clips"]:
                segment, candidate = aliases[row["_pair"]]
                if candidate["_review_key"] not in skipped and row["_validation_issues"]:
                    self._remember(segment, candidate, row, cached=True, request_file=request_file)
            saved["application_complete"] = True
            return
        current_reply = metadata.get("review_protocol") == REVIEW_PROTOCOL and len({s.id for s, _ in aliases.values()}) == 1 and \
            isinstance(saved.get("parsed_response"), dict)
        value = saved.get("parsed_response") if current_reply else {"clips": []}
        if not current_reply:
            for alias, (segment, candidate) in aliases.items():
                for review in saved.get("reviews", []):
                    if isinstance(review, dict) and review.get("response_stage") == "normalized_model_reply" and \
                            (review.get("unit_id"), review.get("review_id"), review.get("source_start"), review.get("source_end")) == \
                            (segment.id, candidate["_review_id"], candidate["source_start"], candidate["source_end"]) and \
                            isinstance(review.get("response"), dict):
                        value["clips"].append({**review["response"], "scene_id": alias})
        result = _match_pairs(value, aliases, references, verify_visual, require_owner=metadata.get("review_protocol") == REVIEW_PROTOCOL)
        for row in result["clips"]:
            segment, candidate = aliases[row["_pair"]]
            key = candidate["_review_key"]
            progress = self.candidate_progress.get(key, {})
            latest_pending = current_reply and verify_visual and progress.get("state") == "unconfirmed" and \
                progress.get("request_file") == request_file and not saved.get("error")
            if (progress and not latest_pending) or self.store.cache_get(candidate["library_id"], key) is not None:
                continue  # Current cached conclusions take precedence over an older incomplete reply.
            if not row["_validation_issues"] and (latest_pending or (row.get("facts_match") is False and row.get("normalization"))):
                row["review_scope"] = REVIEW_SCOPE_VERSION
                if current_reply:
                    row["review_protocol"] = REVIEW_PROTOCOL
                self._load_exclusions(candidate, gateway)
                self.store.cache_put(candidate["library_id"], key, {k: v for k, v in row.items() if not k.startswith("_")})
                self._remember(segment, candidate, row, cached=True, request_file=request_file)
            elif row["_validation_issues"]:
                self._remember(segment, candidate, row, cached=True, request_file=request_file)

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
                    owners = {p.get("unit_id") for p in metadata.get("canonical_pairs",{}).values() if isinstance(p,dict)}
                    if metadata.get("usage_stage") == "visual_review" and len(owners) > 1 and metadata.get("review_protocol") != REVIEW_PROTOCOL:
                        for review in saved.get("reviews",[]):
                            if not isinstance(review,dict) or review.get("unit_id") not in segments:
                                continue
                            response = review.get("response",{})
                            if isinstance(response,dict) and response.get("facts_match") is False and segments[review["unit_id"]].requirement.get("coverage_version") == COVERAGE_VERSION:
                                quarantine_key = stable_hash((relative.as_posix(),review.get("unit_id"),review.get("review_id"),review.get("source_start"),review.get("source_end")))
                                self.legacy_quarantine[quarantine_key] = {"request_file":relative.as_posix(),"candidate_file":path.name,"unit_id":review["unit_id"],
                                    "review_id":review.get("review_id"),"source_start":review.get("source_start"),"source_end":review.get("source_end"),"old_response":response,
                                    "reason":"旧实际审片混合多个单元，当前缺口需单单元核对；原回复保留，未转为批准。"}
                    self._restore_reply(saved, metadata, segments, assets, gateway, verify_visual, relative.as_posix(), directory=directory)
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
        if self.legacy_quarantine:
            write_json(directory/"legacy-review-quarantine.json", {"version":REVIEW_PROTOCOL,"state":"needs_targeted_review","entries":self.legacy_quarantine})
        return result

    def shortlist(self, groups, gateway, directory, stop=None, *, search_round=0):
        """Rank bounded heads; a text protocol error is never a footage judgment."""
        from app.services.library.gateway import ModelAPIError, ModelInputBudgetError
        heads, pairs = {}, []
        for segment, candidates, needed in groups:
            head, capacity = [], 0
            for candidate in sorted(candidates, key=lambda c: c.get("proposal_version") == PROPOSAL_VERSION, reverse=True):
                if not candidate.get("pending_review") or candidate["_review_key"] in self.unconfirmed:
                    continue
                head.append(candidate)
                capacity += min(needed, candidate["source_end"]-candidate["source_start"])
                if capacity >= max(2, needed*2) or len(head) >= 8:
                    break
            heads[segment.id] = head
            pairs.extend((segment, c) for c in head if not c.get("_screened") and c.get("proposal_version") != PROPOSAL_VERSION)

        def submit(batch):
            if not batch:
                return
            check_cancel(stop)
            units = {s.id: s for s, _ in batch}
            clips = {c["_safety_key"]: c for _, c in batch}
            ua = {name: f"U{i}" for i, name in enumerate(units, 1)}
            ca = {key: f"C{i}" for i, key in enumerate(clips, 1)}
            allowed = {ua[name]: [ca[c["_safety_key"]] for s, c in batch if s.id == name] for name in units}
            canonical = {(s.id, c["_review_key"]) for s, c in batch}
            canonical_aliases = {(ua[s.id], ca[c["_safety_key"]]): [s.id, c["_review_key"]] for s, c in batch}
            identity = stable_hash((PROPOSAL_VERSION, gateway.signature("text_fast"),
                [(s.id, self.unit_payload(s), c["_review_key"], c["subtitle_entries"], c.get("observed_facts")) for s, c in batch]))
            path = Path(directory)/"planning"/f"preselect-{identity[:24]}.json"

            def validate(value):
                rows = value.get("selections") if isinstance(value, dict) else None
                if not isinstance(rows, list):
                    raise ModelFormatError("selections须为列表，unit_id和clip_ids使用本次给定编号。")
                selected, issues, seen = {name: [] for name in allowed}, {name: 0 for name in allowed}, set()
                unknown_rows = 0
                for row in rows:
                    owner = row.get("unit_id") if isinstance(row, dict) else None
                    if not isinstance(owner, str) or owner not in allowed:
                        unknown_rows += 1
                        continue
                    if owner in seen:
                        issues[owner] += 1
                    seen.add(owner)
                    ids = row.get("clip_ids")
                    if not isinstance(ids, list):
                        issues[owner] += 1
                        continue
                    for identifier in ids:
                        if not isinstance(identifier, str) or identifier not in allowed[owner] or identifier in selected[owner]:
                            issues[owner] += 1
                        else:
                            selected[owner].append(identifier)
                for owner in set(allowed)-seen:
                    issues[owner] += 1
                return {"selections": [{"unit_id": name, "clip_ids": ids} for name, ids in selected.items()],
                        "issues": issues, "unknown_rows": unknown_rows, "raw_response": value}

            def canonical_choices(parsed):
                return [canonical_aliases[(r["unit_id"], identifier)] for r in parsed["selections"] for identifier in r["clip_ids"]]

            try:
                saved = read_json(path, default={})
            except (OSError, ValueError):
                saved = {}
            chosen, reply, issues, unknown_rows, cache_hit = None, None, {}, 0, False
            if isinstance(saved, dict) and saved.get("key") == identity:
                # Only this exact frozen input may interpret old U/C aliases again.
                if isinstance(saved.get("response"), dict):
                    try:
                        parsed = validate(saved["response"])
                        chosen = canonical_choices(parsed)
                        reply, issues, unknown_rows = saved["response"], parsed["issues"], parsed["unknown_rows"]
                    except ModelFormatError:
                        pass
                if chosen is None and isinstance(saved.get("chosen"), list):
                    chosen = [list(r) for r in dict.fromkeys(tuple(r) for r in saved["chosen"] if isinstance(r, list) and len(r) == 2 and
                              all(isinstance(x, str) for x in r) and tuple(r) in canonical)]
                    old_issues = saved.get("issues", {})
                    issues = {ua[name]: old_issues.get(ua[name], 0) if isinstance(old_issues, dict) and
                        type(old_issues.get(ua[name], 0)) is int else 0 for name in units}
                    if saved.get("incomplete"):
                        issues = {name: max(1, value) for name, value in issues.items()}
                cache_hit = chosen is not None
            if chosen is None:
                if len(clips) <= 3:
                    chosen = [[s.id, c["_review_key"]] for s, c in batch]
                else:
                    payload = {"units": [{"id": ua[name], **self.unit_payload(s), "candidates": allowed[ua[name]]} for name, s in units.items()],
                        "clips": [{"id": ca[key], "background": c["description"], "observed": c.get("observed_facts", []), "observation_partial": True,
                                   "subtitles": [row["text"] for row in c["subtitle_entries"]]} for key, c in clips.items()]}
                    prompt = ("为文案预选值得进一步核对的少数镜头，只返回编号；预选不等于事实确认。"
                        "背景与字幕不证明实际画面必然出现某人或事件；observed只是部分线索，未列不等于不存在。具体人物/事件要对应，抽象论述可用有具体同向联系的情境。"
                        "遵守required_facts和avoid，否认关系不能支持建立关系。按相关性排序，证据不足可空，不按时长硬凑。"
                        "每个unit_id只能选择该单元candidates内的编号，不得跨单元。"
                        "返回{selections:[{unit_id,clip_ids:[]}]}，每个单元一次，不写解释。\n"+json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                    limit = int(gateway.settings["text_fast"].get("max_prompt_chars", 16000))
                    if len(prompt) > limit or len(prompt)*2+1024 >= int(gateway.settings["text_fast"].get("context_tokens", 64000)):
                        if len(batch) > 1:
                            submit(batch[:len(batch)//2]); submit(batch[len(batch)//2:])
                        else:
                            batch[0][1]["_preselection"] = {"version": RETRIEVAL_REPAIR_VERSION, "state": "input_budget", "issues": 1}
                            self.preselection.setdefault(batch[0][0].id, {"invalid_references": 0, "cache_hits": 0})["invalid_references"] += 1
                        return
                    gateway.context.update(stage="候选文字预选", segment="、".join(units))
                    try:
                        value = gateway.json(prompt, model_kind="text_fast", max_output=1200, required={"selections": list}, result_validator=validate,
                            request_metadata={"usage_stage": "text_preselect", "search_round": search_round,
                                              "pairs": [{"unit_id": s.id, "review_key": c["_review_key"]} for s, c in batch]})
                        reply, issues, unknown_rows = value["raw_response"], value["issues"], value["unknown_rows"]
                        chosen = canonical_choices(value)
                    except (ModelAPIError, ModelFormatError, ModelInputBudgetError) as exc:
                        # Keep the bounded local alternatives; no content judgment or retry loop is added here.
                        chosen, issues = [], {name: 1 for name in allowed}
                        reply = {"protocol_error": gateway._redact_request(str(exc))[:500]}
            write_json(path, {"key": identity, "parse_version": RETRIEVAL_REPAIR_VERSION, "chosen": chosen, "response": reply,
                "incomplete": False, "issues": issues, "unknown_rows": unknown_rows})
            for owner in units:
                stats = self.preselection.setdefault(owner, {"invalid_references": 0, "cache_hits": 0})
                stats["invalid_references"] += issues.get(ua[owner], 0)
                stats["cache_hits"] += int(cache_hit)
                if owner == next(iter(units)):
                    stats["unknown_rows"] = stats.get("unknown_rows", 0)+unknown_rows
            for segment, candidate in batch:
                picked = [segment.id, candidate["_review_key"]] in chosen
                candidate["_screened"] = True
                candidate["_preselection"] = {"version": RETRIEVAL_REPAIR_VERSION,
                    "state": "model" if picked else "protocol" if issues.get(ua[segment.id]) else "unselected",
                    "issues": issues.get(ua[segment.id], 0)}
                if picked:
                    candidate["proposal_version"] = PROPOSAL_VERSION
                    candidate["_preselection_rank"] = chosen.index([segment.id, candidate["_review_key"]])
                    candidate["proposal_score"] = 1/(1+candidate["_preselection_rank"])

        batch, unique, unit_ids = [], set(), set()
        for pair in pairs:
            source = pair[1]["_safety_key"]
            if batch and (len(unique | {source}) > 20 or len(unit_ids | {pair[0].id}) > 4):
                submit(batch)
                batch, unique, unit_ids = [], set(), set()
            batch.append(pair); unique.add(source); unit_ids.add(pair[0].id)
        submit(batch)
        result = {}
        for segment, _, _ in groups:
            head = heads[segment.id]
            selected = sorted((c for c in head if c.get("proposal_version") == PROPOSAL_VERSION),
                              key=lambda c: c.get("_preselection_rank", 0))
            # Empty or locally repaired replies keep two ranked alternatives, never an approval.
            if not selected:
                selected = [c for c in head if c.get("_preselection", {}).get("state") != "input_budget"][:2]
                for candidate in selected:
                    candidate.update(proposal_version=PROPOSAL_VERSION, _screened=True)
                    candidate["_preselection"] = {**candidate.get("_preselection", {}),
                        "version": RETRIEVAL_REPAIR_VERSION, "state": "local-alternative"}
            result[segment.id] = {"keys": [c["_review_key"] for c in selected],
                "protocol_issues": max((c.get("_preselection", {}).get("issues", 0) for c in head), default=0),
                "local_alternatives": sum(c.get("_preselection", {}).get("state") == "local-alternative" for c in selected)}
        return result

    def prepare_reviews(self, pairs, gateway, directory, *, search_round=0, _safety_only=False, _repair=None):
        """Freeze jobs on the task thread; do not request a model or modify candidates."""
        groups, jobs = defaultdict(list), []
        for segment, candidate in pairs:
            groups[segment.id].append((segment, candidate))
        batches = [group[offset:offset+6] for group in groups.values() for offset in range(0, len(group), 6)]
        for batch in batches:
            with_images = _safety_only or batch[0][1]["_verify_visual"]
            if not with_images and any(c.get("_safety") is None for _, c in batch):
                jobs.extend(self.prepare_reviews([(s, c) for s, c in batch if c.get("_safety") is None], gateway,
                    directory, search_round=search_round, _safety_only=True, _repair=_repair))
                batch = [(s, c) for s, c in batch if c.get("_safety") and c["_safety"]["usable"] and c["_review_key"] not in self.unconfirmed]
                if not batch:
                    continue
            units = {s.id: s for s, _ in batch}
            clips = {c["_safety_key"]: c for _, c in batch}
            ua = {name: f"U{i}" for i, name in enumerate(units, 1)}
            ca = {key: f"C{i}" for i, key in enumerate(clips, 1)}
            aliases = {f"P{i}": pair for i, pair in enumerate(batch, 1)}
            references = {f"{ca[key]}:L{i}": row["id"] for key, c in clips.items() for i, row in enumerate(c["subtitle_entries"], 1)}
            payload = {"units": [{"id": ua[name], **self.unit_payload(s)} for name, s in units.items()],
                "clips": [{"id": ca[key], "background": c["description"],
                    "subtitles": [{"id": f"{ca[key]}:L{i}", "text": row["text"]} for i, row in enumerate(c["subtitle_entries"], 1)]}
                    for key, c in clips.items()],
                "pairs": [{"scene_id": alias, "unit": ua[s.id], "clip": ca[c["_safety_key"]]} for alias, (s, c) in aliases.items()]}
            prompt = ("本次只有一个文案单元U1。每个pairs中的P编号恰好返回一条结果，scene_id抄P编号，不抄C编号；unit_id固定U1。"
                "每个C素材按clips顺序附3图，三图共同审核一个镜头，不逐图返回三行。frame_index必须填本素材内整数0、1或2，不填3。"
                "usable仅判断画面：剧情story为true，职员表/演职员名单credits、混合mixed、不明unknown为false；普通对白/歌词不因此禁用。"
                "confidence必须为0至1的有限数字。人物/事件不符只影响relation/facts_match/conflicts，不得因此将story标为不可用。"
                "visual_evidence仅写可见动作/状态/环境，不写对白；subtitle_ids只选本素材的C编号字幕，不跨素材、不抄引文。"
                "字幕可能仅部分重叠，不能声称整句在范围内说完。依据和理由简短；visual_evidence建议最多3条，conflicts和subtitle_ids各最多3项。\n")
            example = {"scene_id": next(iter(aliases)), "unit_id": "U1", "usable": True, "content_kind": "story",
                "reason": "当前画面与本单元缺少明确联系", "confidence": 0.8, "frame_index": 0,
                "relation": "unrelated", "facts_match": False, "visual_evidence": [], "conflicts": [], "subtitle_ids": []}
            if batch[0][0].requirement.get("coverage_version") == COVERAGE_VERSION:
                prompt += (
                    "facts为分别需要的原子事实，可由多个镜头共同覆盖；当前镜头只列它明确支持的F编号。未见其他F是未覆盖，不能因此写conflicts。"
                    "supports每项含fact_id（抄当前facts的id）、visual_indices（本C的0至2整数列表）、subtitle_ids（本C字幕编号列表）、reason（具体依据）；每F最多一次。"
                    "人物出现不等于事件发生，被台词提及不等于出镜，不用背景剧情补未观察到的事件；字幕只有范围重叠部分可见，不宣称完整台词已说完。"
                    "该F内部constraints只约束该事实；全局constraints约束本镜头，constraints_match表示全部全局约束满足。"
                    "conflicts只列与当前全局限制明确矛盾的内容，不列未拍到的事实，不跨F混用前世/现世或全部人物。"
                    "relation=direct表示直接支持至少一个具体F；supporting表示有具体同向联系的辅助配图；unrelated/unknown不能放行。"
                    "facts_match=true仅表示单镜头已覆盖全部F，部分满足仍false并保留supports；不输出partial。"
                    "facts为空时supports必须为[]，不得编造空fact_id、M1或“无”；依据motifs判断motif_match，须与原文有明确可见联系，不能靠主题词相同凑时长。"
                    "constraints_match、motif_match、facts_match只用布尔值。\n")
                example.update(constraints_match=True, motif_match=False, supports=[])
            else:
                prompt += ("relation为direct/supporting/unrelated/unknown，facts_match只用true或false，部分满足用false。"
                    "evidence类须直接证据；abstract/state/transition可用有具体同向联系的辅助画面，不要求单镜头证明整套理论。"
                    "遵守人物/时期/事件/关系及avoid；提及人物不等于出现，否认不能支持建立关系；背景仅作线索，不用窗口外剧情补本镜头。\n")
            if not with_images:
                prompt = prompt.replace(
                    "每个C素材按clips顺序附3图，三图共同审核一个镜头，不逐图返回三行。frame_index必须填本素材内整数0、1或2，不填3。",
                    "本次无新图，已独立检查画面可用性；据范围字幕和已有描述核对内容，不明细节标unknown，supports.visual_indices必须为[]。")
            if _safety_only:
                prompt = ("仅检查实际画面可用性，每个P编号一次。每个C按顺序附3张图，frame_index为该C内0至2。"
                    "story剧情为usable=true；职员表/演职员名单credits、混合mixed、不明unknown为false。普通对白、歌词不因此禁用。"
                    "三图共同审核一个镜头，不逐图返回三行；scene_id抄P编号，不抄C编号。不判断文案关系。\n")
                example = {k: example[k] for k in ("scene_id", "usable", "content_kind", "reason", "confidence", "frame_index")}
                example["reason"] = "画面为剧情内容"
                payload = {"clips": [{"id": ca[key]} for key in clips],
                           "pairs": [{"scene_id": a, "clip": ca[c["_safety_key"]]} for a, (_, c) in aliases.items()]}
            prompt += ("以下为一条结果的合法JSON结构示例，不预设结论，字段值按实际观察填写；实际clips须覆盖pairs的每个P一次："
                       + json.dumps({"clips": [example]}, ensure_ascii=False, separators=(",", ":")) + "\n")
            if _repair:
                payload["repair"] = _repair_feedback(aliases, references, _repair)
                prompt += ("这是一次失败条目的局部纠错。previous保留实际坏行，已换成当前目录编号；returned_scene_ids保留旧回复原编号，仅作诊断。"
                    "outside:/invalid:表示当前目录外或错误引用。按expected_scene_id和当前U/P/C目录输出完整结果；证据不足应明确不匹配，不为了通过而改成true。\n")
                prompt += "frame_index_valid是本地标记，不要输出；若为false，旧frame_index可能是默认值，须根据图片重新选0至2。\n"
            prompt = prompt+json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            identity = stable_hash((PROPOSAL_VERSION, [(s.id, c["_review_key"]) for s, c in batch], with_images))
            if _repair:
                identity = stable_hash((identity, "row-repair-v1"))
            profile = gateway.settings["vision" if with_images else "text"]
            too_large = len(prompt) > int(profile.get("max_prompt_chars", 16000)) or \
                len(prompt)*2+(3*len(clips)*1600 if with_images else 0)+1024 >= int(profile.get("context_tokens", 64000))
            if too_large and len(batch) > 1:
                middle = len(batch)//2
                jobs.extend(self.prepare_reviews(batch[:middle], gateway, directory, search_round=search_round, _safety_only=_safety_only, _repair=_repair))
                jobs.extend(self.prepare_reviews(batch[middle:], gateway, directory, search_round=search_round, _safety_only=_safety_only, _repair=_repair))
                continue
            frozen = deepcopy(aliases)
            frames = {c["_safety_key"]: str(self.store.directory(c["library_id"])/"frames"/c["asset_id"]/c["_review_id"]) for _, c in frozen.values()}
            jobs.append({"key": identity, "file": f"candidates/review-{identity[:24]}.json", "unit_id": batch[0][0].id,
                "aliases": aliases, "frozen": frozen, "references": references, "frames": frames, "prompt": prompt,
                "with_images": with_images, "safety_only": _safety_only, "repair": bool(_repair), "search_round": search_round,
                "input_error": "单镜头输入超过本模型预算，请调整模型上下文设置。" if too_large else ""})
        return jobs

    def request_review(self, job, gateway, stop=None, *, media_slots=None):
        """Use an isolated client and frozen data; shared caches/state are committed elsewhere."""
        from app.services.library.gateway import ModelAPIError
        aliases, references, with_images = job["frozen"], job["references"], job["with_images"]
        clips = {c["_safety_key"]: c for _, c in aliases.values()}
        outcome = {"requests": [], "error": job["input_error"], "result": None, "parsed_response": None}
        if not outcome["error"]:
            check_cancel(stop)
            images = []
            if with_images:
                with media_slots if media_slots is not None else nullcontext():
                    for candidate in clips.values():
                        check_cancel(stop)
                        candidate["frame_paths"] = extract_frames(candidate["video_path"], candidate["source_start"],
                            candidate["source_end"], 3, Path(job["frames"][candidate["_safety_key"]]))
                        images.extend(candidate["frame_paths"])
            def validate(value):
                outcome["parsed_response"] = value
                return _match_pairs(value, aliases, references, with_images, relevance=not job["safety_only"], require_owner=not job["safety_only"])
            previous_after = gateway.after_request
            def after(details):
                outcome["requests"].append({k: details[k] for k in ("request_file", "response_file", "usage", "http_status") if k in details})
                if previous_after:
                    previous_after(details)
            gateway.context.update(stage="检查句意与实际镜头", segment=job["unit_id"])
            gateway.after_request = after
            try:
                outcome["result"] = gateway.json(job["prompt"], images=images or None, model_kind="vision" if with_images else "text",
                    max_output=max(2400, len(aliases)*1100), required={"clips": list}, result_validator=validate,
                    attempts=1 if job["repair"] else 3,
                    request_metadata={"usage_stage": "visual_safety" if job["safety_only"] else "visual_review", "search_round": job["search_round"],
                        "candidate_file": job["file"], "canonical_pairs": {alias: {"unit_id": s.id, "review_id": c["_review_id"]} for alias, (s, c) in aliases.items()},
                        "subtitle_aliases": references, "local_repair": job["repair"], "review_protocol": REVIEW_PROTOCOL,
                        "candidates": self._request_sources(list(clips.values()), with_images=with_images)})
            except (ModelFormatError, ModelAPIError) as exc:
                outcome["error"] = gateway._redact_request(str(exc))[:1000]
            finally:
                gateway.after_request = previous_after
        if outcome["result"] is None:
            outcome["result"] = _match_pairs({"clips": []}, aliases, references, with_images)
            for row in outcome["result"]["clips"]:
                row["_validation_issues"] = [{"field": "input" if job["input_error"] else "request", "category": "input_budget" if job["input_error"] else "request_failed", "message": outcome["error"]}]
        return outcome

    def commit_review(self, job, outcome, gateway, directory):
        """Persist every returned row and apply shared conclusions on the task thread."""
        aliases, references, result = job["aliases"], job["references"], outcome["result"]
        path = Path(directory)/job["file"]
        try:
            previous = read_json(path, default={})
        except (OSError, ValueError):
            previous = {}
        history = previous.get("history", []) if isinstance(previous, dict) else []
        history = list(history) if isinstance(history, list) else []
        if isinstance(previous, dict) and previous.get("reviews"):
            history.append({k: previous[k] for k in ("reviews", "requests", "subtitle_aliases", "parsed_response", "error") if k in previous})
        diagnostics = [{"unit_id": aliases[row["_pair"]][0].id, "review_id": aliases[row["_pair"]][1]["_review_id"],
            "source_start": aliases[row["_pair"]][1]["source_start"], "source_end": aliases[row["_pair"]][1]["source_end"],
            "response": row["_raw_response"], "response_stage": "normalized_model_reply", "validation_issues": row["_validation_issues"],
            "normalization": row.get("normalization")} for row in result["clips"]]
        audit = {"key": job["key"], "version": MATCH_POLICY_VERSION, "value": [], "reviews": diagnostics,
            "requests": outcome["requests"], "subtitle_aliases": references, "history": history, "local_repair": job["repair"],
            "parsed_response": outcome["parsed_response"], "error": outcome["error"], "discarded_replies": result.get("discarded_replies", []),
            "receipt_version": "frozen-review-receipt-v1", "application_complete": False}
        write_json(path, audit)  # A received reply must survive failure during cache application.
        approved, deferred, summary = [], [], Counter()
        request_file = outcome["requests"][-1].get("request_file") if outcome["requests"] else None
        for row in result["clips"]:
            segment, candidate = aliases[row["_pair"]]
            if candidate["_review_key"] in job.get("skip_keys", ()):
                summary["cached"] += 1
                continue  # Recovery never overwrites a conclusion already present under the exact current key.
            owner, frozen = job["frozen"][row["_pair"]]
            if owner.id != segment.id or owner.requirement != segment.requirement or any(candidate.get(k) != frozen.get(k) for k in
                    ("_review_key", "fingerprint", "source_start", "source_end", "natural_start", "natural_end")):
                raise ValueError("审核期间原文需求或来源身份发生变化，回包已保存，请从3. 寻找画面继续核对。")
            if "frame_paths" in frozen:
                candidate["frame_paths"] = frozen["frame_paths"]
            candidate.pop("proposal_version", None)
            row.update(review_scope=REVIEW_SCOPE_VERSION, review_protocol=REVIEW_PROTOCOL)
            issues = row["_validation_issues"]
            if issues:
                summary["visual_unconfirmed" if any(i["category"] == "unknown_visual" for i in issues) else "protocol_unconfirmed"] += 1
            elif not row["usable"]:
                summary["source_excluded"] += 1
            elif not job["safety_only"]:
                summary["content_matched" if match_allowed({**candidate, **row, "pending_review": False,
                    "match_policy": MATCH_POLICY_VERSION, "visual_checked": candidate["_verify_visual"]}, segment.requirement)
                        else "content_rejected"] += 1
            if job["with_images"] and row["_usability_valid"]:
                safety = {k: row[k] for k in ("scene_id", "usable", "content_kind", "reason")}
                safety["scope"] = REVIEW_SCOPE_VERSION
                if safety["content_kind"] != "unknown" or job["safety_only"]:
                    self.store.cache_put(candidate["library_id"], candidate["_safety_key"], safety)
                candidate["_safety"] = safety
                if not safety["usable"]:
                    self._reject(candidate, safety, gateway)
                elif isinstance(row.get("visual_evidence"), list) and all(isinstance(x, str) for x in row["visual_evidence"]):
                    self.store.cache_put(candidate["library_id"], "visual-observation:"+stable_hash(candidate["_safety_key"]), {"partial": True, "facts": row["visual_evidence"]})
            if row["_validation_issues"] and not job["repair"] and not outcome["error"] and not any(i["category"] == "unknown_visual" for i in row["_validation_issues"]):
                deferred.append((segment, candidate, row))
            else:
                if not job["safety_only"] and not row["_validation_issues"]:
                    self.store.cache_put(candidate["library_id"], candidate["_review_key"], {k: v for k, v in row.items() if not k.startswith("_")})
                if (not job["safety_only"] or row["_validation_issues"]) and self._remember(segment, candidate, row, request_file=request_file) and not job["safety_only"]:
                    approved.append(candidate)
        audit["value"] = [{k: v for k, v in c.items() if k != "_requirement"} for c in approved]
        audit["review_summary"] = {**summary, "approved": len(approved)}
        audit["application_complete"] = True
        write_json(path, audit)
        followups = []
        if deferred:
            repair = {c["_review_key"]: {"previous": row["_raw_response"], "issues": row["_validation_issues"],
                "aliases": aliases, "references": references} for _, c, row in deferred}
            followups.extend(self.prepare_reviews([(s, c) for s, c, _ in deferred], gateway, directory,
                search_round=job["search_round"], _safety_only=job["safety_only"], _repair=repair))
        if job["safety_only"]:
            ready = [(s, c) for s, c in aliases.values() if (c.get("_safety") or {}).get("usable") and
                     c["_review_key"] not in self.unconfirmed and not any(c is bad for _, bad, _ in deferred)]
            followups.extend(self.prepare_reviews(ready, gateway, directory, search_round=job["search_round"]))
        return approved, followups

    def review_pairs(self, pairs, gateway, directory, stop=None, *, search_round=0, _safety_only=False, _repair=None):
        """Synchronous caller of the same frozen jobs, validator and shared commit path."""
        jobs = self.prepare_reviews(pairs, gateway, directory, search_round=search_round, _safety_only=_safety_only, _repair=_repair)
        accepted = []
        while jobs:
            check_cancel(stop)
            job = jobs.pop(0)
            client = copy(gateway)
            client.context = dict(gateway.context)
            value = self.request_review(job, client, stop)
            approved, followups = self.commit_review(job, value, gateway, directory)
            accepted.extend(approved)
            jobs[0:0] = followups
        return accepted

    def select(self, segment, overview, gateway, directory, used=(), verify_visual=True, stop=None, *, candidates, scene_boundaries, assets):
        prepared = self.prepare(segment, candidates, assets, gateway, verify_visual, scene_boundaries=scene_boundaries)
        prepared = [c for c in prepared if unused_source(c, used)]
        return [c for c in prepared if not c.get("pending_review")] + self.review_pairs(
            [(segment, c) for c in prepared if c.get("pending_review") and unused_source(c, used)], gateway, directory, stop)
