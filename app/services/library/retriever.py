"""Local lexical/semantic-label retrieval followed by bounded model review."""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pysrt

from app.services.library.frame_extractor import extract_frames
from app.services.library.runner import check_cancel
from app.services.library.store import library_ids, stable_hash, write_json
from app.services.subtitle_segmentation import _tokenizer
from app.services.library.validation import confidence, frame_index, ModelFormatError
from app.services.timeline_planner import (
    VISUAL_POLICY_VERSION, MATCH_POLICY_VERSION, match_allowed, natural_candidates, source_shot_key, unused_source,
)


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


def _review_result(value, expected):
    rows = value.get("clips", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows) or \
            len(rows) != len(expected) or {row.get("scene_id") for row in rows} != set(expected):
        raise ModelFormatError("请逐一返回全部给定镜头的 scene_id，每个仅一次，不得遗漏或新增。")
    for row in rows:
        if type(row.get("usable")) is not bool or row.get("content_kind") not in {"story", "credits", "mixed", "unknown"} or \
                not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ModelFormatError("每个镜头须含 usable 布尔值、content_kind（story/credits/mixed/unknown）及非空 reason。")
        if row["usable"] and row["content_kind"] != "story":
            raise ModelFormatError("只有明确的 story 画面可标记 usable=true，名单、混合及不明画面必须为 false。")
    return value


def _match_result(value, expected, with_images):
    _review_result(value, expected)
    for row in value["clips"]:
        if row.get("relation") not in {"direct", "supporting", "unrelated", "unknown"} or type(row.get("facts_match")) is not bool:
            raise ModelFormatError("每个候选须含 relation(direct/supporting/unrelated/unknown) 和 facts_match 布尔值。")
        for name in ("evidence", "conflicts", "subtitle_quotes"):
            items = row.get(name)
            if not isinstance(items, list) or any(not isinstance(x, str) or not x.strip() for x in items):
                raise ModelFormatError(f"候选{name}必须是字符串列表，可为空。")
        if row["relation"] in {"direct", "supporting"} and not row["evidence"]:
            raise ModelFormatError("direct/supporting必须给出可核对的具体画面或台词依据。")
        if with_images and not row.get("frame_index_valid", False):
            raise ModelFormatError("每个视觉检查候选须返回合法 frame_index，范围0至2。")
        transcript = expected[row["scene_id"]].get("range_subtitle", "")
        if any(quote not in transcript for quote in row["subtitle_quotes"]):
            raise ModelFormatError("subtitle_quotes只能逐字引用当前实际范围给定的字幕，不得引用窗口外台词或编造引文。")
    return value


class MaterialRetriever:
    def __init__(self, store, library_id):
        self.store = store
        self.library_ids = sorted(library_ids(library_id))
        self.library_names = {item["id"]: item["name"] for item in store.list_libraries()}
        self.scenes = store.scenes(self.library_ids)
        if not self.scenes:
            raise ValueError("素材库尚无可用的成功场景，请先建立或继续分析素材库。")
        self.version = stable_hash((MATCH_POLICY_VERSION, self.library_ids,
            [(s["id"], s["version"], s["analysis"], s["visual"], s["transcript"]) for s in self.scenes]))
        self.documents = []
        self.subtitles = {}
        self.statistics = {}
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
            description = json.dumps({"subtitle_analysis": analysis, "window_visual": visual}, ensure_ascii=False)
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
                for row in rows if isinstance(rows, list) else []:
                    if isinstance(row, dict) and row.get("fingerprint") == candidate["fingerprint"] and \
                            all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) for k in ("source_start", "source_end")) and \
                            0 <= row["source_start"] < row["source_end"]:
                        self.forbidden.append(row)
            self.exclusion_keys.add(key)
        return key

    def _reject(self, candidate, judgment, gateway):
        key = self._load_exclusions(candidate, gateway)
        row = {name: candidate[name] for name in ("fingerprint", "source_start", "source_end")}
        row["content_kind"] = judgment["content_kind"]
        if row not in self.forbidden:
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
                self.subtitles[identifier] = (stable_hash(text), rows, "processed" if rows else "missing")
            except (OSError, ValueError):
                self.subtitles[identifier] = (stable_hash("missing"), [], "missing")
        identity, rows, state = self.subtitles[identifier]
        transcript = "\n".join(text for start, end, text in rows if end > candidate["source_start"] and start < candidate["source_end"])
        return transcript, identity, state

    def select(self, segment, overview, gateway, directory, used=(), verify_visual=True, stop=None, *, candidates, scene_boundaries, assets):
        """Every admitted natural shot receives an explicit review for this exact unit."""
        assets = {asset["id"]: asset for asset in assets}
        key = stable_hash((MATCH_POLICY_VERSION, self.version, segment.id, segment.text, segment.requirement,
                          gateway.signature("text"), gateway.signature("vision"), verify_visual,
                          [(item["scene_id"], item["source_start"], item["source_end"]) for item in candidates]))
        cache_file = Path(directory) / "candidates" / f"{segment.id}-{key[:20]}.json"
        statistics = self.statistics.setdefault(segment.id, {"checked": 0, "approved": 0, "reasons": {}, "examples": []})
        gateway.context.update(stage="检查句意与实际镜头", segment=segment.id)
        unique = {}
        for candidate in natural_candidates(candidates, scene_boundaries, list(assets.values())):
            if not unused_source(candidate, used):
                continue
            identity = source_shot_key(candidate)
            previous = unique.get(identity)
            if previous is None or candidate["source_end"]-candidate["source_start"] > previous["source_end"]-previous["source_start"]:
                unique[identity] = candidate

        def review_batch(batch, *, relevance, with_images):
            check_cancel(stop)
            images = []
            if with_images:
                for candidate in batch:
                    check_cancel(stop)
                    output_dir = self.store.directory(candidate["library_id"]) / "frames" / candidate["asset_id"] / candidate["_review_id"]
                    candidate["frame_paths"] = extract_frames(candidate["video_path"], candidate["source_start"], candidate["source_end"], 3, output_dir)
                    images.extend(candidate["frame_paths"])
            expected = {item["_review_id"]: item for item in batch}
            prompt = (
                "检查每个实际裁剪范围，逐一返回全部scene_id且每项仅一次。"
                "任一给定图出现滚动职员表、演职员名单或制作/声优等演职员字幕，usable=false；混合或未知也为false。"
                "普通对白字幕、歌词、剧情中的文字不因此禁用。content_kind为story/credits/mixed/unknown，仅story可usable=true。"
                + ("每个候选按顺序附3张实际范围图片，frame_index为该候选0至2。" if with_images else
                   "本次不附新图片，已独立完成画面可用性检查；根据范围字幕和已有描述判断内容，无法确认的细节标unknown。")
            )
            fields = "scene_id,usable:true或false,content_kind,reason,confidence:0至1,frame_index:0至2"
            if relevance:
                prompt += (
                    "内容关系与画面可用性分开判断。relation为direct(直接支持当前具体原文)/supporting(有具体联系的辅助)/unrelated/unknown。"
                    "facts_match表示当前required_facts全部得到支持；身份、事件、时期或关系不符为false，并在conflicts列出冲突。"
                    "evidence列可核对的实际动作、状态或台词；只有同作品、出现同一个人、同样谈成长/责任，不足以证明具体事件。"
                    "evidence类原文必须有直接证据，不能用情绪空镜代替。抽象论述可用有具体关联的辅助画面，但不能编造剧情事实。"
                    "人物在字幕中被提及不等于出现在画面；窗口描述只作背景，不能把窗口外事件当作本镜头发生的事。"
                    "遵守avoid。subtitle_quotes只逐字引用本候选range_subtitle；没有可引用台词时为空列表，不猜人物身份。"
                    "未选中或相关性不足的内容不能以过渡名义放行；reason不超过120字，每项evidence/conflicts/subtitle_quotes最多3条。"
                    f"\n当前原文：{segment.text}\n当前需求：{json.dumps(segment.requirement, ensure_ascii=False)}"
                    f"\n全文背景（不用于替代当前原文）：{overview[:800]}"
                )
                fields += ",relation,facts_match:true或false,evidence:[具体依据],conflicts:[],subtitle_quotes:[]"
            payload = [{"scene_id": item["_review_id"], "source_start": item["source_start"], "source_end": item["source_end"],
                        "window_description": item["description"], "range_subtitle": item["range_subtitle"],
                        "subtitle_state": item["subtitle_state"]} for item in batch]
            result = gateway.json(prompt + f"\n返回JSON {{clips:[{{{fields}}}]}}。\n候选顺序：" + json.dumps(payload, ensure_ascii=False),
                images=images or None, model_kind="vision" if with_images else "text", max_output=3600,
                required={"clips": list}, result_validator=(lambda value: _match_result(value, expected, with_images)) if relevance else
                    (lambda value: _review_result(value, set(expected))),
                request_metadata={"visual_unit_id": segment.id, "candidates": self._request_sources(batch, with_images=with_images),
                                  "selected_scene_ids": [item["scene_id"] for item in batch]})
            return {row["scene_id"]: row for row in result["clips"]}

        pending, selection = [], []
        for candidate in unique.values():
            check_cancel(stop)
            self._load_exclusions(candidate, gateway)
            if not self.range_allowed(candidate):
                continue
            transcript, subtitle_key, subtitle_state = self._range_subtitle(candidate, assets[candidate["asset_id"]])
            safety_key = "visual-usability:" + stable_hash((VISUAL_POLICY_VERSION, candidate["fingerprint"],
                candidate["source_start"], candidate["source_end"], gateway.signature("vision")))
            review_key = "visual-unit-match:" + stable_hash((MATCH_POLICY_VERSION, self.version, safety_key, segment.id,
                segment.text, segment.requirement, subtitle_key, transcript, verify_visual,
                gateway.signature("text"), gateway.signature("vision")))
            candidate.update(_safety_key=safety_key, _review_key=review_key, _review_id=stable_hash(safety_key)[:24],
                             range_subtitle=transcript, subtitle_state=subtitle_state)
            safety = self.store.cache_get(candidate["library_id"], safety_key)
            judgment = self.store.cache_get(candidate["library_id"], review_key)
            try:
                if safety is not None:
                    _review_result({"clips": [safety]}, {candidate["_review_id"]})
                    if not safety["usable"]:
                        self._reject(candidate, safety, gateway)
                        continue
            except (ModelFormatError, TypeError, KeyError):
                safety = None
            if judgment is not None:
                try:
                    _match_result({"clips": [judgment]}, {candidate["_review_id"]: candidate}, verify_visual)
                except (ModelFormatError, TypeError, KeyError):
                    judgment = None
            candidate["_safety"] = safety
            if judgment is None or (not verify_visual and safety is None):
                pending.append(candidate)
            else:
                candidate["_judgment"] = judgment
                selection.append(candidate)
        if not verify_visual:
            safety_pending = [item for item in pending if item["_safety"] is None]
            for offset in range(0, len(safety_pending), 6):
                batch = safety_pending[offset:offset+6]
                rows = review_batch(batch, relevance=False, with_images=True)
                for candidate in batch:
                    safety = rows[candidate["_review_id"]]
                    self.store.cache_put(candidate["library_id"], candidate["_safety_key"], safety)
                    candidate["_safety"] = safety
                    if not safety["usable"]:
                        self._reject(candidate, safety, gateway)
        for offset in range(0, len(pending), 6):
            batch = [item for item in pending[offset:offset+6] if self.range_allowed(item)]
            if not batch:
                continue
            rows = review_batch(batch, relevance=True, with_images=verify_visual)
            for candidate in batch:
                judgment = rows[candidate["_review_id"]]
                self.store.cache_put(candidate["library_id"], candidate["_review_key"], judgment)
                if verify_visual:
                    safety = {k: judgment[k] for k in ("scene_id", "usable", "content_kind", "reason")}
                    self.store.cache_put(candidate["library_id"], candidate["_safety_key"], safety)
                    if not safety["usable"]:
                        self._reject(candidate, safety, gateway)
                candidate["_judgment"] = judgment
                selection.append(candidate)
        approved, rejected = [], []
        for candidate in selection:
            judgment = candidate.pop("_judgment")
            candidate.pop("_safety", None)
            candidate.update(**{name: judgment[name] for name in ("relation", "facts_match", "evidence", "conflicts", "subtitle_quotes")},
                usable=judgment["usable"], content_kind=judgment["content_kind"], visual_policy=VISUAL_POLICY_VERSION,
                match_policy=MATCH_POLICY_VERSION, visual_unit_id=segment.id, visual_checked=verify_visual,
                confidence=confidence(judgment.get("confidence")), reason=judgment["reason"],
                weak_match=judgment["relation"] != "direct" or not verify_visual)
            statistics["checked"] += 1
            if not candidate["usable"] or not self.range_allowed(candidate) or not match_allowed(candidate, segment.requirement):
                reason = "; ".join(judgment["conflicts"]) or judgment["reason"]
                category = "画面不可用" if not candidate["usable"] else judgment["relation"] if judgment["facts_match"] else "事实未对应"
                statistics["reasons"][category] = statistics["reasons"].get(category, 0) + 1
                record = {"scene_id": candidate["scene_id"], "source_start": candidate["source_start"], "source_end": candidate["source_end"],
                          "relation": judgment["relation"], "reason": reason}
                rejected.append(record)
                if len(statistics["examples"]) < 12:
                    statistics["examples"].append(record)
                continue
            anchor_frame = frame_index(judgment.get("frame_index", 1)) if verify_visual else 1
            candidate["anchor"] = candidate["source_start"] + (candidate["source_end"]-candidate["source_start"]) * (anchor_frame+.5)/3
            approved.append(candidate)
            statistics["approved"] += 1
        approved.sort(key=lambda item: (item["relation"] == "direct", item["confidence"]), reverse=True)
        write_json(cache_file, {"key": key, "version": MATCH_POLICY_VERSION, "visual_unit_id": segment.id,
                               "value": approved, "rejected": rejected})
        return approved
