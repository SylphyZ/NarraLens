"""Local lexical/semantic-label retrieval followed by bounded model review."""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from app.services.library.frame_extractor import extract_frames
from app.services.library.runner import check_cancel
from app.services.library.store import library_ids, stable_hash, write_json, read_json
from app.services.subtitle_segmentation import _tokenizer
from app.services.library.validation import confidence, frame_index, cached_result, ModelFormatError
from app.services.timeline_planner import VISUAL_POLICY_VERSION, natural_candidates, source_shot_key, unused_source


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


class MaterialRetriever:
    def __init__(self, store, library_id):
        self.store = store
        self.library_ids = sorted(library_ids(library_id))
        self.library_names = {item["id"]: item["name"] for item in store.list_libraries()}
        self.scenes = store.scenes(self.library_ids)
        if not self.scenes:
            raise ValueError("素材库尚无可用的成功场景，请先建立或继续分析素材库。")
        versions = [(scene["id"], scene["version"], scene["visual"]) for scene in self.scenes]
        # Preserve single-library cache keys; a changed multi-library selection invalidates selection only.
        self.version = stable_hash(versions if len(self.library_ids) == 1 else ("multi-library-v1", self.library_ids, versions))
        self.documents = []
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
        episode_summaries = {}
        for index, scene in enumerate(self.scenes):
            if scene["asset_id"] not in episode_summaries:
                episode_summaries[scene["asset_id"]] = store.cache_get(scene["library_id"], "episode-summary:" + scene["asset_id"]) or {}
            episode_summary = episode_summaries[scene["asset_id"]]
            description = json.dumps({"analysis": scene["analysis"], "visual": scene["visual"], "episode_characters": episode_summary.get("characters", [])}, ensure_ascii=False)
            terms = Counter(self.tokens(description + " " + scene["transcript"]))
            self.documents.append((description, terms))
            for term in terms:
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
        query = " ".join(str(requirement.get(key, "")) for key in ("claim", "characters", "themes", "events", "queries"))
        tokens = set(self.tokens(query))
        # Continue into less related transition footage after the matching pool is exhausted.
        pool = set(range(len(self.scenes)))
        scored = []
        for index in pool:
            scene = self.scenes[index]
            if scene["id"] in excluded or scene["end"] - scene["start"] < 2.24:
                continue
            if scene["id"] in self.rejected_scenes:
                continue
            counts = self.documents[index][1]
            score = sum(math.log(1 + len(self.scenes) / (1 + len(self.inverted.get(term, ())))) * min(3, counts.get(term, 0)) for term in tokens)
            score /= math.sqrt(max(1, sum(counts.values())))
            scored.append((score, index))
        scored.sort(reverse=True)
        results = []
        for score, index in scored[:limit]:
            scene = self.scenes[index]
            if scene["end"] - scene["start"] < .5:
                continue
            results.append({"scene_id": scene["id"], "asset_id": scene["asset_id"], "video_path": scene["path"],
                            "library_id": scene["library_id"], "library_name": self.library_names.get(scene["library_id"], ""),
                            "source_start": scene["start"], "source_end": scene["end"], "episode": scene["episode"],
                            "description": self.documents[index][0][:750], "score": score,
                            "frame_paths": scene["frame_paths"], "anchor": (scene["start"] + scene["end"]) / 2})
        return results

    def select(self, segment, overview, gateway, directory, used=(), verify_visual=True, stop=None, *, candidates, scene_boundaries, assets):
        """Rank one unseen batch, then approve actual natural-shot ranges, including on cache hits."""
        key = stable_hash((VISUAL_POLICY_VERSION, self.version, segment.text, segment.requirement,
                           gateway.signature("text"), [item["scene_id"] for item in candidates]))
        cache_file = Path(directory) / "candidates" / f"{segment.id}-{key[:20]}.json"
        try:
            cached = read_json(cache_file, default={})
        except (OSError, ValueError):
            cached = {}
        ranked = cached_result(cached.get("ranking"), {"clips": list}) if isinstance(cached, dict) and cached.get("key") == key else None
        check_cancel(stop)
        gateway.context.update(stage="选择候选画面", segment=segment.id)
        public = [{key: item[key] for key in ("scene_id", "episode", "description")} for item in candidates]
        if len(self.library_ids) > 1:
            for item, candidate in zip(public, candidates):
                item["library_name"] = candidate["library_name"]
        if ranked is None:
            ranked = gateway.json(
                "选择能服务于本段观点的画面，证据类必须实际支持观点。可以返回空列表，不要强行匹配。"
                "只引用候选 scene_id。返回 {clips:[{scene_id,confidence:0至1,reason,role}],gaps:[素材不足]}。最多选择6个。"
                f"\n全文摘要：{overview[:1600]}\n文案：{segment.text}\n画面需求：{json.dumps(segment.requirement, ensure_ascii=False)[:2500]}"
                f"\n候选：{json.dumps(public, ensure_ascii=False)}", max_output=2200, required={"clips": list},
                request_metadata={"candidates": self._request_sources(candidates)},
            )
            write_json(cache_file, {"key": key, "ranking": ranked})
        by_id = {item["scene_id"]: item for item in candidates}
        selection = []
        for item in ranked.get("clips", [])[:6]:
            if not isinstance(item, dict) or item.get("scene_id") not in by_id or any(row["scene_id"] == item["scene_id"] for row in selection):
                continue
            score = confidence(item.get("confidence"))
            selection.append({**by_id[item["scene_id"]], "confidence": score, "reason": str(item.get("reason", ""))[:500], "weak_match": score < .6})
        selected_ids = {item["scene_id"] for item in selection}
        selection.extend({**item, "confidence": 0, "reason": "本批未被语义评选选中，仅作为未用的过渡画面候选。", "weak_match": True}
                         for item in candidates if item["scene_id"] not in selected_ids)
        natural = natural_candidates(selection, scene_boundaries, assets)
        # Overlapping windows may describe the same shot. Prefer its longest usable window.
        unique = {}
        for item in natural:
            if unused_source(item, used):
                identity = source_shot_key(item)
                previous = unique.get(identity)
                if previous is None or (item["source_end"]-item["source_start"], item["confidence"]) > \
                        (previous["source_end"]-previous["source_start"], previous["confidence"]):
                    unique[identity] = item
        pending, selection = [], []
        for candidate in unique.values():
            check_cancel(stop)
            self._load_exclusions(candidate, gateway)
            if not self.range_allowed(candidate):
                continue
            safety_key = "visual-usability:" + stable_hash((VISUAL_POLICY_VERSION, candidate["fingerprint"],
                candidate["source_start"], candidate["source_end"], gateway.signature("vision")))
            review_key = "visual-relevance:" + stable_hash((safety_key, segment.text, segment.requirement))
            candidate.update(_safety_key=safety_key, _review_key=review_key, _review_id=stable_hash(safety_key)[:24])
            safety = self.store.cache_get(candidate["library_id"], safety_key)
            judgment = self.store.cache_get(candidate["library_id"], review_key) if verify_visual else safety
            try:
                if safety is not None:
                    _review_result({"clips": [safety]}, {candidate["_review_id"]})
                    if not safety["usable"]:
                        self._reject(candidate, safety, gateway)
                        continue
                if judgment is not None:
                    _review_result({"clips": [judgment]}, {candidate["_review_id"]})
            except (ModelFormatError, TypeError, KeyError):
                judgment = None
            if judgment is None:
                pending.append(candidate)
            else:
                candidate["_judgment"] = judgment
                selection.append(candidate)
        for offset in range(0, len(pending), 6):
            inspect = [candidate for candidate in pending[offset:offset + 6] if self.range_allowed(candidate)]
            if not inspect:
                continue
            images = []
            for candidate in inspect:
                check_cancel(stop)
                output_dir = self.store.directory(candidate["library_id"]) / "frames" / candidate["asset_id"] / candidate["_review_id"]
                # Window thumbnails cannot approve a crop in a mixed story/credits window.
                candidate["frame_paths"] = extract_frames(candidate["video_path"], candidate["source_start"], candidate["source_end"], 3, output_dir)
                images.extend(candidate["frame_paths"])
            check_cancel(stop)
            gateway.context["stage"] = "检查实际镜头画面"
            verified = gateway.json(
                "每个候选是一个实际裁剪范围，按顺序各附3张图。先独立判断画面能否用于解说剪辑："
                "任一图出现滚动职员表、演职员名单或画面上的制作/声优等演职员字幕，usable必须为false；"
                "混合剧情与名单、无法确认时也为false。普通对白字幕、歌词、剧情中的文字不因此禁用。"
                "content_kind为story/credits/mixed/unknown之一，只有明确的story可usable=true。"
                "confidence仅表示对当前文案的相关性，相关性高不能放行名单画面。未知人物不要猜身份。"
                "逐一返回所有候选且每个仅一次：{clips:[{scene_id,usable:true或false,content_kind,reason,confidence:0至1,frame_index:0至2}]}。"
                + (f"\n文案：{segment.text}\n需求：{json.dumps(segment.requirement, ensure_ascii=False)[:1800]}" if verify_visual else "\n本次只判断画面可用性，无需判断文案相关性。")
                + f"\n候选顺序：{json.dumps([{'scene_id': row['_review_id'], 'source_start': row['source_start'], 'source_end': row['source_end'], 'window_description': row['description'][:300]} for row in inspect], ensure_ascii=False)}",
                images=images, max_output=2600, required={"clips": list},
                result_validator=lambda value: _review_result(value, {item["_review_id"] for item in inspect}),
                request_metadata={"candidates": self._request_sources(inspect, with_images=True),
                                  "selected_scene_ids": [candidate["scene_id"] for candidate in inspect]},
            )
            judgments = {row.get("scene_id"): row for row in verified.get("clips", []) if isinstance(row, dict)}
            for candidate in inspect:
                judgment = judgments[candidate["_review_id"]]
                self.store.cache_put(candidate["library_id"], candidate["_safety_key"], judgment)
                if verify_visual:
                    self.store.cache_put(candidate["library_id"], candidate["_review_key"], judgment)
                if not judgment["usable"]:
                    self._reject(candidate, judgment, gateway)
                candidate["_judgment"] = judgment
                selection.append(candidate)
        approved = []
        for candidate in selection:
            judgment = candidate.pop("_judgment")
            if not judgment["usable"] or not self.range_allowed(candidate):
                continue
            score = confidence(judgment.get("confidence")) if judgment.get("frame_index_valid", False) else 0.0
            anchor_frame = frame_index(judgment.get("frame_index", 1))
            candidate.update(usable=True, content_kind="story", visual_policy=VISUAL_POLICY_VERSION,
                confidence=min(candidate["confidence"], score) if verify_visual else candidate["confidence"],
                weak_match=candidate["weak_match"] or not verify_visual or score < .6,
                reason=candidate["reason"] + "；画面检查：" + judgment["reason"][:300],
                anchor=candidate["source_start"] + (candidate["source_end"]-candidate["source_start"]) * (anchor_frame+.5)/3)
            approved.append(candidate)
        approved.sort(key=lambda item: item["confidence"], reverse=True)
        # Preserve a compact inspection record; never restore unchecked source ranges from it.
        write_json(cache_file, {"key": key, "ranking": ranked, "value": approved, "gaps": ranked.get("gaps", [])})
        return approved
