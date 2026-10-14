"""Local lexical/semantic-label retrieval followed by bounded model review."""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from app.services.library.frame_extractor import extract_frames
from app.services.library.runner import check_cancel
from app.services.library.store import library_ids, stable_hash, write_json
from app.services.subtitle_segmentation import _tokenizer
from app.services.library.validation import confidence, frame_index


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
            if with_images:
                frames = candidate["frame_paths"][:3]
                source.update(frame_paths=frames, image_indices=list(range(offset, offset + len(frames))))
                offset += len(frames)
            sources.append(source)
        return sources

    def retrieve(self, requirement, limit=12, used=None):
        used = used or {}
        query = " ".join(str(requirement.get(key, "")) for key in ("claim", "characters", "themes", "events", "queries"))
        tokens = set(self.tokens(query))
        matching = set()
        for term in tokens:
            matching.update(self.inverted.get(term, set()))
        pool = matching or set(range(len(self.scenes)))
        scored = []
        for index in pool:
            scene = self.scenes[index]
            counts = self.documents[index][1]
            score = sum(math.log(1 + len(self.scenes) / (1 + len(self.inverted.get(term, ())))) * min(3, counts.get(term, 0)) for term in tokens)
            score /= math.sqrt(max(1, sum(counts.values())))
            score -= used.get(scene["id"], 0) * .3
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

    def select(self, segment, overview, gateway, directory, used=None, verify_visual=True, stop=None):
        cache_file = Path(directory) / "candidates" / f"{segment.id}.json"
        key = stable_hash(("selection-v1", self.version, segment.requirement, gateway.signature("text"), gateway.signature("vision"), verify_visual, used or {}))
        if cache_file.is_file():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cached = {}
            if not isinstance(cached, dict):
                cached = {}
            if cached.get("key") == key:
                current_scenes = {scene["id"]: scene for scene in self.scenes}
                restored = []
                for candidate in cached.get("value", []) if isinstance(cached.get("value"), list) else []:
                    if not isinstance(candidate, dict):
                        continue
                    if not isinstance(candidate.get("scene_id"), str):
                        continue
                    scene = current_scenes.get(candidate.get("scene_id"))
                    if scene:
                        score = confidence(candidate.get("confidence"))
                        try:
                            anchor = float(candidate.get("anchor"))
                        except (ValueError, TypeError, OverflowError):
                            anchor = (scene["start"] + scene["end"]) / 2
                        if not math.isfinite(anchor) or not scene["start"] <= anchor <= scene["end"]:
                            anchor = (scene["start"] + scene["end"]) / 2
                        restored.append({**candidate, "asset_id": scene["asset_id"], "source_start": scene["start"], "source_end": scene["end"],
                                         "library_id": scene["library_id"], "library_name": self.library_names.get(scene["library_id"], ""),
                                         "episode": scene["episode"],
                                         "anchor": anchor,
                                         "video_path": scene["path"], "frame_paths": scene["frame_paths"], "confidence": score,
                                         "weak_match": bool(candidate.get("weak_match", True) or score < .6),
                                         "reason": str(candidate.get("reason") or "")[:500]})
                if restored:
                    return restored
        candidates = self.retrieve(segment.requirement, used=used)
        if not candidates:
            raise ValueError("没有可用的候选视频范围。")
        check_cancel(stop)
        gateway.context.update(stage="选择候选画面", segment=segment.id)
        public = [{key: item[key] for key in ("scene_id", "episode", "description")} for item in candidates]
        if len(self.library_ids) > 1:
            for item, candidate in zip(public, candidates):
                item["library_name"] = candidate["library_name"]
        ranked = gateway.json(
            "选择能服务于本段观点的画面，证据类必须实际支持观点。可以返回空列表，不要强行匹配。"
            "只引用候选 scene_id。返回 {clips:[{scene_id,confidence:0至1,reason,role}],gaps:[素材不足]}。最多选择6个。"
            f"\n全文摘要：{overview[:1600]}\n文案：{segment.text}\n画面需求：{json.dumps(segment.requirement, ensure_ascii=False)[:2500]}"
            f"\n候选：{json.dumps(public, ensure_ascii=False)}", max_output=2200, required={"clips": list},
            request_metadata={"candidates": self._request_sources(candidates)},
        )
        by_id = {item["scene_id"]: item for item in candidates}
        selection = []
        for item in ranked.get("clips", [])[:6]:
            if not isinstance(item, dict) or item.get("scene_id") not in by_id or any(row["scene_id"] == item["scene_id"] for row in selection):
                continue
            score = confidence(item.get("confidence"))
            selection.append({**by_id[item["scene_id"]], "confidence": score, "reason": str(item.get("reason", ""))[:500], "weak_match": score < .6})
        if not selection:
            selection = [{**item, "confidence": 0, "reason": "未找到明确支持观点的素材，使用候选过渡画面。", "weak_match": True} for item in candidates[:4]]
        if verify_visual:
            inspect = selection[:3]
            images = []
            for candidate in inspect:
                check_cancel(stop)
                if len(candidate["frame_paths"]) < 3 or any(not Path(path).is_file() for path in candidate["frame_paths"][:3]):
                    output_dir = self.store.directory(candidate["library_id"]) / "frames" / candidate["asset_id"] / stable_hash((candidate["scene_id"], "verify"))[:20]
                    candidate["frame_paths"] = extract_frames(candidate["video_path"], candidate["source_start"], candidate["source_end"], 3, output_dir)
                images.extend(candidate["frame_paths"][:3])
            check_cancel(stop)
            gateway.context["stage"] = "确认候选画面"
            verified = gateway.json(
                "每个候选对应连续的三张图，共按候选顺序排列。判断画面是否支持文案，未知人物不要猜身份。"
                "返回 {clips:[{scene_id,confidence:0至1,reason,frame_index:0至2}]}。只评价给定候选。"
                f"\n文案：{segment.text}\n需求：{json.dumps(segment.requirement, ensure_ascii=False)[:1800]}"
                f"\n候选顺序：{json.dumps([{'scene_id': row['scene_id'], 'description': row['description'][:300]} for row in inspect], ensure_ascii=False)}",
                images=images, max_output=1800, required={"clips": list},
                request_metadata={"candidates": self._request_sources(inspect, with_images=True),
                                  "selected_scene_ids": [candidate["scene_id"] for candidate in selection]},
            )
            judgments = {row.get("scene_id"): row for row in verified.get("clips", []) if isinstance(row, dict)}
            for candidate in inspect:
                judgment = judgments.get(candidate["scene_id"], {})
                score = confidence(judgment.get("confidence")) if judgment.get("frame_index_valid", False) else 0.0
                anchor_frame = frame_index(judgment.get("frame_index", 1))
                candidate.update({"confidence": min(candidate["confidence"], score),
                                  "weak_match": candidate["weak_match"] or score < .6,
                                  "reason": candidate["reason"] + "；画面确认：" + str(judgment.get("reason", "未返回明确判断"))[:300],
                                  "anchor": candidate["source_start"] + (candidate["source_end"] - candidate["source_start"]) * (anchor_frame + .5) / 3})
            for candidate in selection[3:]:
                candidate["weak_match"] = True
                candidate["reason"] += "；未做本段的视觉确认。"
        else:
            for candidate in selection:
                candidate["weak_match"] = True
                candidate["reason"] += "；本次关闭了视觉确认。"
        selection.sort(key=lambda item: item["confidence"], reverse=True)
        write_json(cache_file, {"key": key, "value": selection, "gaps": ranked.get("gaps", [])})
        return selection
