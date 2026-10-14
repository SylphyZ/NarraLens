"""Keep manuscript immutable; derive cached visual requirements and cue boundaries."""

import json
import re
from pathlib import Path

from loguru import logger

from app.models.timeline import NarrationSegment
from app.services.library.runner import check_cancel
from app.services.library.store import stable_hash, write_json
from app.services.library.validation import ModelFormatError, cached_result
from app.services.subtitle_segmentation import split_by_words, word_boundary_parts


def split_document(text, limit=420):
    if not text or not text.strip():
        raise ValueError("请输入完整文案。")
    result = []
    position = 0
    while position < len(text):
        end = min(len(text), position + limit)
        if end < len(text):
            boundaries = [match.end() for match in re.finditer(r"[。！？!?；;\n]", text[position:end])]
            if boundaries and boundaries[-1] >= limit // 3:
                end = position + boundaries[-1]
            else:
                # Keep the original bytes/characters while avoiding a Chinese word boundary.
                from app.services.subtitle_segmentation import _tokenizer
                size = 0
                for token in _tokenizer().cut(text[position:min(len(text), end + 32)]):
                    if size and size + len(token) > limit:
                        break
                    size += len(token)
                end = position + max(1, size)
        fragment = text[position:end]
        if fragment.strip():
            result.append(NarrationSegment(f"P{len(result)+1:04d}", position, end, fragment))
        position = end
    # Whitespace between paragraphs may be omitted from TTS, never substantive text.
    if len(result) > 1 and len(result[-1].text.strip()) < 40:
        last = result.pop()
        result[-1].text_end = last.text_end
        result[-1].text = text[result[-1].text_start:last.text_end]
    if re.sub(r"\s", "", "".join(item.text for item in result)) != re.sub(r"\s", "", text):
        raise ValueError("文案分段覆盖校验失败。")
    return result


def _cached_json(path, key, callback, required=None, validator=None):
    path = Path(path)
    if path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = {}
        if not isinstance(cached, dict):
            cached = {}
        if cached.get("key") == key:
            value = cached_result(cached.get("value"), required)
            if value is not None:
                try:
                    return validator(value) if validator else value
                except ModelFormatError:
                    pass
    value = callback()
    write_json(path, {"key": key, "value": value})
    return value


def plan_document(text, gateway, directory, stop=None, progress=None):
    directory = Path(directory)
    segments = split_document(text)
    model_version = gateway.signature("text")
    summaries = []
    for index, offset in enumerate(range(0, len(text), 6000)):
        check_cancel(stop)
        fragment = text[offset:offset + 6000]
        gateway.context.update(stage="提取文案论述", segment=f"outline-{index + 1}")
        key = stable_hash(("outline-v1", model_version, fragment))
        summary = _cached_json(directory / "planning" / f"outline-{index}.json", key, lambda: gateway.json(
            "提取深度解析文案的论述结构，不改写原文，不添加剧情。返回 {summary:不超过800字,arguments:[主要观点]}。\n文案：\n" + fragment,
            max_output=1600, required={"summary": str},
        ), required={"summary": str})
        summaries.append(str(summary.get("summary", ""))[:1600])
    level = 0
    while len(summaries) > 1:
        merged = []
        for index in range(0, len(summaries), 5):
            check_cancel(stop)
            chunk = summaries[index:index + 5]
            gateway.context.update(stage="合并文案论述", segment=f"merge-{level}-{index}")
            key = stable_hash(("outline-merge-v1", model_version, chunk))
            result = _cached_json(directory / "planning" / f"merge-{level}-{index}.json", key, lambda: gateway.json(
                "合并这些连续文案的论述摘要，保持观点的先后和前后联系，不补写剧情。返回 {summary:不超过1000字}。\n" + json.dumps(chunk, ensure_ascii=False), max_output=1600, required={"summary": str},
            ), required={"summary": str})
            merged.append(str(result.get("summary", ""))[:2000])
        summaries = merged
        level += 1
    overview = summaries[0]
    for index, segment in enumerate(segments):
        check_cancel(stop)
        if progress:
            progress(index / len(segments), f"分析文案段落 {index + 1}/{len(segments)}")
        previous = segments[index - 1].text[-200:] if index else ""
        following = segments[index + 1].text[:200] if index + 1 < len(segments) else ""
        gateway.context.update(stage="分析段落画面需求", segment=segment.id)
        key = stable_hash(("requirement-v1", model_version, overview, segment.text, previous, following))
        requirement = _cached_json(directory / "planning" / f"{segment.id}.json", key, lambda: gateway.json(
            "为以下文案段落制定画面需求，不改写、删减或生成待朗读文案。区分剧情证据、人物状态、跨集对照和过渡。"
            "返回 {claim,kind,characters:[字符串],themes:[字符串],events:[字符串],queries:[最多6条跨集检索描述],avoid:[易误用的情境]}。"
            f"\n全文论述：{overview}\n上文：{previous}\n本段原文：{segment.text}\n下文：{following}", max_output=1800, required={"claim": str, "queries": list},
        ), required={"claim": str, "queries": list})
        segment.requirement = requirement
    write_json(directory / "document_plan.json", {"text_hash": stable_hash(text), "overview": overview,
               "segments": [{"id": item.id, "text_start": item.text_start, "text_end": item.text_end, "text": item.text, "requirement": item.requirement} for item in segments]})
    return overview, segments


def _validate_subtitle_chunks(value, original, limit=48, boundaries=None):
    """Validate exact content/order, then recover boundaries from original text."""
    selected = None
    if isinstance(value, dict) and "break_after" in value and boundaries is not None:
        selected = value["break_after"]
        if (not isinstance(selected, list) or any(type(item) is not int for item in selected)
                or selected != sorted(set(selected)) or any(item < 1 or item > len(boundaries) for item in selected)):
            raise ModelFormatError(f"break_after 必须是递增、不重复的整数编号列表，范围1至{len(boundaries)}。")
        ends = selected if selected and selected[-1] == len(boundaries) else selected + [len(boundaries)]
        chunks, start = [], 0
        for end in ends:
            chunks.append("".join(boundaries[start:end]))
            start = end
        value = {"chunks": chunks}
    chunks = value.get("chunks") if isinstance(value, dict) else None
    if not isinstance(chunks, list) or not chunks or not all(isinstance(item, str) for item in chunks):
        raise ModelFormatError("字幕分段必须是非空字符串列表 chunks。")
    normalized = [re.sub(r"\s", "", item) for item in chunks]
    if any(not item or not re.search(r"\w", item) for item in normalized):
        raise ModelFormatError("字幕分段不能包含空条目或只有标点的条目。")
    positions = [index for index, character in enumerate(original) if not character.isspace()]
    canonical = "".join(original[index] for index in positions)
    if "".join(normalized) != canonical:
        raise ModelFormatError("字幕分段改动、遗漏或重复了原文。请逐字复制，保留标点和原顺序，只改变分段。")
    # Length is a local layout constraint; do not repeat a paid semantic request for it.
    fitted = []
    for chunk, plain in zip(chunks, normalized):
        fitted.extend(split_by_words(chunk, min(32, limit), max_chars=limit) if len(plain) > limit else [chunk])
    index = 0
    while index < len(fitted):
        if len(fitted) > 1 and len(re.sub(r"\s", "", fitted[index])) < 4:
            start = index - 1 if index else 0
            pair = fitted[start:start + 2]
            combined = "".join(pair)
            replacements = (split_by_words(combined, min(32, limit), max_chars=limit)
                            if len(re.sub(r"\s", "", combined)) > limit else [combined])
            if [re.sub(r"\s", "", item) for item in replacements] != [re.sub(r"\s", "", item) for item in pair]:
                fitted[start:start + 2] = replacements
                index = max(0, start - 1)
                continue
        index += 1
    fitted_plain = [re.sub(r"\s", "", item) for item in fitted]
    if "".join(fitted_plain) != canonical:
        raise ValueError("字幕长度校正未完整保留原文。")
    adjustments = value.get("length_adjustments", [])
    adjustments = adjustments if isinstance(adjustments, list) else []
    if fitted_plain != normalized:
        adjustments = [*adjustments, {"reason": "length", "before": chunks, "after": fitted}]
        logger.warning("字幕长度校正：{}条调整为{}条，原文完整保留。", len(chunks), len(fitted))
    normalized = fitted_plain
    restored, consumed, start = [], 0, 0
    for chunk in normalized:
        consumed += len(chunk)
        end = positions[consumed - 1] + 1
        restored.append(original[start:end].strip())
        start = end
    return {"chunks": restored, **({"length_adjustments": adjustments} if adjustments else {})}


def fit_single_line_units(units, options):
    """Fit display cues before TTS/alignment; keep the cached semantic plan intact."""
    if not options.get("subtitle_single_line", False):
        return units
    from app.services.generate_video import wrap_text
    width = 1080 if options.get("aspect") == "portrait" else 1920
    ratio = float(options.get("subtitle_width_ratio", .7))
    if not .5 <= ratio <= .95:
        raise ValueError("字幕单行宽度应在画面宽度的50%至95%之间。")
    result = {}
    for identifier, chunks in units.items():
        rows = []
        for text in chunks:
            wrapped, _ = wrap_text(" ".join(text.split()), width * ratio,
                                   font=options.get("font") or "Arial", fontsize=int(options.get("font_size", 72)),
                                   preserve_words=True)
            rows.extend(line.strip() for line in wrapped.splitlines() if line.strip())
        if "".join("".join(row.split()) for row in rows) != "".join("".join(row.split()) for row in chunks):
            raise ValueError("单行字幕细分没有完整保留原文。")
        result[identifier] = rows
    return result


def subtitle_text_ranges(segments, units):
    """Recover absolute original spans without altering the existing chunks API."""
    ranges = {}
    for segment in segments:
        positions = [i for i, char in enumerate(segment.text) if not char.isspace()]
        consumed, rows = 0, []
        for index, text in enumerate(units[segment.id]):
            size = sum(not char.isspace() for char in text)
            if not size or consumed + size > len(positions):
                raise ValueError("字幕原文位置无效。")
            start = positions[consumed]
            consumed += size
            end = positions[consumed - 1] + 1
            if "".join(segment.text[start:end].split()) != "".join(text.split()):
                raise ValueError("字幕正文与原文位置不一致。")
            rows.append({"index": index, "text_start": segment.text_start + start,
                         "text_end": segment.text_start + end})
        if consumed != len(positions):
            raise ValueError("字幕原文位置没有完整覆盖段落。")
        ranges[segment.id] = rows
    return ranges


def plan_subtitles(segments, gateway, directory, stop=None, progress=None):
    """Use the configured efficient text model for faithful semantic cue splitting."""
    directory = Path(directory)
    signature = gateway.signature("text_fast")
    units = {}
    adjustments = {}
    for index, segment in enumerate(segments):
        check_cancel(stop)
        if progress:
            progress(index / len(segments), f"高效率模型字幕分段 {index+1}/{len(segments)}")
        previous = segments[index-1].text[-120:] if index else ""
        following = segments[index+1].text[:120] if index+1 < len(segments) else ""
        boundaries = word_boundary_parts(segment.text)
        if not boundaries:
            raise ValueError(f"文案段落{segment.id}没有可朗读文字。")
        # The validated cache payload is still {chunks: [...]}; reuse successful plans.
        key = stable_hash(("subtitle-split-v1", signature, segment.text, previous, following, 4, 48))
        gateway.context.update(stage="字幕语义分段", segment=segment.id)
        validator = lambda value: _validate_subtitle_chunks(value, segment.text, boundaries=boundaries)
        rows, count = [], 0
        for number, fragment in enumerate(boundaries, 1):
            count += len(re.sub(r"\s", "", fragment))
            rows.append([number, count, fragment])
        prompt = (
            "为已有影视深度解析文案选择自然、易读的字幕边界，只选择编号，不复制或改写原文。"
            "subtitle_words中每项为[id,chars,text]，即编号、累计字数、原文词组；每个编号只是可选边界，不是一条字幕。"
            "优先完整语义、人物名称、词组和标点停顿；每条尽量16至32字，最多48个非空白字符。"
            "多条分段时每条至少4字，短碎片并入相邻条目。单个不可拆分的词或书名超过48字时独立成条。"
            "chars是从本段开头累计的非空白字符数，相邻断点的chars之差就是一条字幕的字数；第一条从0开始。"
            "break_after列出在哪些编号后结束字幕，必须递增、不重复；程序自动补上本段结尾，保证全文按顺序保留。"
            "同时处理待分段内容中的所有段落。参考前后文仅供理解，不参与分段。尽量在段落处断开。"
            "只返回JSON对象{\"break_after\":[整数编号,整数编号]}，不要返回字幕文字、解释或Markdown。"
            "\n输入数据：\n" + json.dumps({"previous_context": previous, "subtitle_words": rows,
                                              "following_context": following}, ensure_ascii=False)
        )
        result = _cached_json(directory / "planning" / f"subtitle-{segment.id}.json", key,
            lambda: gateway.json(prompt, model_kind="text_fast", max_output=1200, required={"break_after": list},
                                 result_validator=validator), required={"chunks": list}, validator=validator)
        units[segment.id] = result["chunks"]
        if result.get("length_adjustments"):
            adjustments[segment.id] = result["length_adjustments"]
    write_json(directory / "subtitle_plan.json", {"version": 4, "model_signature": signature, "units": units,
                                                "ranges": subtitle_text_ranges(segments, units),
                                                "length_adjustments": adjustments})
    return units
