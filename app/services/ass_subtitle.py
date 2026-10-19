"""Select Chinese dialogue before ASS/SSA styling and layout are discarded."""

import re
from bisect import bisect_left
from dataclasses import dataclass, field


_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]")
_OTHER_SCRIPT = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f\uac00-\ud7af\u1100-\u11ff\u0400-\u052f\u0600-\u06ff\u0e00-\u0e7f]")
_LATIN = re.compile(r"[A-Za-z\u00c0-\u024f]")
_CHINESE_STYLE = re.compile(r"(?:^|[^a-z])(?:chs|cht|chn|cn|zh|zho|chi|chinese|sc|tc)(?:$|[^a-z])", re.IGNORECASE)
_FOREIGN_STYLE = re.compile(r"(?:^|[^a-z])(?:jp|jpn|ja|japanese|kr|kor|ko|korean|en|eng|english)(?:$|[^a-z])|日文|日语|日語|韩文|韓文|英文", re.IGNORECASE)
ASS_SELECTION_VERSION = "chinese-body-v2"
_DEFAULT_FIELDS = ("layer", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text")


@dataclass(frozen=True)
class AssEvent:
    line_index: int
    fields: tuple[str, ...]
    text_index: int
    style: str
    start: str
    end: str
    text: str


@dataclass(frozen=True)
class ChineseAssResult:
    text: str
    original: tuple[AssEvent, ...]
    selected: tuple[AssEvent, ...]
    removed_events: int
    removed_lines: int
    duplicate_events: int
    uncertain_events: int
    report: dict = field(default_factory=dict)


def ass_role(style: str) -> str:
    """Role is independent of language; a Chinese OP is not dialogue."""
    if re.search(r"(?:^|[^a-z])(?:op|ed|opening|ending|lyrics|karaoke)(?:$|[^a-z])|歌词|歌詞|片头曲|片尾曲", style, re.I):
        return "lyrics"
    if re.search(r"(?:^|[^a-z])(?:note|notes|annotation|comment)(?:$|[^a-z])|注释|註釋|注釈|译注|譯注|註解", style, re.I):
        return "annotation"
    if re.search(r"(?:^|[^a-z])(?:sign|screen|title)(?:$|[^a-z])|屏幕字|標題|标题|招牌", style, re.I):
        return "screen"
    return "body"


def plain_ass_lines(text: str) -> list[str]:
    """Remove override tags and vector drawings, keeping explicit line breaks."""
    parts = []
    drawing = False
    for part in re.split(r"(\{[^}]*\})", text):
        if part.startswith("{") and part.endswith("}"):
            mode = re.search(r"\\p(\d+)\b", part)
            if mode:
                drawing = int(mode.group(1)) != 0
        elif not drawing:
            parts.append(part)
    visible = "".join(parts).replace(r"\N", "\n").replace(r"\n", "\n").replace(r"\h", " ")
    return [line.strip() for line in visible.splitlines() if line.strip()]


def parse_ass_events(text: str) -> list[AssEvent]:
    events = []
    in_events = False
    columns = _DEFAULT_FIELDS
    for line_index, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_events = stripped.casefold() == "[events]"
            continue
        if not in_events or ":" not in stripped:
            continue
        kind, body = stripped.split(":", 1)
        if kind.casefold() == "format":
            columns = tuple(field.strip().casefold() for field in body.split(","))
            continue
        if kind.casefold() != "dialogue":
            continue
        if not {"start", "end", "style", "text"}.issubset(columns):
            raise ValueError("ASS/SSA Events Format must contain Start, End, Style and Text.")
        text_index = columns.index("text")
        before = body.lstrip().split(",", text_index)
        after_count = len(columns) - text_index - 1
        after = before[-1].rsplit(",", after_count) if after_count else [before[-1]]
        fields = tuple(before[:-1] + after)
        if len(fields) != len(columns):
            raise ValueError(f"Invalid ASS/SSA dialogue at line {line_index + 1}.")
        events.append(AssEvent(
            line_index, fields, text_index, fields[columns.index("style")].strip(),
            fields[columns.index("start")].strip(), fields[columns.index("end")].strip(), fields[text_index],
        ))
    if not events:
        raise ValueError("ASS/SSA subtitle has no dialogue events.")
    return events


def is_chinese_style(style: str) -> bool:
    return bool(_CHINESE_STYLE.search(style) or any(word in style for word in ("中文", "简体", "繁体", "简中", "繁中", "簡體", "繁體", "簡中")))


def _chinese_candidate(line: str) -> bool:
    # Han characters alone do not prove a language; explicit styles take priority.
    return bool(_HAN.search(line) and not _OTHER_SCRIPT.search(line))


def _time_ms(value: str) -> int:
    hours, minutes, seconds = value.split(":")
    return round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)


def _parallel(left: AssEvent, right: AssEvent) -> bool:
    start, end = _time_ms(left.start), _time_ms(left.end)
    other_start, other_end = _time_ms(right.start), _time_ms(right.end)
    shortest = min(end - start, other_end - other_start)
    overlap = min(end, other_end) - max(start, other_start)
    return shortest > 0 and overlap / shortest >= 0.7


def select_chinese_ass(text: str, styles: list[str] | None = None, line_mode: str = "auto") -> ChineseAssResult:
    """Use explicit styles/line selection, otherwise conservatively remove parallel translations."""
    if line_mode not in {"auto", "first", "last"}:
        raise ValueError("Unsupported Chinese subtitle line selection.")
    events = parse_ass_events(text)
    available = {event.style for event in events}
    chosen = set(styles or [])
    if chosen and not chosen.intersection(available):
        raise ValueError("Selected Chinese styles do not exist in this subtitle file.")
    # Language-labelled lyrics must never turn into a file-wide style whitelist.
    explicit = bool(chosen)
    candidates = [event for event in events if any(_chinese_candidate(line) for line in plain_ass_lines(event.text))]
    candidates.sort(key=lambda event: _time_ms(event.start))
    candidate_starts = [_time_ms(event.start) for event in candidates]
    max_candidate_duration = max((_time_ms(event.end) - _time_ms(event.start) for event in candidates), default=0)
    source_lines = text.splitlines()
    replacements = {}
    selected = []
    removed_events = removed_lines = duplicates = uncertain = 0
    seen = set()
    decisions = []

    for event in events:
        lines = plain_ass_lines(event.text)
        role = ass_role(event.style)
        decision = {"line": event.line_index + 1, "style": event.style, "role": role,
                    "start_ms": _time_ms(event.start), "end_ms": _time_ms(event.end)}
        decisions.append(decision)
        if (chosen and event.style not in chosen) or not lines:
            decision.update(action="removed", reason="manual_style" if lines else "invisible_or_drawing")
            replacements[event.line_index] = None
            removed_events += 1
            continue
        chinese = [line for line in lines if _chinese_candidate(line)]
        if not explicit and _FOREIGN_STYLE.search(event.style) and not is_chinese_style(event.style):
            decision.update(action="removed", reason="explicit_foreign_language")
            replacements[event.line_index] = None
            removed_events += 1
            continue
        if not chosen and not chinese and any(_OTHER_SCRIPT.search(line) or _LATIN.search(line) for line in lines):
            low = bisect_left(candidate_starts, _time_ms(event.start) - max_candidate_duration)
            high = bisect_left(candidate_starts, _time_ms(event.end))
            if any(_parallel(event, candidate) for candidate in candidates[low:high]):
                decision.update(action="removed", reason="parallel_translation")
                replacements[event.line_index] = None
                removed_events += 1
                continue
            uncertain += 1  # Standalone foreign text may be a name or a line with no Chinese translation.
            decision["uncertain"] = "standalone_language"
        kept = list(lines)
        if line_mode != "auto":
            kept = [lines[0] if line_mode == "first" else lines[-1]]
        else:
            if chinese:
                # A separate foreign-script/Latin line beside Chinese is a translation candidate.
                kept = [line for line in lines if _chinese_candidate(line) or not (_OTHER_SCRIPT.search(line) or _LATIN.search(line))]
                if not explicit and not is_chinese_style(event.style):
                    uncertain += 1  # Japanese kanji-only text cannot be reliably distinguished.
                    decision["uncertain"] = "neutral_style_han_language"
        removed_lines += len(lines) - len(kept)
        kept = list(dict.fromkeys(kept))
        visible = r"\N".join(kept)
        identity = (_time_ms(event.start), _time_ms(event.end), visible, role)
        if identity in seen:
            decision.update(action="removed", reason="same_time_visible_text_duplicate")
            replacements[event.line_index] = None
            duplicates += 1
            continue
        seen.add(identity)
        decision.update(action="kept", reason="manual_selection" if explicit else "visible_language_selection",
                        removed_lines=len(lines)-len(kept), text=visible)
        fields = list(event.fields)
        fields[event.text_index] = visible
        replacements[event.line_index] = "Dialogue: " + ",".join(fields)
        selected.append(AssEvent(event.line_index, tuple(fields), event.text_index, event.style, event.start, event.end, visible))

    if not selected:
        raise ValueError("No subtitles remain after Chinese selection; check the selected styles and lines.")
    filtered = "\n".join(
        replacements.get(index, line) for index, line in enumerate(source_lines)
        if replacements.get(index, line) is not None
    )
    body = [d for d in decisions if d["role"] == "body" and d["action"] == "kept"]
    report = {"version": ASS_SELECTION_VERSION, "manual_styles": explicit, "styles": sorted(available),
              "body_events": len(body), "lyrics_events": sum(d["role"] == "lyrics" and d["action"] == "kept" for d in decisions),
              "uncertain_events": uncertain, "decisions": decisions}
    return ChineseAssResult(filtered, tuple(events), tuple(selected), removed_events, removed_lines, duplicates, uncertain, report)
