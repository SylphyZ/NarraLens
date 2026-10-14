"""Import Markdown narration as editable plain text without model rewriting."""

import html
import re
from html.parser import HTMLParser


class _PlainHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in {"br", "p", "div", "li"}:
            self.parts.append("\n")
        if tag == "img":
            self.parts.append(dict(attrs).get("alt", ""))

    def handle_endtag(self, tag):
        if tag in {"p", "div", "li"}:
            self.parts.append("\n")


def markdown_to_narration(source: str) -> str:
    """Remove Markdown presentation markers; keep textual content for review."""
    source = source.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    fenced = False
    for line in source.splitlines():
        if re.match(r"^\s*(`{3,}|~{3,})", line):
            fenced = not fenced
            continue
        if fenced:
            lines.append(line)
            continue
        if re.fullmatch(r"\s*(?:[-*_]\s*){3,}", line):
            continue
        if re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*", line):
            continue
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"\s+#+\s*$", "", line)
        line = re.sub(r"^(\s*>\s*)+", "", line)
        line = re.sub(r"^\s*[-+*]\s+(?:\[[ xX]\]\s*)?", "", line)
        line = re.sub(r"^\s*\d+[.)]\s+", "", line)
        if line.strip().startswith("|") and line.strip().endswith("|"):
            line = "，".join(cell.strip() for cell in line.strip().strip("|").split("|") if cell.strip())
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"!?\[([^\]\n]*)\]\([^\n]*?\)", r"\1", text)
    text = re.sub(r"!?\[([^\]\n]*)\]\[[^\]\n]*\]", r"\1", text)
    for pattern in (r"\*\*(.+?)\*\*", r"__(.+?)__", r"~~(.+?)~~", r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"(?<!\w)_([^_\n]+)_(?!\w)", r"`([^`\n]+)`"):
        text = re.sub(pattern, r"\1", text)
    text = re.sub(r"\\([\\`*_{}\[\]()#+.!>-])", r"\1", text)
    if re.search(r"</?[A-Za-z][^>]*>", text):
        parser = _PlainHTML()
        parser.feed(text)
        text = "".join(parser.parts)
    else:
        text = html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

