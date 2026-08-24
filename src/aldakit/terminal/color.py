"""Minimal Alda syntax spans for ANSI rendering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    style: str


def tokenize_alda(line: str) -> list[Span]:
    """Return display-only syntax spans for one line of Alda source."""
    spans: list[Span] = []
    index = 0
    while index < len(line):
        char = line[index]
        if char == "#":
            spans.append(Span(index, len(line), "comment"))
            break
        if char == "(":
            end = _balanced_end(line, index)
            spans.append(Span(index, end, "attribute"))
            index = end
            continue
        if char.isalpha():
            end = index + 1
            while end < len(line) and (line[end].isalnum() or line[end] == "-"):
                end += 1
            if end < len(line) and line[end] == ":":
                spans.append(Span(index, end + 1, "instrument"))
                index = end + 1
                continue
        if char in "abcdefg" and (index == 0 or not line[index - 1].isalnum()):
            end = index + 1
            while end < len(line) and line[end] in "+-_":
                end += 1
            spans.append(Span(index, end, "note"))
            duration = end
            while duration < len(line) and (line[duration].isdigit() or line[duration] == "."):
                duration += 1
            if duration > end:
                if line[duration:duration + 2] == "ms":
                    duration += 2
                elif duration < len(line) and line[duration] == "s":
                    duration += 1
                spans.append(Span(end, duration, "duration"))
                end = duration
            index = end
            continue
        if char == "r" and (index + 1 == len(line) or not line[index + 1].isalpha()):
            spans.append(Span(index, index + 1, "rest"))
            end = index + 1
            while end < len(line) and (line[end].isdigit() or line[end] == "."):
                end += 1
            if end > index + 1:
                spans.append(Span(index + 1, end, "duration"))
                index = end
                continue
        elif char == "o" and index + 1 < len(line) and line[index + 1].isdigit():
            end = index + 2
            while end < len(line) and line[end].isdigit():
                end += 1
            spans.append(Span(index, end, "octave"))
            index = end
            continue
        elif char in "<>|":
            spans.append(Span(index, index + 1, "octave" if char != "|" else "barline"))
        index += 1
    return spans


def _balanced_end(text: str, start: int) -> int:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index + 1
    return len(text)
