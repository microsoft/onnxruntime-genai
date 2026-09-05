# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Whisper timestamp-token parsing and result writers."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

TIMESTAMP_BEGIN = 50364
SECONDS_PER_TIMESTAMP = 0.02


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    tokens: list[int]
    words: list[dict] | None = None


def segments_from_tokens(tokens, decode, offset: float = 0.0) -> list[Segment]:
    """Parse Whisper timestamp tokens into segments.

    If timestamps are disabled, the complete decoded sequence is returned as one
    segment. This mirrors Whisper's timestamp-pair segmentation without requiring
    a second decoder pass.
    """
    tokens = list(tokens)
    timestamps = [index for index, token in enumerate(tokens) if token >= TIMESTAMP_BEGIN]
    if not timestamps:
        return [Segment(0, offset, offset, decode(tokens).strip(), tokens)]

    result = []
    previous = 0
    for index in range(1, len(tokens)):
        if tokens[index - 1] >= TIMESTAMP_BEGIN and tokens[index] >= TIMESTAMP_BEGIN:
            segment_tokens = tokens[previous:index]
            text_tokens = [token for token in segment_tokens if token < TIMESTAMP_BEGIN]
            if text_tokens:
                result.append(
                    Segment(
                        len(result),
                        offset + (tokens[previous] - TIMESTAMP_BEGIN) * SECONDS_PER_TIMESTAMP,
                        offset + (tokens[index - 1] - TIMESTAMP_BEGIN) * SECONDS_PER_TIMESTAMP,
                        decode(text_tokens).strip(),
                        segment_tokens,
                    )
                )
            previous = index

    if previous < len(tokens) - 1:
        text_tokens = [token for token in tokens[previous:] if token < TIMESTAMP_BEGIN]
        if text_tokens:
            start = tokens[previous] if tokens[previous] >= TIMESTAMP_BEGIN else TIMESTAMP_BEGIN
            end = tokens[timestamps[-1]]
            result.append(
                Segment(
                    len(result),
                    offset + (start - TIMESTAMP_BEGIN) * SECONDS_PER_TIMESTAMP,
                    offset + (end - TIMESTAMP_BEGIN) * SECONDS_PER_TIMESTAMP,
                    decode(text_tokens).strip(),
                    tokens[previous:],
                )
            )
    return result


def _timestamp(seconds: float, separator: str) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}{separator}{milliseconds:03}"


def write_result(segments: list[Segment], output_dir: str, stem: str, formats: list[str]) -> None:
    """Write OpenAI Whisper-compatible txt, json, jsonl, srt, tsv, and vtt output."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for output_format in formats:
        path = directory / f"{stem}.{output_format}"
        if output_format == "txt":
            path.write_text("\n".join(segment.text for segment in segments) + "\n", encoding="utf-8")
        elif output_format == "json":
            path.write_text(json.dumps({"text": " ".join(s.text for s in segments), "segments": [asdict(s) for s in segments]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        elif output_format == "jsonl":
            path.write_text("".join(json.dumps(asdict(segment), ensure_ascii=False) + "\n" for segment in segments), encoding="utf-8")
        elif output_format == "tsv":
            path.write_text("start\tend\ttext\n" + "".join(f"{round(s.start * 1000)}\t{round(s.end * 1000)}\t{s.text.replace(chr(9), ' ')}\n" for s in segments), encoding="utf-8")
        elif output_format in {"srt", "vtt"}:
            separator = "," if output_format == "srt" else "."
            lines = ["WEBVTT", ""] if output_format == "vtt" else []
            for index, segment in enumerate(segments, 1):
                if output_format == "srt":
                    lines.append(str(index))
                lines.extend([f"{_timestamp(segment.start, separator)} --> {_timestamp(segment.end, separator)}", segment.text, ""])
            path.write_text("\n".join(lines), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
