# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Whisper cross-attention alignment utilities."""

from dataclasses import dataclass

import numpy as np


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


def _median_filter(values: np.ndarray, width: int = 7) -> np.ndarray:
    if values.shape[-1] <= width // 2:
        return values
    padded = np.pad(values, ((0, 0), (0, 0), (width // 2, width // 2)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=-1)
    return np.median(windows, axis=-1)


def _dtw(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    text_length, frame_length = cost.shape
    cumulative = np.full((text_length + 1, frame_length + 1), np.inf, dtype=np.float32)
    trace = np.zeros((text_length + 1, frame_length + 1), dtype=np.uint8)
    cumulative[0, 0] = 0

    for text_index in range(1, text_length + 1):
        for frame_index in range(1, frame_length + 1):
            candidates = (
                cumulative[text_index - 1, frame_index - 1],
                cumulative[text_index - 1, frame_index],
                cumulative[text_index, frame_index - 1],
            )
            step = int(np.argmin(candidates))
            cumulative[text_index, frame_index] = cost[text_index - 1, frame_index - 1] + candidates[step]
            trace[text_index, frame_index] = step

    text_indices, frame_indices = [], []
    text_index, frame_index = text_length, frame_length
    while text_index or frame_index:
        text_indices.append(text_index - 1)
        frame_indices.append(frame_index - 1)
        step = trace[text_index, frame_index]
        if step == 0:
            text_index -= 1
            frame_index -= 1
        elif step == 1:
            text_index -= 1
        else:
            frame_index -= 1
    return np.asarray(text_indices[::-1]), np.asarray(frame_indices[::-1])


def word_timestamps(token_text: list[str], cross_qk: np.ndarray, seconds_per_frame: float = 0.02) -> list[WordTimestamp]:
    """Extract word timings with Whisper's cross-attention DTW alignment.

    ``cross_qk`` is the CPU copy returned by ``Generator.get_output("cross_qk")``
    for one generated sequence, with shape ``[alignment_heads, tokens, frames]``.
    """
    if not token_text:
        return []
    if cross_qk.ndim != 3 or cross_qk.shape[1] < len(token_text):
        raise ValueError("cross_qk must have shape [alignment_heads, tokens, frames].")

    weights = cross_qk[:, -len(token_text) :, :].astype(np.float32, copy=False)
    weights -= weights.max(axis=-1, keepdims=True)
    weights = np.exp(weights)
    weights /= weights.sum(axis=-1, keepdims=True)
    std = weights.std(axis=-2, keepdims=True)
    weights = (weights - weights.mean(axis=-2, keepdims=True)) / np.maximum(std, 1e-10)
    matrix = _median_filter(weights).mean(axis=0)
    text_indices, frame_indices = _dtw(-matrix)

    token_times = np.zeros(len(token_text), dtype=np.float32)
    for token_index in range(len(token_text)):
        frames = frame_indices[text_indices == token_index]
        token_times[token_index] = (frames[-1] if len(frames) else 0) * seconds_per_frame

    words: list[WordTimestamp] = []
    for token_index, piece in enumerate(token_text):
        if not piece:
            continue
        starts_word = piece[0].isspace() or not words
        if starts_word:
            words.append(WordTimestamp(piece, token_times[token_index], token_times[token_index]))
        else:
            words[-1].word += piece
            words[-1].end = token_times[token_index]

    for index, word in enumerate(words):
        word.word = word.word.strip()
        if index + 1 < len(words):
            word.end = max(word.end, words[index + 1].start)
    return [word for word in words if word.word]
