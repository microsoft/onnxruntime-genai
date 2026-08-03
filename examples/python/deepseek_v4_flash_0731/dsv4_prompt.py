"""Prompt encoding for DeepSeek-V4-Flash-0731.

The checkpoint ships no Jinja chat template; the prompt format lives in its
`encoding/encoding_dsv4.py` reference module instead.  This wraps that module
plus the HF tokenizer so callers deal only in token ids and text.
"""

import functools
import os
import sys

EOS_TOKEN_ID = 1
_THINK_CLOSE = "</think>"


@functools.lru_cache(maxsize=4)
def _encoding_module(ckpt_dir):
    path = os.path.join(ckpt_dir, "encoding")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path} not found; point --tokenizer at the checkpoint")
    if path not in sys.path:
        sys.path.insert(0, path)
    import encoding_dsv4

    return encoding_dsv4


class DSV4Prompt:
    """Tokenizer + the checkpoint's own message encoder."""

    def __init__(self, ckpt_dir, thinking_mode="chat", reasoning_effort="low"):
        from transformers import AutoTokenizer

        self.ckpt_dir = os.path.abspath(os.path.expanduser(ckpt_dir))
        self.enc = _encoding_module(self.ckpt_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt_dir)
        self.thinking_mode = thinking_mode
        self.reasoning_effort = reasoning_effort
        self.eos_token_ids = [self.tokenizer.eos_token_id or EOS_TOKEN_ID]

    def render(self, messages):
        """OpenAI-style messages -> the model's prompt string."""
        return self.enc.encode_messages(messages, thinking_mode=self.thinking_mode,
                                        reasoning_effort=self.reasoning_effort)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def encode_messages(self, messages):
        return self.tokenizer.encode(self.render(messages))

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def parse(self, completion_text):
        """Split a completion into ``reasoning_content`` / ``content``.

        The reference parser expects a well-formed turn; a generation truncated by
        the token budget has no ``</think>``, so fall back to splitting by hand.
        """
        try:
            return self.enc.parse_message_from_completion_text(
                completion_text, thinking_mode=self.thinking_mode)
        except Exception:
            pass
        if self.thinking_mode == "thinking" and _THINK_CLOSE in completion_text:
            reasoning, content = completion_text.split(_THINK_CLOSE, 1)
        else:
            reasoning, content = "", completion_text
        return {"role": "assistant", "reasoning_content": reasoning.strip(),
                "content": content.strip(), "tool_calls": []}
