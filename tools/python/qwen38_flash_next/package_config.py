# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""The `model.qwen38_graph` section of genai_config.json: migration of the config of an earlier package.

The qwen38_flash_graph engine rejects every key its config parser does not know when a package is loaded. The builders read the
config of an existing package and write the config of the package they build, so they run the section through
`migrate_qwen38_graph_config` and never carry a key of an earlier config surface into the new package.

The `mtp` / `drafter` / `head_r2` session labels of `weights_pool.txt` are a package file format and not config keys: the migration
does not touch them.
"""

from __future__ import annotations

# CUDA graph ids; the engine derives them from the run kind, the row count and the phase.
GRAPH_ID_KEYS = ("target_s1_graph_id", "target_s2_graph_id", "head_s1_graph_id", "head_s2_graph_id")
# Earlier names of the `fold.head_r2` session, in the order in which they are considered when `head_r2` itself is absent.
HEAD_R2_ALIASES = ("mtp", "drafter")


def migrate_qwen38_graph_config(qwen: dict) -> list[str]:
    """Rewrites `qwen`, a `model.qwen38_graph` object, in place to the keys the engine accepts; returns what it changed.

    * `fold.mtp` / `fold.drafter` become `fold.head_r2` (an existing `fold.head_r2` wins and the alias is dropped);
    * the CUDA graph ids (`target_s1_graph_id`, `target_s2_graph_id`, `head_s1_graph_id`, `head_s2_graph_id`,
      `fold.target_graph_id`, `fold.head_r2.graph_id`) are dropped;
    * `mtp.skip_refeed` and `fold.force_reject` are dropped (the engine always re-feeds, and the force-reject test hook is the
      environment variable ORT_QWEN38_FORCE_REJECT);
    * `prefill_chunk_tokens` is dropped. It only applied with `prefill.enabled` false, and there it becomes `prefill.chunk`, the one
      chunk setting.

    Keys the engine never knew are left alone, so the engine still reports them when the package is loaded.
    """
    changes: list[str] = []

    def drop(section, key: str, path: str) -> None:
        if isinstance(section, dict) and key in section:
            del section[key]
            changes.append(f"dropped {path}")

    for key in GRAPH_ID_KEYS:
        drop(qwen, key, key)
    drop(qwen.get("mtp"), "skip_refeed", "mtp.skip_refeed")

    fold = qwen.get("fold")
    if isinstance(fold, dict):
        drop(fold, "force_reject", "fold.force_reject")
        drop(fold, "target_graph_id", "fold.target_graph_id")
        for alias in HEAD_R2_ALIASES:
            if alias not in fold:
                continue
            session = fold.pop(alias)
            if "head_r2" in fold:
                changes.append(f"dropped fold.{alias} (fold.head_r2 is set)")
            else:
                fold["head_r2"] = session
                changes.append(f"renamed fold.{alias} to fold.head_r2")
        drop(fold.get("head_r2"), "graph_id", "fold.head_r2.graph_id")

    if "prefill_chunk_tokens" in qwen:
        tokens = qwen.pop("prefill_chunk_tokens")
        prefill = qwen.get("prefill")
        if not isinstance(prefill, dict):
            prefill = {}
        if tokens and not prefill.get("enabled"):
            prefill["chunk"] = tokens
            qwen["prefill"] = prefill
            changes.append("renamed prefill_chunk_tokens to prefill.chunk")
        else:
            changes.append("dropped prefill_chunk_tokens (prefill.chunk is the one chunk setting)")
    return changes
