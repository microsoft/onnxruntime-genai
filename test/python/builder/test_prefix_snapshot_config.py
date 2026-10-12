# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import sys
from importlib import import_module
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py" / "models"))
apply_runtime_config = import_module("builder_config").apply_runtime_config


def test_runtime_snapshot_tiers_and_count_cap_are_preserved():
    generated = {"engine": {"dynamic_batching": {"num_blocks": 128}}}
    policy = {
        "prefix_caching": True,
        "dflash2_max_snapshots": 8,
        "dflash2_snapshot_cache": {
            "vram_bytes": 4_294_967_296,
            "ram_bytes": 8_589_934_592,
            "disk_bytes": 17_179_869_184,
            "directory": "spill",
        },
    }
    runtime = {"engine": {"dynamic_batching": policy}}
    updated = apply_runtime_config(generated, runtime)
    assert updated["engine"]["dynamic_batching"] == {"num_blocks": 128, **policy}
    assert generated == {"engine": {"dynamic_batching": {"num_blocks": 128}}}
    profile = {
        "id": "snapshot-storage",
        "eligibility": {"minimum_total_device_memory_bytes": 0},
        "overlay": {"engine": {"dynamic_batching": policy}},
    }
    assert apply_runtime_config(generated, {"runtime_profiles": [profile]})["runtime_profiles"] == [profile]


@pytest.mark.parametrize("tier", ["vram_bytes", "ram_bytes", "disk_bytes"])
@pytest.mark.parametrize("invalid", [-1, 1.5, True, "16", 9_007_199_254_740_992])
def test_runtime_rejects_invalid_snapshot_byte_limits(tier, invalid):
    generated = {"engine": {"dynamic_batching": {"num_blocks": 128}}}
    runtime = {"engine": {"dynamic_batching": {"dflash2_snapshot_cache": {tier: invalid}}}}
    with pytest.raises(ValueError, match="non-negative integer byte count"):
        apply_runtime_config(generated, runtime)


@pytest.mark.parametrize("invalid", [-1, 1.5, True, 2_147_483_648])
def test_runtime_rejects_invalid_snapshot_count_cap(invalid):
    generated = {"engine": {"dynamic_batching": {"num_blocks": 128}}}
    with pytest.raises(ValueError, match="dflash2_max_snapshots"):
        apply_runtime_config(generated, {"engine": {"dynamic_batching": {"dflash2_max_snapshots": invalid}}})
