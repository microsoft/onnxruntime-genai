import importlib.util
from pathlib import Path

import torch


def _load_engine_module():
    path = (Path(__file__).parents[2] / "examples" / "python" /
            "deepseek_v4_flash_0731" / "dsv4_engine.py")
    spec = importlib.util.spec_from_file_location("dsv4_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sparse_speculative_sampling_matches_target_distribution():
    engine = _load_engine_module()
    count = 200_000
    generator = torch.Generator().manual_seed(7)

    target_idx = torch.tensor([[0, 1]]).expand(count, -1)
    target_prob = torch.tensor([[0.7, 0.3]]).expand(count, -1)
    draft_idx = torch.tensor([[0, 1, 2]]).expand(count, -1)
    draft_prob = torch.tensor([[0.2, 0.3, 0.5]]).expand(count, -1)
    drafts = torch.multinomial(draft_prob[0], count, replacement=True,
                               generator=generator)

    accepted, correction = engine._speculative_sample(
        torch, target_idx, target_prob, draft_idx, draft_prob, drafts, generator)
    output = torch.where(accepted, drafts, correction)
    observed = torch.bincount(output, minlength=3).float() / count

    torch.testing.assert_close(observed, torch.tensor([0.7, 0.3, 0.0]),
                               atol=0.004, rtol=0)