"""URI model sources: scheme dispatch in `resolve_input` and its use by the builder."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py" / "models"))

from loaders import llmman, resolver


@pytest.fixture
def fake_pull(monkeypatch):
    calls = []

    def pull_and_resolve(reference, progress=None):
        calls.append(reference)
        return "/models/" + reference

    monkeypatch.setattr(llmman, "pull_and_resolve", pull_and_resolve)
    return calls


@pytest.mark.parametrize("value", ["oci://ghcr.io/org/model:tag", "OCI://ghcr.io/org/model:tag"])
def test_oci_scheme_is_resolved_through_llmman(fake_pull, value):
    assert resolver.resolve_input(value) == "/models/ghcr.io/org/model:tag"
    assert fake_pull == ["ghcr.io/org/model:tag"]


@pytest.mark.parametrize(
    "value",
    [
        "microsoft/Phi-3-mini-4k-instruct",
        # A bare registry reference is indistinguishable from a Hugging Face repo id.
        "ghcr.io/org/model:tag",
        "oci",
        "/local/dir",
        "model.gguf",
        "azureml://registries/r/models/m",
        "",
        None,
    ],
)
def test_everything_else_passes_through(fake_pull, value):
    assert resolver.resolve_input(value) == value
    assert fake_pull == []


@pytest.mark.parametrize(
    "model_name,input_path,resolved_input",
    [
        # oci:// is accepted on either flag.
        (None, "oci://ghcr.io/org/model:tag", "/models/ghcr.io/org/model:tag"),
        ("oci://ghcr.io/org/model:tag", "", ""),
    ],
)
def test_get_hf_details_reads_config_from_the_resolved_directory(
    fake_pull, monkeypatch, tmp_path, model_name, input_path, resolved_input
):
    builder = pytest.importorskip("builder")
    seen = []

    def from_pretrained(name, **kwargs):
        seen.append(name)
        return object()

    monkeypatch.setattr(builder.AutoConfig, "from_pretrained", from_pretrained)
    monkeypatch.setattr(builder.AutoTokenizer, "from_pretrained", from_pretrained)
    monkeypatch.setattr(builder, "add_special_token_ids", lambda config, tokenizer: None)
    monkeypatch.setattr(builder.os.path, "isdir", lambda path: path.startswith("/models/"))

    details = builder.get_hf_details(model_name, input_path, str(tmp_path), {})
    assert details["hf_name"] == "/models/ghcr.io/org/model:tag"
    assert details["input_path"] == resolved_input
    assert seen == [details["hf_name"]] * 2
    assert fake_pull == ["ghcr.io/org/model:tag"]


def test_create_model_loads_weights_from_the_resolved_input(monkeypatch, tmp_path):
    builder = pytest.importorskip("builder")
    loaded = []

    class FakeModel:
        exclude_embeds = False

        def __init__(self, *args):
            pass

        def make_model(self, input_path):
            loaded.append(input_path)

        def save_model(self, *args):
            pass

        def make_genai_config(self, *args):
            pass

        def save_processing(self, *args):
            pass

    monkeypatch.setattr(builder, "LlamaModel", FakeModel)
    hf_details = {
        "extra_kwargs": {},
        "hf_name": "/models/m",
        "hf_config": SimpleNamespace(architectures=["LlamaForCausalLM"]),
        "input_path": "/models/m",
    }
    builder.create_model(
        None,
        "oci://ghcr.io/org/model:tag",
        str(tmp_path / "out"),
        "fp16",
        "cpu",
        str(tmp_path / "cache"),
        hf_details=hf_details,
    )
    assert loaded == ["/models/m"]
