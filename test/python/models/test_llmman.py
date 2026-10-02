"""The `llmman serve` client: the daemon protocol behind oci:// model paths.

Exercised against a real HTTP server on a loopback port rather than mocks, so
the NDJSON streaming contract is genuinely tested.
"""

import http.server
import json
import socketserver
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py" / "models"))

from loaders import llmman


def _ndjson(*objs):
    return "".join(json.dumps(o) + "\n" for o in objs)


class _FakeDaemon:
    """A minimal stand-in for `llmman serve`, on a real loopback port."""

    def __init__(self):
        self.version = {"version": "0.1.0", "pid": 1}
        self.pull_body = _ndjson({"status": "success"})
        self.pull_status = 200
        self.last_request = None
        daemon = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body, ctype):
                raw = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self._send(200, json.dumps(daemon.version), "application/json")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                daemon.last_request = json.loads(self.rfile.read(length))
                self._send(daemon.pull_status, daemon.pull_body, "application/x-ndjson")

        self._server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def daemon():
    d = _FakeDaemon()
    yield d
    d.close()


def test_accepts_a_llmman_daemon(daemon):
    llmman.check_daemon(daemon.url)


def test_rejects_a_non_llmman_server(daemon):
    daemon.version = {"hello": "world"}
    with pytest.raises(RuntimeError, match="not an llmman daemon"):
        llmman.check_daemon(daemon.url)


def test_reports_nothing_listening_actionably():
    with pytest.raises(RuntimeError, match="llmman serve"):
        llmman.check_daemon("http://127.0.0.1:1")


def test_pull_succeeds_and_forwards_progress(daemon):
    daemon.pull_body = _ndjson(
        {"status": "pulling manifest"},
        {"status": "pulling blobs", "completed": 50, "total": 100},
        {"status": "success"},
    )
    seen = []
    llmman.pull(daemon.url, "ghcr.io/org/model:tag", lambda *a: seen.append(a))

    assert daemon.last_request == {"model": "ghcr.io/org/model:tag"}
    assert seen == [("pulling manifest", 0, 0), ("pulling blobs", 50, 100)]


def test_reports_an_in_band_error_at_http_200(daemon):
    # The daemon streams errors in-band, so a 200 does not mean success.
    daemon.pull_body = _ndjson({"status": "pulling"}, {"error": "unauthorized"})
    with pytest.raises(RuntimeError, match="unauthorized"):
        llmman.pull(daemon.url, "ref")


def test_rejects_a_stream_that_ends_without_success(daemon):
    daemon.pull_body = _ndjson({"status": "pulling blobs"})
    with pytest.raises(RuntimeError, match="without reporting success"):
        llmman.pull(daemon.url, "ref")


def test_reports_a_non_ok_status(daemon):
    daemon.pull_status = 400
    daemon.pull_body = '{"error":"bad request"}'
    with pytest.raises(RuntimeError):
        llmman.pull(daemon.url, "ref")


def test_tolerates_a_non_json_diagnostic_line(daemon):
    daemon.pull_body = "not json\n" + _ndjson({"status": "success"})
    llmman.pull(daemon.url, "ref")


def test_recognizes_the_oci_scheme():
    assert llmman.is_oci_ref("oci://ghcr.io/org/model:tag")
    assert llmman.is_oci_ref("OCI://ghcr.io/org/model:tag")


@pytest.mark.parametrize(
    "value",
    [
        "microsoft/Phi-3-mini-4k-instruct",
        "ghcr.io/org/model:tag",
        "/local/path/to/model",
        "",
        None,
    ],
)
def test_leaves_every_other_shape_alone(value):
    # A bare HF repo id must never be claimed.
    assert not llmman.is_oci_ref(value)


def test_strips_the_scheme_only_when_present():
    assert llmman.strip_scheme("oci://ghcr.io/org/model:tag") == "ghcr.io/org/model:tag"
    assert llmman.strip_scheme("microsoft/Phi-3") == "microsoft/Phi-3"


@pytest.mark.parametrize("ref", ["oci://", "oci://   "])
def test_rejects_an_empty_reference(ref):
    with pytest.raises(ValueError):
        llmman.resolve_model(ref)


@pytest.mark.parametrize(
    "host,want",
    [
        ("", "http://127.0.0.1:17434"),
        ("1.2.3.4:9999", "http://1.2.3.4:9999"),
        ("1.2.3.4", "http://1.2.3.4:17434"),
        # A wildcard bind is meaningful to the server but not to a client.
        ("0.0.0.0:9999", "http://127.0.0.1:9999"),
        ("[::]:9999", "http://[::1]:9999"),
        # The scheme and any path prefix (e.g. behind a reverse proxy) are kept.
        ("https://example.com:8443", "https://example.com:8443"),
        ("http://1.2.3.4:9999/llmman/", "http://1.2.3.4:9999/llmman"),
    ],
)
def test_endpoint_parsing(monkeypatch, host, want):
    monkeypatch.setenv(llmman.HOST_ENV, host)
    assert llmman.endpoint() == want


@pytest.fixture
def fake_pull(monkeypatch):
    llmman.resolve_model.cache_clear()
    calls = []

    def pull_and_resolve(reference, progress=None):
        calls.append(reference)
        return "/models/" + reference

    monkeypatch.setattr(llmman, "pull_and_resolve", pull_and_resolve)
    yield calls
    llmman.resolve_model.cache_clear()


def test_resolve_input_pulls_an_oci_ref_once(fake_pull):
    assert llmman.resolve_input("oci://ghcr.io/org/model:tag") == "/models/ghcr.io/org/model:tag"
    assert llmman.resolve_input("oci://ghcr.io/org/model:tag") == "/models/ghcr.io/org/model:tag"
    assert fake_pull == ["ghcr.io/org/model:tag"]


@pytest.mark.parametrize("value", ["microsoft/Phi-3", "/local/dir", "model.gguf", "", None])
def test_resolve_input_passes_everything_else_through(fake_pull, value):
    assert llmman.resolve_input(value) == value
    assert fake_pull == []


def test_get_hf_details_reads_config_from_the_pulled_directory(fake_pull, monkeypatch, tmp_path):
    builder = pytest.importorskip("builder")
    seen = []

    def from_pretrained(name, **kwargs):
        seen.append(name)
        return object()

    monkeypatch.setattr(builder.AutoConfig, "from_pretrained", from_pretrained)
    monkeypatch.setattr(builder.AutoTokenizer, "from_pretrained", from_pretrained)
    monkeypatch.setattr(builder, "add_special_token_ids", lambda config, tokenizer: None)
    monkeypatch.setattr(builder.os.path, "isdir", lambda path: path.startswith("/models/"))

    # oci:// is accepted on either flag.
    for model_name, input_path in [(None, "oci://ghcr.io/org/model:tag"), ("oci://ghcr.io/org/model:tag", "")]:
        seen.clear()
        details = builder.get_hf_details(model_name, input_path, str(tmp_path), {})
        assert details["hf_name"] == "/models/ghcr.io/org/model:tag"
        assert seen == [details["hf_name"]] * 2
    assert fake_pull == ["ghcr.io/org/model:tag"]


def test_load_weights_uses_the_pulled_directory(fake_pull, monkeypatch):
    base = pytest.importorskip("builders.base")
    quant_model = pytest.importorskip("loaders.quant_model")
    seen = []

    def from_pretrained(quant_type, input_path, **kwargs):
        seen.append(input_path)
        return object()

    monkeypatch.setattr(quant_model.QuantModel, "from_pretrained", from_pretrained)
    model = SimpleNamespace(
        quant_type="awq",
        quant_attrs={},
        num_attn_heads=1,
        num_kv_heads=1,
        head_size=1,
        intermediate_size=1,
        num_layers=1,
        extra_options={},
    )
    base.Model.load_weights(model, "oci://ghcr.io/org/model:tag")
    assert seen == ["/models/ghcr.io/org/model:tag"]
