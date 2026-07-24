# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# ruff: noqa: PLC0415

"""Tests for the GenAI telemetry module.

These tests verify:
- Telemetry singleton behavior
- Opt-out mechanisms (env var, API, CI detection)
- Device ID generation and persistence
- System info collection
- Event emission (model build, benchmark, model load, inference, error)
- Decorator and context manager patterns
- Telemetry never crashes the application
"""

import os
import stat
import sys
import tempfile
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import MagicMock, call, patch

_TELEMETRY_SOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "python", "py"))
_TELEMETRY_SOURCE_PATH_ADDED = _TELEMETRY_SOURCE_PATH not in sys.path
if _TELEMETRY_SOURCE_PATH_ADDED:
    sys.path.insert(0, _TELEMETRY_SOURCE_PATH)


def tearDownModule():
    if _TELEMETRY_SOURCE_PATH_ADDED and _TELEMETRY_SOURCE_PATH in sys.path:
        sys.path.remove(_TELEMETRY_SOURCE_PATH)


class _HermeticTelemetryTestCase(unittest.TestCase):
    """Base for tests that construct ``GenAITelemetry``.

    Guarantees no unit test touches the network or the real user profile: the
    HTTP transport is stubbed (``self.mock_send``) and the durable-store
    directory is redirected to a temp dir. Ambient CI / opt-out signals are
    cleared so each test's chosen mode is not masked by the test runner's
    environment.
    """

    _ENV_SIGNALS = (
        "ORT_DISABLE_TELEMETRY",
        "CI",
        "TF_BUILD",
        "GITHUB_ACTIONS",
        "JENKINS_URL",
        "TRAVIS",
        "CIRCLECI",
        "GITLAB_CI",
        "BUILD_ID",
    )

    def setUp(self):
        import tempfile

        import telemetry.deviceid as deviceid
        from telemetry.telemetry import GenAITelemetry

        GenAITelemetry._instance = None

        self._tmpdir = tempfile.mkdtemp()
        self._patchers = []

        env_patcher = patch.dict(os.environ, {}, clear=False)
        env_patcher.start()
        self._patchers.append(env_patcher)
        for var in self._ENV_SIGNALS:
            os.environ.pop(var, None)

        self.sent_payloads = []

        def _record_send(payload, timeout_sec, item_count=1):
            self.sent_payloads.append(bytes(payload))
            return (True, 204)

        send_patcher = patch(
            "telemetry.library.transport.HttpJsonPostTransport.send",
            side_effect=_record_send,
        )
        self.mock_send = send_patcher.start()
        self._patchers.append(send_patcher)

        dir_patcher = patch("telemetry.telemetry.get_telemetry_base_dir", return_value=self._tmpdir)
        dir_patcher.start()
        self._patchers.append(dir_patcher)

        system_info_patcher = patch("telemetry.telemetry.get_system_info", return_value={})
        system_info_patcher.start()
        self._patchers.append(system_info_patcher)
        provider_info_patcher = patch(
            "telemetry.telemetry.get_execution_provider_info",
            return_value={"available_providers": []},
        )
        provider_info_patcher.start()
        self._patchers.append(provider_info_patcher)

        deviceid._device_id_state.update({"device_id": None, "status": deviceid.DeviceIdStatus.NEW})
        deviceid_platform_patcher = patch("telemetry.deviceid.platform.system", return_value="Linux")
        deviceid_platform_patcher.start()
        self._patchers.append(deviceid_platform_patcher)
        deviceid_dir_patcher = patch(
            "telemetry.deviceid.get_telemetry_base_dir",
            return_value=Path(self._tmpdir),
        )
        deviceid_dir_patcher.start()
        self._patchers.append(deviceid_dir_patcher)

    def tearDown(self):
        import shutil

        import telemetry.deviceid as deviceid
        from telemetry.telemetry import GenAITelemetry

        instance = GenAITelemetry._instance
        if instance is not None:
            # Quiesce background threads before un-stubbing the network. The
            # heartbeat join is unbounded on purpose: if it returned while the
            # thread were still alive, restoring the real transport would let it
            # POST real device data from a unit test. The heartbeat is bounded by
            # system_info's per-probe subprocess timeouts (and cached after the
            # first call), so this never hangs the suite.
            if instance._heartbeat_thread is not None:
                instance._heartbeat_thread.join()
            instance.shutdown(5)
        for p in reversed(self._patchers):
            p.stop()
        GenAITelemetry._instance = None
        deviceid._device_id_state.update({"device_id": None, "status": deviceid.DeviceIdStatus.NEW})
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _join_heartbeat(self):
        from telemetry.telemetry import GenAITelemetry

        t = GenAITelemetry._instance
        if t is not None and t._heartbeat_thread is not None:
            t._heartbeat_thread.join()

    def _deliver(self):
        """Join the heartbeat and drain the uploader so every queued event is
        recorded in ``self.sent_payloads`` deterministically."""
        from telemetry.telemetry import GenAITelemetry

        t = GenAITelemetry._instance
        if t is None:
            return None
        if t._heartbeat_thread is not None:
            t._heartbeat_thread.join()
        if t._uploader is not None:
            self.assertTrue(t._uploader.stop_loop())
            for _ in range(20):
                if t._store is None or t._store.count() == 0:
                    break
                t._uploader.drain_once()
        return t

    def _sent_event_names(self):
        names = []
        for payload in self.sent_payloads:
            for token in (
                b"GenAIHeartbeat",
                b"GenAIModelBuild",
                b"GenAIBenchmark",
                b"GenAIModelLoad",
                b"GenAIInference",
                b"GenAIAction",
                b"GenAIError",
            ):
                if token in payload:
                    names.append(token.decode())
        return names


class TestOptOut(_HermeticTelemetryTestCase):
    """Test the three-state telemetry semantics: enabled / opt-out / CI."""

    def test_ci_sends_nothing(self):
        from telemetry.telemetry import GenAITelemetry

        os.environ["CI"] = "true"
        t = GenAITelemetry()
        self.assertFalse(t._enabled)
        self.assertFalse(t.accepts_detailed_events)
        # CI creates no store/uploader and no heartbeat — nothing is recorded.
        self.assertIsNone(t._store)
        self.assertIsNone(t._heartbeat_thread)
        self.assertFalse(self.mock_send.called)

    def test_github_actions_sends_nothing(self):
        from telemetry.telemetry import GenAITelemetry

        os.environ["GITHUB_ACTIONS"] = "true"
        t = GenAITelemetry()
        self.assertFalse(t._enabled)
        self.assertIsNone(t._store)
        self.assertIsNone(t._heartbeat_thread)
        self.assertFalse(self.mock_send.called)

    def test_opt_out_sends_nothing(self):
        from telemetry.telemetry import GenAITelemetry

        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        with patch("telemetry.telemetry.get_encrypted_device_id_and_status") as mock_device_id:
            t = GenAITelemetry()
        # Full process-lifetime opt-out: no resources and no network sends.
        self.assertFalse(t._enabled)
        self.assertIsNone(t._store)
        self.assertIsNone(t._uploader)
        self.assertIsNone(t._heartbeat_thread)
        # A detailed-event method must be a no-op and must not raise.
        t.log_model_build(action="create_model", duration_ms=1.0, success=True)
        self._deliver()
        mock_device_id.assert_not_called()
        self.assertEqual(self.sent_payloads, [])

    def test_enabled_records_heartbeat_and_events(self):
        from telemetry.telemetry import GenAITelemetry

        t = GenAITelemetry()
        self.assertTrue(t._enabled)
        self.assertTrue(t.accepts_detailed_events)
        self.assertIsNotNone(t._store)
        t.log_model_build(action="create_model", duration_ms=1.0, success=True)
        self._deliver()
        names = self._sent_event_names()
        self.assertIn("GenAIHeartbeat", names)
        self.assertIn("GenAIModelBuild", names)

    def test_disable_enable_api(self):
        from telemetry.telemetry import GenAITelemetry

        t = GenAITelemetry()
        self._join_heartbeat()
        self.assertTrue(t._enabled)
        self.assertIsNotNone(t._store)
        t.disable_telemetry()
        self.assertFalse(t._enabled)
        t.enable_telemetry()
        self.assertTrue(t._enabled)
        self.assertIsNotNone(t._store)

    def test_runtime_disable_skips_pending_heartbeat(self):
        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        telemetry._enabled = False
        telemetry._telemetry_disabled = False
        telemetry._store = MagicMock()
        with patch("telemetry.telemetry.get_encrypted_device_id_and_status") as mock_device_id:
            telemetry._send_heartbeat()

        mock_device_id.assert_not_called()
        telemetry._store.store.assert_not_called()

    def test_initialization_keeps_exporter_diagnostics_configurable(self):
        from telemetry.library.event_source import event_source
        from telemetry.telemetry import GenAITelemetry

        event_source.logger.disabled = False
        GenAITelemetry()
        self._join_heartbeat()

        self.assertFalse(event_source.logger.disabled)

    def test_enable_telemetry_does_not_override_env_opt_out(self):
        from telemetry.telemetry import GenAITelemetry

        os.environ["ORT_DISABLE_TELEMETRY"] = "true"
        t = GenAITelemetry()
        self.assertFalse(t._enabled)
        self.assertIsNone(t._store)
        t.shutdown()
        os.environ.pop("ORT_DISABLE_TELEMETRY")

        # Full suppression remains latched after env removal and reinitialization.
        self.assertIs(GenAITelemetry(), t)
        t.enable_telemetry()
        self.assertFalse(t._enabled)
        self.assertIsNone(t._store)
        self.assertIsNone(t._heartbeat_thread)
        self.assertEqual(self.sent_payloads, [])

    def test_closed_store_allows_initialization_retry(self):
        from telemetry.telemetry import GenAITelemetry

        closed_store = MagicMock(is_open=False)
        open_store = MagicMock(is_open=True)
        with (
            patch("telemetry.telemetry.OfflineEventStore", side_effect=[closed_store, open_store]),
            patch("telemetry.telemetry.EventUploader") as mock_uploader,
        ):
            first = GenAITelemetry()
            self.assertFalse(first._initialized)

            second = GenAITelemetry()

        self.assertIs(second, first)
        self.assertTrue(second._initialized)
        self.assertTrue(second._enabled)
        mock_uploader.assert_called_once_with(open_store, instrumentation_key=second._instrumentation_key)

    def test_initialization_failure_closes_partial_resources(self):
        from telemetry.telemetry import GenAITelemetry

        store = MagicMock(is_open=True)
        uploader = MagicMock()
        uploader.stop_loop.return_value = True
        heartbeat = MagicMock(ident=None)
        heartbeat.is_alive.return_value = False
        heartbeat.start.side_effect = RuntimeError("thread start failed")
        with (
            patch("telemetry.telemetry.OfflineEventStore", return_value=store),
            patch("telemetry.telemetry.EventUploader", return_value=uploader),
            patch("telemetry.telemetry.threading.Thread", return_value=heartbeat),
        ):
            telemetry = GenAITelemetry()

        uploader.stop_loop.assert_called_once()
        uploader.flush.assert_called_once()
        uploader.close.assert_called_once()
        store.close.assert_called_once()
        self.assertIsNone(telemetry._heartbeat_thread)
        self.assertIsNone(telemetry._uploader)
        self.assertIsNone(telemetry._store)
        self.assertFalse(telemetry._enabled)
        self.assertFalse(telemetry._initialized)


class TestVersionResolution(unittest.TestCase):
    def test_installed_package_exposes_telemetry_modules(self):
        import importlib

        try:
            importlib.import_module("onnxruntime_genai")
        except ImportError:
            self.skipTest("onnxruntime_genai is not installed in this test environment")

        telemetry = importlib.import_module("onnxruntime_genai.telemetry")
        path_utils = importlib.import_module("onnxruntime_genai.telemetry.path_utils")
        self.assertTrue(hasattr(telemetry, "GenAITelemetry"))
        self.assertTrue(hasattr(path_utils, "sanitize_model_identifier"))

    def test_variant_distribution_version_is_resolved(self):
        from telemetry.telemetry import _get_app_version

        with (
            patch.dict(sys.modules, {"onnxruntime_genai": None}),
            patch(
                "telemetry.telemetry.distribution_version",
                side_effect=[PackageNotFoundError, "0.15.0"],
            ) as mock_version,
        ):
            self.assertEqual(_get_app_version(), "0.15.0")

        self.assertEqual(
            mock_version.call_args_list,
            [call("onnxruntime-genai"), call("onnxruntime-genai-cuda")],
        )


class TestBenchmarkTelemetryIdentifiers(unittest.TestCase):
    @staticmethod
    def _load_helper():
        import importlib.util

        helper_path = Path(__file__).parents[2] / "benchmark" / "python" / "telemetry_utils.py"
        spec = importlib.util.spec_from_file_location("benchmark_telemetry_utils", helper_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_sanitizes_paths_without_changing_model_ids(self):
        module = self._load_helper()

        self.assertEqual(
            module.sanitize_model_identifier(r"C:\Users\alice\models\model.onnx"),
            "[path]",
        )
        self.assertEqual(
            module.sanitize_model_identifier("/home/alice/models/model.onnx"),
            "[path]",
        )
        self.assertEqual(module.sanitize_model_identifier("microsoft/phi-3-mini"), "microsoft/phi-3-mini")
        self.assertEqual(module.normalize_execution_provider("NvTensorRtRtx"), "trt-rtx")
        path_utils_module = sys.modules[module.sanitize_model_identifier.__module__]
        with patch.object(path_utils_module.os.path, "exists") as mock_exists:
            self.assertEqual(module.sanitize_model_identifier("microsoft/phi-3-mini"), "microsoft/phi-3-mini")
        mock_exists.assert_not_called()

    def test_source_telemetry_loader_restores_sys_path(self):
        import types

        module = self._load_helper()
        telemetry_stub = types.ModuleType("telemetry")

        class StubTelemetry:
            pass

        telemetry_stub.GenAITelemetry = StubTelemetry
        source_root = str(Path(__file__).parents[2] / "src" / "python" / "py")
        source_index = sys.path.index(source_root) if source_root in sys.path else None
        if source_index is not None:
            sys.path.pop(source_index)
        try:
            before = list(sys.path)
            with patch.dict(
                sys.modules,
                {
                    "onnxruntime_genai": None,
                    "onnxruntime_genai.telemetry": None,
                    "telemetry": telemetry_stub,
                },
            ):
                self.assertIsInstance(module.get_telemetry(), StubTelemetry)
            self.assertEqual(sys.path, before)
        finally:
            if source_index is not None:
                sys.path.insert(source_index, source_root)


class TestActionFastPath(unittest.TestCase):
    def test_disabled_action_skips_stack_inspection(self):
        from telemetry.telemetry_extensions import action

        telemetry = MagicMock(accepts_detailed_events=False)

        @action
        def work():
            return 42

        with (
            patch("telemetry.telemetry_extensions._get_telemetry", return_value=telemetry),
            patch("telemetry.telemetry_extensions._resolve_invoked_from") as mock_resolve,
        ):
            self.assertEqual(work(), 42)

        mock_resolve.assert_not_called()

    def test_nested_actions_log_error_once(self):
        from telemetry.telemetry_extensions import action

        telemetry = MagicMock(accepts_detailed_events=True)

        @action
        @action
        def fail():
            raise ValueError("boom")

        with (
            patch("telemetry.telemetry_extensions._get_telemetry", return_value=telemetry),
            patch("telemetry.telemetry_extensions.log_error") as mock_log_error,
            self.assertRaisesRegex(ValueError, "boom"),
        ):
            fail()

        mock_log_error.assert_called_once()

    def test_positional_function_uses_function_action_name(self):
        from telemetry.telemetry_extensions import action

        telemetry = MagicMock(accepts_detailed_events=True)

        @action
        def work(value):
            return value

        with (
            patch("telemetry.telemetry_extensions._get_telemetry", return_value=telemetry),
            patch("telemetry.telemetry_extensions._resolve_invoked_from", return_value="test"),
            patch("telemetry.telemetry_extensions.log_action") as mock_log_action,
        ):
            self.assertEqual(work("value"), "value")

        self.assertEqual(mock_log_action.call_args.kwargs["action_name"], "work")

    def test_action_context_without_start_time_reports_zero_duration(self):
        from telemetry.telemetry_extensions import ActionContext

        telemetry = MagicMock(accepts_detailed_events=True)
        with (
            patch("telemetry.telemetry_extensions._get_telemetry", return_value=telemetry),
            patch("telemetry.telemetry_extensions._resolve_invoked_from", return_value="test"),
            patch("telemetry.telemetry_extensions.time.perf_counter", return_value=100.0),
            patch("telemetry.telemetry_extensions.log_action") as mock_log_action,
        ):
            context = ActionContext("work")
            context.__exit__(None, None, None)

        self.assertEqual(mock_log_action.call_args.kwargs["duration_ms"], 0)


class TestPathRedaction(unittest.TestCase):
    """Test absolute-path redaction in error telemetry."""

    def test_redacts_paths_and_usernames(self):
        from telemetry.telemetry import _redact_paths

        self.assertEqual(_redact_paths(r"err C:\Users\alice\model.onnx"), "err [path]")
        self.assertEqual(_redact_paths("/var/data/run/output.log"), "[path]")
        # Last segment is a directory/username (no extension) -> fully redacted.
        self.assertEqual(_redact_paths("at /home/bob"), "at [path]")
        # UNC paths are redacted too.
        self.assertEqual(_redact_paths(r"unc \\server\share\secret"), "unc [path]")
        self.assertEqual(_redact_paths(r"err C:\Users\Alice Smith\models\phi.onnx"), "err [path]")
        self.assertEqual(_redact_paths("err /home/Alice Smith/models/phi.onnx"), "err [path]")

    def test_redaction_relative_path_and_general_length_contract(self):
        from telemetry.telemetry import _redact_paths

        self.assertEqual(_redact_paths("a/b/c"), "[path]")
        self.assertEqual(_redact_paths(r"Load Users\bob\model.onnx failed"), "Load [path]")
        self.assertEqual(_redact_paths("models/foo.onnx"), "models/foo.onnx")
        self.assertEqual(_redact_paths("ratio 3/4 and and/or"), "ratio 3/4 and and/or")
        self.assertEqual(_redact_paths("before /home/alice/model.onnx\nafter"), "before [path]")
        self.assertEqual(len(_redact_paths("x" * 300).encode("utf-8")), 256)
        self.assertEqual(_redact_paths("x" * 255 + "€"), "x" * 255)

    def test_error_messages_are_capped_at_40960_utf8_bytes(self):
        from telemetry.path_utils import MAX_ERROR_MESSAGE_LENGTH
        from telemetry.telemetry import GenAITelemetry
        from telemetry.telemetry_extensions import log_error

        telemetry = MagicMock()
        with patch("telemetry.telemetry_extensions._get_telemetry", return_value=telemetry):
            log_error("RuntimeError", "x" * (MAX_ERROR_MESSAGE_LENGTH + 100))
            truncated = telemetry.log.call_args.args[1]["exception_message"]
            self.assertEqual(len(truncated.encode("utf-8")), MAX_ERROR_MESSAGE_LENGTH)

            log_error("RuntimeError", "x" * (MAX_ERROR_MESSAGE_LENGTH - 1) + "€")
            multibyte = telemetry.log.call_args.args[1]["exception_message"]
            self.assertEqual(multibyte, "x" * (MAX_ERROR_MESSAGE_LENGTH - 1))

        core = object.__new__(GenAITelemetry)
        core._enabled = True
        core._store = object()
        core._emit = MagicMock()
        core.log_error("RuntimeError", "x" * (MAX_ERROR_MESSAGE_LENGTH + 100))
        core_message = core._emit.call_args.args[1]["exception_message"]
        self.assertEqual(len(core_message.encode("utf-8")), MAX_ERROR_MESSAGE_LENGTH)

    def test_format_exception_message_redacts_source_line_paths(self):
        from telemetry.telemetry import _format_exception_message

        try:
            raise RuntimeError(r"open C:\Users\alice\secret\weights.bin failed")
        except RuntimeError as exc:
            message = _format_exception_message(exc, exc.__traceback__)
        # The username must not survive in the source line or the message.
        self.assertNotIn("alice", message)
        self.assertIn("[path]", message)

    def test_format_exception_message_keeps_external_basename(self):
        from telemetry.telemetry import _format_exception_message

        with patch(
            "telemetry.telemetry.traceback.format_exception",
            return_value=['  File "/home/Alice Smith/project/external.py", line 7, in run\n'],
        ):
            message = _format_exception_message(RuntimeError("boom"))

        self.assertEqual(message, 'File "[path]", line 7, in run')

    def test_format_exception_message_keeps_internal_basename_and_context(self):
        from telemetry.telemetry import _format_exception_message

        with patch(
            "telemetry.telemetry.traceback.format_exception",
            return_value=['  File "/home/user/onnxruntime_genai/telemetry/telemetry.py", line 9, in run\n'],
        ):
            message = _format_exception_message(RuntimeError("boom"))

        self.assertEqual(message, 'File "[path]", line 9, in run')

    def test_public_log_error_redacts_paths(self):
        from telemetry.telemetry_extensions import log_error

        telemetry = MagicMock()
        with patch("telemetry.telemetry_extensions._get_telemetry", return_value=telemetry):
            log_error(
                "FileNotFoundError",
                r"missing C:\Users\Alice Smith\models\phi.onnx",
                metadata={"exception_message": r"C:\Users\Mallory\secret.txt"},
            )

        attributes = telemetry.log.call_args.args[1]
        self.assertEqual(attributes["exception_message"], "missing [path]")

    def test_core_event_methods_redact_model_names(self):
        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        telemetry._enabled = True
        telemetry._store = object()
        telemetry._emit = MagicMock()
        model_path = r"C:\Users\Alice Smith\models\phi.onnx"
        calls = (
            lambda: telemetry.log_model_build("build", 1.0, True, model_name=model_path),
            lambda: telemetry.log_benchmark(model_name=model_path),
            lambda: telemetry.log_model_load(model_name=model_path),
            lambda: telemetry.log_inference(model_name=model_path),
            lambda: telemetry.log_error("RuntimeError", "boom", model_name=model_path),
        )

        for invoke in calls:
            telemetry._emit.reset_mock()
            invoke()
            self.assertEqual(telemetry._emit.call_args.args[1]["model_name"], "[path]")

    def test_core_model_build_recursively_scrubs_extra_options(self):
        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        telemetry._enabled = True
        telemetry._store = object()
        telemetry._emit = MagicMock()
        telemetry.log_model_build(
            "build",
            1.0,
            True,
            extra_options={
                "adapter_path": Path("/home/alice/private/adapter"),
                Path("/home/alice/private/key"): {"paths": [r"C:\Users\Alice Smith\model.onnx"]},
                "batch_size": 4,
            },
        )

        extra_options = telemetry._emit.call_args.args[1]["extra_options"]
        self.assertEqual(extra_options["adapter_path"], "[path]")
        self.assertEqual(extra_options["[path]"]["paths"], ["[path]"])
        self.assertEqual(extra_options["batch_size"], 4)

    def test_action_and_error_metadata_are_recursively_scrubbed(self):
        from telemetry.telemetry_extensions import log_action, log_error

        telemetry = MagicMock()
        metadata = {
            "path": r"C:\Users\alice\models\model.onnx",
            r"C:\Users\alice\secret": "value",
            "pathlike": Path("/home/alice/models/model.onnx"),
            "pathlike_key": {Path("/home/alice/private/key"): "value"},
            "nested": {
                "/home/alice/private/key": "value",
                "paths": ["/home/alice/model.onnx"],
            },
        }
        with patch("telemetry.telemetry_extensions._get_telemetry", return_value=telemetry):
            log_action("test", "work", 1.0, True, metadata)
            action_attributes = telemetry.log.call_args.args[1]
            log_error("RuntimeError", "boom", metadata)
            error_attributes = telemetry.log.call_args.args[1]

        for attributes in (action_attributes, error_attributes):
            self.assertEqual(attributes["path"], "[path]")
            self.assertEqual(attributes["pathlike"], "[path]")
            self.assertEqual(attributes["pathlike_key"]["[path]"], "value")
            self.assertEqual(attributes["nested"]["paths"], ["[path]"])
            self.assertEqual(attributes["[path]"], "value")
            self.assertEqual(attributes["nested"]["[path]"], "value")

    def test_public_helpers_never_propagate_failures(self):
        from telemetry.telemetry_extensions import log_action, log_error

        with patch("telemetry.telemetry_extensions._get_telemetry", side_effect=RuntimeError("telemetry failed")):
            log_action("test", "work", 1.0, True, metadata=["not", "a", "dict"])
            log_error("RuntimeError", "boom", metadata=["not", "a", "dict"])


class TestDeviceId(unittest.TestCase):
    """Test device ID generation."""

    def setUp(self):
        import telemetry.deviceid as deviceid

        self._tmpdir = tempfile.TemporaryDirectory()
        self._get_telemetry_base_dir = deviceid.get_telemetry_base_dir
        self._platform_patcher = patch("telemetry.deviceid.platform.system", return_value="Linux")
        self._dir_patcher = patch("telemetry.deviceid.get_telemetry_base_dir", return_value=Path(self._tmpdir.name))
        self._platform_patcher.start()
        self._dir_patcher.start()
        deviceid._device_id_state.update({"device_id": None, "status": deviceid.DeviceIdStatus.NEW})

    def tearDown(self):
        import telemetry.deviceid as deviceid

        self._dir_patcher.stop()
        self._platform_patcher.stop()
        deviceid._device_id_state.update({"device_id": None, "status": deviceid.DeviceIdStatus.NEW})
        self._tmpdir.cleanup()

    def test_get_encrypted_device_id(self):
        import telemetry.deviceid as deviceid

        device_id, status = deviceid.get_encrypted_device_id_and_status()
        # Should return a non-empty hex string (SHA256 = 64 hex chars)
        if status != deviceid.DeviceIdStatus.FAILED:
            self.assertEqual(len(device_id), 64)
            # Should be uppercase hex
            self.assertTrue(all(c in "0123456789ABCDEF" for c in device_id))
        self.assertIn(status, list(deviceid.DeviceIdStatus))

    def test_windows_base_dir_uses_shared_developer_tools_path(self):
        self._get_telemetry_base_dir.cache_clear()
        try:
            with (
                patch("telemetry.deviceid.platform.system", return_value="Windows"),
                patch.dict(os.environ, {"LOCALAPPDATA": r"C:\Users\test\AppData\Local"}, clear=False),
            ):
                path = self._get_telemetry_base_dir()
            self.assertEqual(
                path,
                Path(r"C:\Users\test\AppData\Local") / "Microsoft" / "DeveloperTools" / ".onnxruntime",
            )
        finally:
            self._get_telemetry_base_dir.cache_clear()

    def test_device_id_consistent(self):
        import telemetry.deviceid as deviceid

        id1, _ = deviceid.get_encrypted_device_id_and_status()
        id2, _ = deviceid.get_encrypted_device_id_and_status()
        self.assertEqual(id1, id2)

    def test_file_store_uses_owner_only_creation_mode(self):
        import telemetry.deviceid as deviceid

        with patch.object(Path, "mkdir") as mock_mkdir:
            deviceid._FileStore().store_id("test-device-id")

        mock_mkdir.assert_called_once_with(mode=0o700, parents=True, exist_ok=True)

    def test_permission_tightening_is_best_effort(self):
        import telemetry.deviceid as deviceid

        with patch.object(Path, "chmod", side_effect=OSError):
            deviceid._chmod_best_effort(Path(self._tmpdir.name), 0o700)

    def test_windows_store_uses_least_privilege_access(self):
        import telemetry.deviceid as deviceid

        winreg = MagicMock(
            HKEY_CURRENT_USER=object(),
            KEY_SET_VALUE=0x0002,
            KEY_CREATE_SUB_KEY=0x0004,
            KEY_WOW64_64KEY=0x0100,
            REG_SZ=1,
        )
        key_handle = object()
        winreg.CreateKeyEx.return_value.__enter__.return_value = key_handle

        with patch.dict(sys.modules, {"winreg": winreg}):
            deviceid._WindowsStore().store_id("test-device-id")

        winreg.CreateKeyEx.assert_called_once_with(
            winreg.HKEY_CURRENT_USER,
            deviceid._WindowsStore.REGISTRY_PATH,
            reserved=0,
            access=winreg.KEY_SET_VALUE | winreg.KEY_CREATE_SUB_KEY | winreg.KEY_WOW64_64KEY,
        )
        winreg.SetValueEx.assert_called_once_with(
            key_handle,
            deviceid._WindowsStore.REGISTRY_KEY,
            0,
            winreg.REG_SZ,
            "test-device-id",
        )


class TestSystemInfo(unittest.TestCase):
    """Test system information collection."""

    def setUp(self):
        from telemetry.system_info import get_system_info

        get_system_info.cache_clear()
        self.addCleanup(get_system_info.cache_clear)

    def test_get_system_info(self):
        from telemetry.system_info import get_system_info

        failed_probe = MagicMock(returncode=1, stdout="")
        with patch("telemetry.system_info.subprocess.run", return_value=failed_probe) as mock_run:
            info = get_system_info()

        # Should have all expected keys
        expected_keys = [
            "os",
            "os_version",
            "os_arch",
            "processor_count",
            "python_version",
            "gpu_name",
            "total_memory_mb",
        ]
        for key in expected_keys:
            self.assertIn(key, info, f"Missing key: {key}")

        # OS should be a known value
        self.assertIn(info["os"], ["Windows", "Linux", "Darwin", ""])

        # Processor count should be positive
        self.assertGreater(info["processor_count"], 0)

        # Python version should match
        self.assertTrue(info["python_version"].startswith(str(sys.version_info.major)))
        mock_run.assert_called()

    def test_nvidia_gpu_count_uses_output_rows(self):
        from telemetry.system_info import _get_gpu_info

        result = MagicMock(
            returncode=0,
            stdout="GPU A, 555.1, 8192\nGPU B, 555.1, 16384\n",
        )
        with patch("telemetry.system_info.subprocess.run", return_value=result):
            info = _get_gpu_info()

        self.assertEqual(info["gpu_name"], "GPU A")
        self.assertEqual(info["gpu_memory_mb"], 8192)
        self.assertEqual(info["gpu_count"], 2)

    def test_windows_wmi_gpu_count_uses_output_rows(self):
        from telemetry.system_info import _get_gpu_info

        nvidia_result = MagicMock(returncode=1, stdout="")
        wmi_result = MagicMock(
            returncode=0,
            stdout=("Node,AdapterRAM,DriverVersion,Name\nHOST,8589934592,31.0,GPU A\nHOST,4294967296,30.0,GPU B\n"),
        )
        with (
            patch("telemetry.system_info.platform.system", return_value="Windows"),
            patch("telemetry.system_info.subprocess.run", side_effect=[nvidia_result, wmi_result]),
        ):
            info = _get_gpu_info()

        self.assertEqual(info["gpu_name"], "GPU A")
        self.assertEqual(info["gpu_memory_mb"], 8192)
        self.assertEqual(info["gpu_count"], 2)

    def test_unknown_cpu_count_defaults_to_one(self):
        from telemetry.system_info import get_system_info

        get_system_info.cache_clear()
        try:
            with (
                patch("telemetry.system_info.os.cpu_count", return_value=None),
                patch("telemetry.system_info._get_cpu_model", return_value=""),
                patch("telemetry.system_info._get_total_memory_mb", return_value=0),
                patch("telemetry.system_info._get_gpu_info", return_value={}),
                patch("telemetry.system_info._get_device_manufacturer", return_value=""),
                patch("telemetry.system_info._get_device_model", return_value=""),
                patch("telemetry.system_info._get_ort_version", return_value=""),
            ):
                info = get_system_info()
            self.assertEqual(info["processor_count"], 1)
        finally:
            get_system_info.cache_clear()

    def test_system_info_cached(self):
        from telemetry.system_info import get_system_info

        failed_probe = MagicMock(returncode=1, stdout="")
        with patch("telemetry.system_info.subprocess.run", return_value=failed_probe) as mock_run:
            info1 = get_system_info()
            probe_count = mock_run.call_count
            info2 = get_system_info()
        self.assertIs(info1, info2)
        self.assertEqual(mock_run.call_count, probe_count)

    def test_execution_provider_info(self):
        from telemetry.system_info import get_execution_provider_info

        info = get_execution_provider_info()
        self.assertIn("available_providers", info)
        self.assertIsInstance(info["available_providers"], list)


class TestTelemetryEvents(_HermeticTelemetryTestCase):
    """Detailed-event methods are safe no-ops when telemetry is opted out."""

    def _opted_out_telemetry(self):
        from telemetry.telemetry import GenAITelemetry

        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        return GenAITelemetry()

    def test_log_model_build_when_disabled(self):
        """Ensure log_model_build doesn't crash when telemetry is disabled."""
        t = self._opted_out_telemetry()
        # Should not raise
        t.log_model_build(
            action="create_model",
            duration_ms=1234.5,
            success=True,
            model_name="test-model",
            model_type="llama",
            hidden_size=4096,
            num_layers=32,
            num_attn_heads=32,
            num_kv_heads=8,
            vocab_size=32000,
            context_length=4096,
            io_dtype="FLOAT16",
            quant_type="INT4",
            execution_provider="cuda",
        )

    def test_log_benchmark_when_disabled(self):
        """Ensure log_benchmark doesn't crash when telemetry is disabled."""
        t = self._opted_out_telemetry()
        t.log_benchmark(
            model_name="test-model",
            precision="fp16",
            backend="onnxruntime-genai",
            device="cuda",
            batch_size=1,
            prompt_length=128,
            tokens_generated=256,
            token_generation_latency_ms=5.0,
            token_generation_throughput=200.0,
            time_to_first_token_ms=50.0,
        )

    def test_log_model_load_when_disabled(self):
        t = self._opted_out_telemetry()
        t.log_model_load(
            model_name="test-model",
            model_type="phi3",
            execution_provider="cuda",
            total_load_time_ms=5000.0,
            num_sessions=3,
        )

    def test_log_inference_when_disabled(self):
        t = self._opted_out_telemetry()
        t.log_inference(
            model_name="test-model",
            time_to_first_token_ms=45.0,
            total_generation_time_ms=2000.0,
            total_tokens_generated=200,
            input_token_count=50,
        )

    def test_log_error_when_disabled(self):
        t = self._opted_out_telemetry()
        t.log_error(
            exception_type="RuntimeError",
            exception_message="Test error",
            action="test_action",
        )


class TestActionDecorator(_HermeticTelemetryTestCase):
    """Test the @action decorator and ActionContext."""

    def setUp(self):
        super().setUp()
        # Action helpers construct the singleton lazily; keep them opted out so
        # they emit no detailed events during the test.
        os.environ["ORT_DISABLE_TELEMETRY"] = "1"

    def test_action_decorator_success(self):
        from telemetry.telemetry_extensions import action

        @action
        def my_function():
            return 42

        result = my_function()
        self.assertEqual(result, 42)

    def test_action_decorator_exception(self):
        from telemetry.telemetry_extensions import action

        @action
        def my_failing_function():
            raise ValueError("test error")

        with self.assertRaises(ValueError):
            my_failing_function()

    def test_action_context_manager(self):
        from telemetry.telemetry_extensions import ActionContext

        with ActionContext("test_operation") as ctx:
            ctx.add_metadata("key", "value")
            result = 1 + 1

        self.assertEqual(result, 2)

    def test_action_context_manager_exception(self):
        from telemetry.telemetry_extensions import ActionContext

        with self.assertRaises(RuntimeError), ActionContext("test_operation"):
            raise RuntimeError("test error")


class TestSerializationHelper(unittest.TestCase):
    """Test Common Schema JSON serialization."""

    def test_serialize_basic_types(self):
        from telemetry.library.serialization import CommonSchemaJsonSerializationHelper as H

        self.assertIsNone(H.serialize_value(None))
        self.assertTrue(H.serialize_value(True))
        self.assertFalse(H.serialize_value(False))
        self.assertEqual(H.serialize_value(42), 42)
        self.assertEqual(H.serialize_value(3.14), 3.14)
        self.assertEqual(H.serialize_value("hello"), "hello")

    def test_serialize_list(self):
        from telemetry.library.serialization import CommonSchemaJsonSerializationHelper as H

        self.assertEqual(H.serialize_value([1, "two", 3.0]), [1, "two", 3.0])

    def test_serialize_dict(self):
        from telemetry.library.serialization import CommonSchemaJsonSerializationHelper as H

        result = H.serialize_value({"key": "value", "num": 42})
        self.assertEqual(result, {"key": "value", "num": 42})
        self.assertEqual(H.serialize_value({0: "zero", "": "skip"}), {"0": "zero"})
        self.assertEqual(H.serialize_value({False: "false"}), {"False": "false"})

    def test_create_event_envelope(self):
        from datetime import datetime, timezone

        from telemetry.library.serialization import CommonSchemaJsonSerializationHelper as H

        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        envelope = H.create_event_envelope(
            event_name="TestEvent",
            timestamp=ts,
            ikey="o:test-key",
            data={"key": "value"},
        )
        self.assertEqual(envelope["ver"], "4.0")
        self.assertEqual(envelope["name"], "TestEvent")
        self.assertEqual(envelope["iKey"], "o:test-key")
        self.assertEqual(envelope["data"], {"key": "value"})


class TestPayloadBuilder(unittest.TestCase):
    """Test payload builder."""

    def test_basic_build(self):
        from telemetry.library.payload_builder import PayloadBuilder

        builder = PayloadBuilder(max_size_bytes=-1, max_items=-1)
        builder.add(b'{"event":"test1"}')
        builder.add(b'{"event":"test2"}')
        payload = builder.build()
        self.assertEqual(payload, b'{"event":"test1"}\n{"event":"test2"}')

    def test_max_items_limit(self):
        from telemetry.library.payload_builder import PayloadBuilder

        builder = PayloadBuilder(max_size_bytes=-1, max_items=1)
        builder.add(b'{"event":"test1"}')
        self.assertFalse(builder.can_add(b'{"event":"test2"}'))

    def test_max_size_limit(self):
        from telemetry.library.payload_builder import PayloadBuilder

        builder = PayloadBuilder(max_size_bytes=20, max_items=-1)
        builder.add(b'{"event":"test1"}')
        self.assertFalse(builder.can_add(b'{"event":"test2"}'))

    def test_empty_build(self):
        from telemetry.library.payload_builder import PayloadBuilder

        builder = PayloadBuilder(max_size_bytes=-1, max_items=-1)
        self.assertEqual(builder.build(), b"")
        self.assertTrue(builder.is_empty)


class TestConnectionStringParser(unittest.TestCase):
    """Test connection string parsing."""

    def test_valid_connection_string(self):
        from telemetry.library.connection_string_parser import ConnectionStringParser

        parser = ConnectionStringParser("InstrumentationKey=abc-def-ghi")
        self.assertEqual(parser.instrumentation_key, "abc-def-ghi")

    def test_empty_connection_string(self):
        from telemetry.library.connection_string_parser import ConnectionStringParser

        with self.assertRaises(ValueError):
            ConnectionStringParser("")

    def test_missing_key(self):
        from telemetry.library.connection_string_parser import ConnectionStringParser

        with self.assertRaises(ValueError):
            ConnectionStringParser("SomeOtherKey=value")


class TestOfflineEventStore(unittest.TestCase):
    """Test the SQLite-backed durable event queue."""

    def _new_store(self, **kw):
        import tempfile

        import telemetry.offline_store as store_module

        db = os.path.join(tempfile.mkdtemp(), "genai_telemetry.db")
        store = store_module.OfflineEventStore(db, **kw)
        self.addCleanup(store.close)
        return store

    def test_empty_permission_path_is_ignored(self):
        import telemetry.offline_store as store_module

        with patch.object(store_module.os, "name", "posix"), patch.object(store_module.os, "chmod") as mock_chmod:
            store_module._chmod_best_effort("", 0o700)

        mock_chmod.assert_not_called()

    def test_closes_connection_when_initialization_fails(self):
        import telemetry.offline_store as store_module

        connection = MagicMock()
        connection.execute.side_effect = RuntimeError("pragma failed")
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(store_module.sqlite3, "connect", return_value=connection),
        ):
            store = store_module.OfflineEventStore(os.path.join(temp_dir, "failed.db"))

        self.assertFalse(store.is_open)
        connection.close.assert_called_once()

    def test_store_and_fifo_batch(self):
        s = self._new_store()
        for i in range(5):
            s.store(f'{{"e":{i}}}'.encode())
        self.assertEqual(s.count(), 5)
        batch = s.get_batch(3)
        self.assertEqual([p for _, p in batch], [b'{"e":0}', b'{"e":1}', b'{"e":2}'])

    def test_delete(self):
        s = self._new_store()
        s.store(b'{"a":1}')
        s.store(b'{"b":2}')
        ids = [i for i, _ in s.get_batch(10)]
        s.delete(ids[:1])
        self.assertEqual(s.count(), 1)

    def test_trim_to_watermark(self):
        s = self._new_store(max_records=8)
        for i in range(20):
            s.store(f'{{"i":{i}}}'.encode())
        # Over capacity trims back to ~75%.
        self.assertLessEqual(s.count(), 8)

    def test_empty_payload_rejected(self):
        s = self._new_store()
        self.assertFalse(s.store(b""))

    def test_user_version_stamped(self):
        import sqlite3

        import telemetry.offline_store as store_module

        s = self._new_store()
        conn = sqlite3.connect(s.db_path)
        try:
            v = conn.execute("PRAGMA user_version").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(v, store_module.SCHEMA_VERSION)

    @unittest.skipIf(os.name == "nt", "POSIX permissions")
    def test_store_uses_owner_only_permissions(self):
        s = self._new_store()
        self.assertEqual(stat.S_IMODE(os.stat(os.path.dirname(s.db_path)).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(s.db_path).st_mode), 0o600)


class TestProcessDrainLock(unittest.TestCase):
    """Test the cross-platform single-drainer advisory lock."""

    def _lock_path(self):
        import tempfile

        return os.path.join(tempfile.mkdtemp(), "telemetry.db.lock")

    def test_mutual_exclusion(self):
        from telemetry.process_lock import ProcessDrainLock

        path = self._lock_path()
        a = ProcessDrainLock(path)
        b = ProcessDrainLock(path)
        self.assertTrue(a.acquire())
        self.assertFalse(b.acquire())  # held by a
        a.release()
        self.assertTrue(b.acquire())  # released
        b.release()

    def test_reacquire_is_idempotent(self):
        from telemetry.process_lock import ProcessDrainLock

        a = ProcessDrainLock(self._lock_path())
        self.assertTrue(a.acquire())
        self.assertTrue(a.acquire())  # already held
        self.assertTrue(a.held)
        a.release()
        self.assertFalse(a.held)


class TestUploaderDrainLogic(unittest.TestCase):
    """Test the uploader's success/poison/transient handling (no real network)."""

    def _setup(self):
        import tempfile

        import telemetry.offline_store as store_module
        import telemetry.uploader as uploader_module

        db = os.path.join(tempfile.mkdtemp(), "genai_telemetry.db")
        store = store_module.OfflineEventStore(db)
        uploader = uploader_module.EventUploader(store, instrumentation_key="abc-def")
        self.addCleanup(store.close)
        self.addCleanup(uploader.close)
        return store, uploader

    def test_success_deletes(self):
        store, uploader = self._setup()
        store.store(b'{"ok":1}')
        uploader._transport.send = lambda *a, **k: (True, 204)
        delivered, left = uploader.drain_once()
        self.assertEqual((delivered, left), (1, 0))
        self.assertEqual(store.count(), 0)

    def test_request_drain_only_wakes_lock_holder(self):
        _, uploader = self._setup()
        uploader._wake = MagicMock()
        uploader._drain_lock = MagicMock(held=False)

        uploader.request_drain()
        uploader._wake.set.assert_not_called()

        uploader._drain_lock.held = True
        uploader.request_drain()
        uploader._wake.set.assert_called_once()

    def test_poison_4xx_dropped(self):
        store, uploader = self._setup()
        store.store(b'{"bad":1}')
        uploader._transport.send = lambda *a, **k: (False, 400)
        uploader.drain_once()
        self.assertEqual(store.count(), 0)  # dropped, not retried forever

    def test_transient_5xx_retained(self):
        store, uploader = self._setup()
        store.store(b'{"later":1}')
        uploader._transport.send = lambda *a, **k: (False, 503)
        delivered, left = uploader.drain_once()
        self.assertEqual((delivered, left), (0, 1))
        self.assertEqual(store.count(), 1)  # kept for retry

    def test_oversized_first_row_is_dropped(self):
        import telemetry.uploader as uploader_module

        store, uploader = self._setup()
        store.store(b"12345")
        uploader._transport.send = MagicMock()
        with patch.object(
            uploader_module.OneCollectorTransportOptions,
            "DEFAULT_MAX_PAYLOAD_SIZE_BYTES",
            4,
        ):
            delivered, left = uploader.drain_once()
        self.assertEqual((delivered, left), (1, 0))
        self.assertEqual(store.count(), 0)
        uploader._transport.send.assert_not_called()

    def test_flush_releases_process_lock(self):
        _, uploader = self._setup()
        uploader.flush(0.01)
        self.assertFalse(uploader._drain_lock.held)

    def test_flush_does_not_touch_lock_while_thread_is_alive(self):
        _, uploader = self._setup()
        uploader._thread = MagicMock()
        uploader._thread.is_alive.return_value = True
        uploader._drain_lock.acquire = MagicMock()
        uploader._drain_lock.release = MagicMock()

        uploader.flush(0.01)

        uploader._drain_lock.acquire.assert_not_called()
        uploader._drain_lock.release.assert_not_called()

    def test_stop_keeps_lock_when_thread_does_not_stop(self):
        _, uploader = self._setup()
        uploader.stop_loop = MagicMock(return_value=False)
        uploader._drain_lock.release = MagicMock()
        uploader.stop(0)
        uploader._drain_lock.release.assert_not_called()


class TestShutdownSafety(unittest.TestCase):
    def test_live_uploader_keeps_store_open(self):
        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        telemetry._heartbeat_thread = None
        telemetry._uploader = MagicMock()
        telemetry._uploader.stop_loop.return_value = False
        telemetry._store = MagicMock()

        telemetry.shutdown(0)

        telemetry._store.close.assert_not_called()

    def test_shutdown_uses_one_overall_budget(self):
        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        telemetry._heartbeat_thread = MagicMock()
        telemetry._heartbeat_thread.is_alive.return_value = False
        telemetry._uploader = MagicMock()
        telemetry._uploader.stop_loop.return_value = True
        telemetry._store = MagicMock()
        heartbeat = telemetry._heartbeat_thread
        uploader = telemetry._uploader

        with patch("telemetry.telemetry.time.monotonic", side_effect=[100.0, 101.0, 102.0, 103.0]):
            telemetry.shutdown(5.0)

        heartbeat.join.assert_called_once_with(4.0)
        uploader.stop_loop.assert_called_once_with(3.0)
        uploader.flush.assert_called_once_with(2.0)
        self.assertIsNone(telemetry._heartbeat_thread)
        self.assertIsNone(telemetry._uploader)
        self.assertIsNone(telemetry._store)
        self.assertFalse(telemetry._initialized)

    def test_enable_does_not_replace_live_uploader(self):
        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        telemetry._instrumentation_key = "abc-def"
        telemetry._enabled = False
        telemetry._store = MagicMock()
        telemetry._uploader = MagicMock(_send_timeout=10.0)
        telemetry._uploader.stop_loop.return_value = False
        old_uploader = telemetry._uploader

        with (
            patch("telemetry.telemetry._is_ci_environment", return_value=False),
            patch("telemetry.telemetry.EventUploader") as mock_new_uploader,
        ):
            telemetry.enable_telemetry()

        self.assertIs(telemetry._uploader, old_uploader)
        self.assertFalse(telemetry._enabled)
        old_uploader.stop_loop.assert_called_once_with(11.0)
        mock_new_uploader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
