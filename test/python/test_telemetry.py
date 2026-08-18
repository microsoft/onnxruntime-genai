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
from unittest.mock import MagicMock, call, mock_open, patch

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
        "CODEBUILD_BUILD_ID",
        "BUILDKITE",
        "TEAMCITY_VERSION",
        "APPVEYOR",
        "BITBUCKET_BUILD_NUMBER",
        "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI",
        "ORT_RUNNING_UNIT_TESTS",
    )

    def setUp(self):
        import tempfile

        import telemetry.deviceid as deviceid
        from telemetry.telemetry import GenAITelemetry

        GenAITelemetry._instance = None
        GenAITelemetry._process_disabled = False

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
        GenAITelemetry._process_disabled = False
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

    def test_ci_and_unit_test_signals_match_native_contract(self):
        from telemetry.telemetry import _CI_ENV_VARS, _UNIT_TEST_ENV_VAR, _is_ci_environment

        for signal in (*sorted(_CI_ENV_VARS), _UNIT_TEST_ENV_VAR):
            with self.subTest(signal=signal), patch.dict(os.environ, {signal: " true "}, clear=True):
                self.assertTrue(_is_ci_environment())

    def test_false_ci_values_do_not_disable_telemetry(self):
        from telemetry.telemetry import _is_ci_environment

        for value in ("", "0", "false", " no ", "OFF"):
            with self.subTest(value=value), patch.dict(os.environ, {"CI": value}, clear=True):
                self.assertFalse(_is_ci_environment())

    def test_environment_opt_out_sends_heartbeat_only(self):
        from telemetry.telemetry import GenAITelemetry

        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        t = GenAITelemetry()
        self.assertFalse(t._enabled)
        self.assertFalse(t.accepts_detailed_events)
        self.assertIsNone(t._store)
        self.assertIsNone(t._uploader)
        self.assertIsNotNone(t._heartbeat_thread)

        t.log_model_build(action="create_model", duration_ms=1.0, success=True)
        self._deliver()
        self.assertEqual(self._sent_event_names(), ["GenAIHeartbeat"])

    def test_public_disable_before_initialization_sends_heartbeat_only_once(self):
        from telemetry.telemetry import GenAITelemetry, disable_telemetry

        with (
            patch("telemetry.telemetry.OfflineEventStore") as store,
            patch("telemetry.telemetry.EventUploader") as uploader,
        ):
            disable_telemetry()
            telemetry = GenAITelemetry._instance
            self._join_heartbeat()
            disable_telemetry()

        store.assert_not_called()
        uploader.assert_not_called()
        self.assertIsNotNone(telemetry)
        self.assertFalse(telemetry._enabled)
        self.assertIsNone(telemetry._store)
        self.assertEqual(self._sent_event_names(), ["GenAIHeartbeat"])

    def test_enabled_records_heartbeat_and_events(self):
        import uuid

        from telemetry.telemetry import GenAITelemetry

        t = GenAITelemetry()
        session_guid = uuid.UUID(t._app_session_guid)
        self.assertEqual(session_guid.version, 4)
        self.assertEqual(session_guid.variant, uuid.RFC_4122)
        self.assertTrue(t._enabled)
        self.assertTrue(t.accepts_detailed_events)
        self.assertIsNotNone(t._store)
        t.log_model_build(action="create_model", duration_ms=1.0, success=True)
        self._deliver()
        names = self._sent_event_names()
        self.assertIn("GenAIHeartbeat", names)
        self.assertIn("GenAIModelBuild", names)

    def test_runtime_disable_is_process_latched(self):
        from telemetry.telemetry import GenAITelemetry

        t = GenAITelemetry()
        self._join_heartbeat()
        self.assertTrue(t._enabled)
        self.assertIsNotNone(t._store)
        t.disable_telemetry()
        t.log_model_build(action="create_model", duration_ms=1.0, success=True)
        t.disable_telemetry()
        self.assertFalse(t._enabled)
        self.assertTrue(t._telemetry_disabled)
        t.shutdown()

        self.assertIs(GenAITelemetry(), t)
        self.assertFalse(t._enabled)
        self.assertIsNone(t._store)
        self.assertEqual(self._sent_event_names(), ["GenAIHeartbeat"])

    def test_disable_during_heartbeat_collection_does_not_duplicate_it(self):
        import threading

        from telemetry.telemetry import GenAITelemetry

        release_heartbeat = threading.Event()

        def get_system_info():
            release_heartbeat.wait(5)
            return {}

        with patch(
            "telemetry.telemetry.get_system_info",
            side_effect=get_system_info,
        ):
            telemetry = GenAITelemetry()
            telemetry.disable_telemetry()
            release_heartbeat.set()
            telemetry._heartbeat_thread.join(5)
            telemetry.disable_telemetry()

        self.assertEqual(self._sent_event_names(), ["GenAIHeartbeat"])

    def test_heartbeat_is_attempted_directly_after_system_enrichment(self):
        import json
        import threading

        from telemetry.telemetry import GenAITelemetry

        release_enrichment = threading.Event()

        def get_system_info():
            release_enrichment.wait(5)
            return {"cpu_model": "test cpu"}

        with (
            patch("telemetry.telemetry.get_system_info", side_effect=get_system_info),
            patch("telemetry.telemetry.EventUploader.start"),
        ):
            telemetry = GenAITelemetry()
            self.assertEqual(telemetry._store.count(), 0)
            self.assertEqual(self.sent_payloads, [])
            release_enrichment.set()
            telemetry._heartbeat_thread.join(5)

        self.assertEqual(len(self.sent_payloads), 1)
        payload = json.loads(self.sent_payloads[0])
        self.assertEqual(payload["data"]["cpuModel"], "test cpu")

    def test_initialization_keeps_exporter_diagnostics_configurable(self):
        from telemetry.library.event_source import event_source
        from telemetry.telemetry import GenAITelemetry

        event_source.logger.disabled = False
        GenAITelemetry()
        self._join_heartbeat()

        self.assertFalse(event_source.logger.disabled)

    def test_env_opt_out_remains_latched_after_reinitialization(self):
        from telemetry.telemetry import GenAITelemetry

        os.environ["ORT_DISABLE_TELEMETRY"] = "true"
        t = GenAITelemetry()
        self._join_heartbeat()
        self.assertFalse(t._enabled)
        self.assertIsNone(t._store)
        t.shutdown()
        os.environ.pop("ORT_DISABLE_TELEMETRY")

        # Detailed-event suppression remains latched and does not create a
        # second Heartbeat after the environment variable is removed.
        self.assertIs(GenAITelemetry(), t)
        self.assertFalse(t._enabled)
        self.assertIsNone(t._store)
        self.assertIsNone(t._heartbeat_thread)
        self.assertEqual(self._sent_event_names(), ["GenAIHeartbeat"])

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

        heartbeat = MagicMock(ident=None)
        heartbeat.is_alive.return_value = False
        heartbeat.start.side_effect = RuntimeError("thread start failed")
        with (
            patch("telemetry.telemetry.OfflineEventStore") as store,
            patch("telemetry.telemetry.EventUploader") as uploader,
            patch("telemetry.telemetry.threading.Thread", return_value=heartbeat),
        ):
            telemetry = GenAITelemetry()

        store.assert_not_called()
        uploader.assert_not_called()
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


class TestTelemetryPackaging(unittest.TestCase):
    @staticmethod
    def _configured_packages(telemetry_enabled: bool):
        setup_template = Path(__file__).parents[2] / "src" / "python" / "setup.py.in"
        source = (
            setup_template.read_text(encoding="utf-8")
            .replace("@TARGET_NAME@", "onnxruntime-genai")
            .replace("@VERSION_INFO@", "0.0.0")
            .replace("@PYTHON_TELEMETRY_ENABLED@", str(telemetry_enabled))
        )
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="description")),
            patch("setuptools.setup") as setup,
        ):
            exec(compile(source, str(setup_template), "exec"), {"__name__": "__main__"})
        return setup.call_args.kwargs["packages"]

    def test_telemetry_enabled_wheel_includes_python_telemetry(self):
        packages = self._configured_packages(True)

        self.assertIn("onnxruntime_genai.telemetry", packages)

    def test_telemetry_disabled_wheel_excludes_python_telemetry(self):
        packages = self._configured_packages(False)

        self.assertNotIn("onnxruntime_genai.telemetry", packages)
        self.assertIn("onnxruntime_genai.models", packages)


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
        self.assertEqual(module.sanitize_model_identifier(Path("private/model.onnx")), "[path]")
        self.assertEqual(module.sanitize_model_identifier("/secret.onnx"), "[path]")
        self.assertEqual(module.sanitize_model_identifier("microsoft/phi-3-mini"), "microsoft/phi-3-mini")
        self.assertEqual(module.normalize_execution_provider("NvTensorRtRtx"), "trt-rtx")
        path_utils_module = sys.modules[module.sanitize_model_identifier.__module__]
        with patch.object(path_utils_module.os.path, "exists") as mock_exists:
            self.assertEqual(module.sanitize_model_identifier("microsoft/phi-3-mini"), "microsoft/phi-3-mini")
        mock_exists.assert_not_called()

    def test_emits_current_mean_and_median_metrics(self):
        import statistics

        for aggregation, aggregate in (("mean", statistics.fmean), ("median", statistics.median)):
            with self.subTest(aggregation=aggregation):
                module = self._load_helper()
                telemetry = MagicMock()
                module.get_telemetry = MagicMock(return_value=telemetry)
                tokenization_latency_ms = aggregate([1.0, 3.0, 20.0])
                prompt_latency_ms = aggregate([2.0, 4.0, 30.0])
                ttft_ms = aggregate([3.0, 7.0, 50.0])

                module.emit_benchmark_telemetry(
                    model_name=r"C:\Users\alice\models\model.onnx",
                    precision="fp16",
                    execution_provider="NvTensorRtRtx",
                    batch_size=2,
                    prompt_length=16,
                    tokens_generated=8,
                    tokenization_latency_ms=tokenization_latency_ms,
                    tokenization_throughput=100.0,
                    prompt_processing_latency_ms=prompt_latency_ms,
                    prompt_processing_throughput=200.0,
                    token_generation_latency_ms=5.0,
                    token_generation_throughput=300.0,
                    sampling_latency_ms=6.0,
                    sampling_throughput=400.0,
                    wall_clock_time_ms=7.0,
                    wall_clock_throughput=500.0,
                    time_to_first_token_ms=ttft_ms,
                    peak_memory_gpu_mb=8.0,
                    peak_memory_cpu_mb=9.0,
                    session_id=10,
                )

                attributes = telemetry.log_benchmark.call_args.kwargs
                self.assertEqual(attributes["model_name"], "[path]")
                self.assertEqual(attributes["device"], "trt-rtx")
                self.assertEqual(attributes["tokenization_latency_ms"], tokenization_latency_ms)
                self.assertEqual(attributes["prompt_processing_latency_ms"], prompt_latency_ms)
                self.assertEqual(attributes["time_to_first_token_ms"], ttft_ms)

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
        import telemetry.path_utils as path_utils
        from telemetry.path_utils import MAX_ERROR_MESSAGE_LENGTH
        from telemetry.telemetry import GenAITelemetry
        from telemetry.telemetry_extensions import log_error

        telemetry = MagicMock()
        with patch("telemetry.telemetry_extensions._get_telemetry", return_value=telemetry):
            log_error("RuntimeError", "x" * (MAX_ERROR_MESSAGE_LENGTH + 100))
            truncated = telemetry.log.call_args.args[1]["exceptionMessage"]
            self.assertEqual(len(truncated.encode("utf-8")), MAX_ERROR_MESSAGE_LENGTH)

            log_error("RuntimeError", "x" * (MAX_ERROR_MESSAGE_LENGTH - 1) + "€")
            multibyte = telemetry.log.call_args.args[1]["exceptionMessage"]
            self.assertEqual(multibyte, "x" * (MAX_ERROR_MESSAGE_LENGTH - 1))

        core = object.__new__(GenAITelemetry)
        core._enabled = True
        core._store = object()
        core._emit = MagicMock()
        core.log_error("RuntimeError", "x" * (MAX_ERROR_MESSAGE_LENGTH + 100))
        core_message = core._emit.call_args.args[1]["exceptionMessage"]
        self.assertEqual(len(core_message.encode("utf-8")), MAX_ERROR_MESSAGE_LENGTH)

        slash_heavy_url = "https://example.test/" + "segment/" * MAX_ERROR_MESSAGE_LENGTH
        with patch(
            "telemetry.path_utils._token_start",
            wraps=path_utils._token_start,
        ) as token_start:
            scrubbed = path_utils.scrub_error_message_for_telemetry(slash_heavy_url)

        self.assertEqual(len(scrubbed.encode("utf-8")), MAX_ERROR_MESSAGE_LENGTH)
        self.assertTrue(scrubbed.startswith("https://example.test/"))
        self.assertEqual(token_start.call_count, 1)

        slash_heavy_token = "a" + "/" * MAX_ERROR_MESSAGE_LENGTH + "b"
        with patch(
            "telemetry.path_utils._token_start",
            wraps=path_utils._token_start,
        ) as token_start:
            scrubbed = path_utils.scrub_error_message_for_telemetry(slash_heavy_token)

        self.assertEqual(len(scrubbed.encode("utf-8")), MAX_ERROR_MESSAGE_LENGTH)
        self.assertEqual(token_start.call_count, 1)

    def test_local_file_uris_are_redacted_but_remote_urls_are_preserved(self):
        from telemetry.path_utils import scrub_string_for_telemetry

        self.assertEqual(
            scrub_string_for_telemetry("Error at file:///home/alice/secret/model.onnx"),
            "Error at [path]",
        )
        self.assertEqual(
            scrub_string_for_telemetry("sqlite:////home/alice/private.db"),
            "[path]",
        )
        self.assertEqual(
            scrub_string_for_telemetry("https://example.test/model"),
            "https://example.test/model",
        )

    def test_format_exception_message_redacts_source_line_paths(self):
        from telemetry.telemetry import _format_exception_message

        try:
            raise RuntimeError(r"open C:\Users\alice\secret\weights.bin failed")
        except RuntimeError as exc:
            message = _format_exception_message(exc, exc.__traceback__)
        # The username must not survive in the source line or the message.
        self.assertNotIn("alice", message)
        self.assertIn("[path]", message)

    def test_format_exception_message_redacts_external_file_path(self):
        from telemetry.telemetry import _format_exception_message

        with patch(
            "telemetry.telemetry.traceback.format_exception",
            return_value=['  File "/home/Alice Smith/project/external.py", line 7, in run\n'],
        ):
            message = _format_exception_message(RuntimeError("boom"))

        self.assertEqual(message, 'File "[path]", line 7, in run')

    def test_format_exception_message_redacts_internal_file_path_and_keeps_context(self):
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
        self.assertEqual(attributes["exceptionMessage"], "missing [path]")

    def test_core_event_methods_redact_model_names(self):
        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        telemetry._enabled = True
        telemetry._store = object()
        telemetry._next_model_session_id = 1
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
            self.assertEqual(telemetry._emit.call_args.args[1]["modelName"], "[path]")

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

        extra_options = telemetry._emit.call_args.args[1]["extraOptions"]
        self.assertEqual(extra_options["adapter_path"], "[path]")
        self.assertEqual(extra_options["[path]"]["paths"], ["[path]"])
        self.assertEqual(extra_options["batch_size"], 4)

    def test_generic_log_scrubs_all_serializable_fallbacks(self):
        import json

        from telemetry.telemetry import GenAITelemetry

        class PrivateValue:
            def __str__(self):
                return r"C:\Users\alice\private\model.onnx"

        telemetry = object.__new__(GenAITelemetry)
        telemetry._enabled = True
        telemetry._store = MagicMock()
        telemetry._uploader = None
        telemetry._app_name = "onnxruntime-genai"
        telemetry._app_version = "test"
        telemetry._app_session_guid = "session"
        telemetry._envelope_ikey = "o:test"

        telemetry.log(
            "GenAIAction",
            {
                PrivateValue(): {Path("private/model.onnx"), PrivateValue()},
                "rooted": "/secret.onnx",
                "repository": "microsoft/phi-3-mini",
            },
        )

        payload = telemetry._store.store_with_id.call_args.args[0]
        serialized = json.loads(payload)
        serialized_text = payload.decode("utf-8")
        self.assertNotIn("alice", serialized_text.lower())
        self.assertNotIn("secret.onnx", serialized_text)
        self.assertEqual(serialized["data"]["repository"], "microsoft/phi-3-mini")
        self.assertIn("[path]", serialized["data"])
        self.assertTrue(all(value == "[path]" for value in serialized["data"]["[path]"]))

    def test_non_finite_event_is_rejected_without_affecting_next_event(self):
        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        telemetry._enabled = True
        telemetry._store = MagicMock()
        telemetry._uploader = None
        telemetry._app_name = "onnxruntime-genai"
        telemetry._app_version = "test"
        telemetry._app_session_guid = "session"
        telemetry._envelope_ikey = "o:test"
        telemetry._next_model_session_id = 1

        telemetry.log_benchmark(tokenization_latency_ms=float("nan"))
        telemetry.log_benchmark(tokenization_latency_ms=1.0)

        self.assertEqual(telemetry._store.store_with_id.call_count, 1)

    def test_embedded_rooted_paths_are_scrubbed_in_final_payloads(self):
        import json

        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        telemetry._enabled = True
        telemetry._store = MagicMock()
        telemetry._uploader = None
        telemetry._app_name = "onnxruntime-genai"
        telemetry._app_version = "test"
        telemetry._app_session_guid = "session"
        telemetry._envelope_ikey = "o:test"

        telemetry.log(
            "GenAIAction",
            {
                "message": "missing /Alice_resume.pdf",
                "assignment": "file=/Carol_resume.pdf",
                "parenthesized": "missing(/Dana_resume.pdf)",
                "ratio": "n/a",
                "url": "https://example.com/model",
            },
        )
        telemetry.log_error("FileNotFoundError", "missing /Bob_resume.pdf")

        payloads = [
            json.loads(call.args[0])["data"]
            for call in telemetry._store.store_with_id.call_args_list
        ]
        self.assertEqual(payloads[0]["message"], "missing [path]")
        self.assertEqual(payloads[0]["assignment"], "file=[path]")
        self.assertEqual(payloads[0]["parenthesized"], "missing([path]")
        self.assertEqual(payloads[0]["ratio"], "n/a")
        self.assertEqual(payloads[0]["url"], "https://example.com/model")
        self.assertEqual(payloads[1]["exceptionMessage"], "missing [path]")

    def test_exception_message_keeps_error_specific_size_limit(self):
        import json

        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        telemetry._enabled = True
        telemetry._store = MagicMock()
        telemetry._uploader = None
        telemetry._app_name = "onnxruntime-genai"
        telemetry._app_version = "test"
        telemetry._app_session_guid = "session"
        telemetry._envelope_ikey = "o:test"
        message = "x" * 4096

        telemetry.log_error("RuntimeError", message)

        payload = json.loads(telemetry._store.store_with_id.call_args.args[0])
        self.assertEqual(payload["data"]["exceptionMessage"], message)

    def test_non_string_exception_message_uses_general_scrubbing(self):
        import json

        from telemetry.telemetry import GenAITelemetry

        class Unstringifiable:
            def __str__(self):
                raise RuntimeError("cannot stringify")

        telemetry = object.__new__(GenAITelemetry)
        telemetry._enabled = True
        telemetry._store = MagicMock()
        telemetry._uploader = None
        telemetry._app_name = "onnxruntime-genai"
        telemetry._app_version = "test"
        telemetry._app_session_guid = "session"
        telemetry._envelope_ikey = "o:test"

        telemetry.log("GenAIAction", {"exceptionMessage": Unstringifiable()})

        payload = json.loads(telemetry._store.store_with_id.call_args.args[0])
        self.assertEqual(
            payload["data"]["exceptionMessage"],
            "[unsupported:Unstringifiable]",
        )

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

        self.assertEqual(action_attributes["invokedFrom"], "test")
        self.assertEqual(action_attributes["actionName"], "work")
        self.assertEqual(action_attributes["durationMs"], 1.0)
        self.assertEqual(error_attributes["exceptionType"], "RuntimeError")
        self.assertEqual(error_attributes["exceptionMessage"], "boom")
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

    def test_get_hashed_device_id(self):
        import telemetry.deviceid as deviceid

        device_id, status = deviceid.get_hashed_device_id_and_status()
        # Shared product-salted FNV-1a with the custom-device-id prefix.
        if status != deviceid.DeviceIdStatus.FAILED:
            self.assertEqual(len(device_id), 18)
            self.assertTrue(device_id.startswith("c:"))
            self.assertTrue(all(c in "0123456789abcdef" for c in device_id[2:]))
            raw_id = deviceid._device_id_state["device_id"]
            expected = deviceid._fnv1a_hex(deviceid._DEVICE_ID_HASH_SALT + raw_id)
            self.assertEqual(device_id, f"c:{expected}")
        self.assertIn(status, list(deviceid.DeviceIdStatus))

    def test_device_id_hash_matches_native_known_vector(self):
        import telemetry.deviceid as deviceid

        raw_id = "00000000-0000-4000-8000-000000000000"
        self.assertEqual(
            deviceid._fnv1a_hex(deviceid._DEVICE_ID_HASH_SALT + raw_id),
            "912603c603e23b6b",
        )

    def test_noncanonical_device_ids_are_repaired(self):
        import telemetry.deviceid as deviceid

        device_id_path = Path(self._tmpdir.name) / "deviceid"
        for stored_value in (
            "00000000000040008000000000000000",
            "{00000000-0000-4000-8000-000000000000}",
        ):
            with self.subTest(stored_value=stored_value):
                device_id_path.write_text(stored_value, encoding="utf-8")
                deviceid._device_id_state.update(
                    {"device_id": None, "status": deviceid.DeviceIdStatus.NEW}
                )

                repaired = deviceid.get_device_id()

                self.assertTrue(deviceid._is_valid_device_id(repaired))
                self.assertNotEqual(repaired, stored_value)
                self.assertEqual(device_id_path.read_text(encoding="utf-8"), repaired)
                self.assertEqual(
                    deviceid._device_id_state["status"],
                    deviceid.DeviceIdStatus.CORRUPTED,
                )

    def test_non_utf8_device_id_is_repaired(self):
        import telemetry.deviceid as deviceid

        device_id_path = Path(self._tmpdir.name) / "deviceid"
        device_id_path.write_bytes(b"\xff\xfe")

        repaired = deviceid.get_device_id()

        self.assertTrue(deviceid._is_valid_device_id(repaired))
        self.assertEqual(device_id_path.read_text(encoding="utf-8"), repaired)
        self.assertEqual(
            deviceid._device_id_state["status"],
            deviceid.DeviceIdStatus.CORRUPTED,
        )

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

    def test_relative_posix_storage_environment_uses_absolute_home_fallback(self):
        self._get_telemetry_base_dir.cache_clear()
        try:
            with (
                patch("telemetry.deviceid.platform.system", return_value="Linux"),
                patch.dict(
                    os.environ,
                    {"XDG_CACHE_HOME": "relative-cache", "HOME": "relative-home"},
                    clear=False,
                ),
                patch(
                    "telemetry.deviceid.os.getuid",
                    side_effect=AttributeError,
                    create=True,
                ),
                patch.object(Path, "home", return_value=Path(self._tmpdir.name)),
            ):
                path = self._get_telemetry_base_dir()

            self.assertTrue(path.is_absolute())
            self.assertEqual(
                path,
                Path(self._tmpdir.name)
                / ".cache"
                / "Microsoft"
                / "DeveloperTools"
                / ".onnxruntime",
            )
        finally:
            self._get_telemetry_base_dir.cache_clear()

    def test_device_id_consistent(self):
        import telemetry.deviceid as deviceid

        id1, _ = deviceid.get_hashed_device_id_and_status()
        id2, _ = deviceid.get_hashed_device_id_and_status()
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

    def test_windows_store_rejects_wrong_registry_type(self):
        import telemetry.deviceid as deviceid

        winreg = MagicMock(
            HKEY_CURRENT_USER=object(),
            KEY_READ=0x0001,
            KEY_WOW64_64KEY=0x0100,
            REG_SZ=1,
            REG_BINARY=3,
        )
        winreg.QueryValueEx.return_value = (b"not-a-string", winreg.REG_BINARY)

        with (
            patch.dict(sys.modules, {"winreg": winreg}),
            self.assertRaises(ValueError),
        ):
            _ = deviceid._WindowsStore().retrieve_id

    def test_concurrent_processes_publish_one_device_id(self):
        import subprocess

        source_path = str(Path(__file__).parents[2] / "src" / "python" / "py")
        script = (
            "import platform; "
            "import telemetry.deviceid as d; "
            "platform.system=lambda:'Linux'; "
            "d.get_telemetry_base_dir.cache_clear(); "
            "print(d.get_device_id())"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = source_path + os.pathsep + env.get("PYTHONPATH", "")
        env["XDG_CACHE_HOME"] = self._tmpdir.name

        def run_processes():
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                for _ in range(6)
            ]
            results = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=15)
                self.assertEqual(process.returncode, 0, stderr)
                results.append(stdout.strip())
            self.assertEqual(len(set(results)), 1)
            self.assertTrue(results[0])
            return results[0]

        run_processes()
        device_id_path = (
            Path(self._tmpdir.name)
            / "Microsoft"
            / "DeveloperTools"
            / ".onnxruntime"
            / "deviceid"
        )
        device_id_path.write_text("corrupted", encoding="utf-8")
        repaired_id = run_processes()
        self.assertEqual(device_id_path.read_text(encoding="utf-8"), repaired_id)


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
        self.assertNotIn("process_name", info)
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

    def test_common_and_event_fields_use_onnx_style_names(self):
        import json

        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        telemetry._enabled = True
        telemetry._store = MagicMock()
        telemetry._uploader = None
        telemetry._app_name = "onnxruntime-genai"
        telemetry._app_version = "1.0"
        telemetry._app_session_guid = "123e4567-e89b-42d3-a456-426614174000"
        telemetry._envelope_ikey = "o:test"
        telemetry._emit("TestEvent", {"durationMs": 1.0})

        data = json.loads(telemetry._store.store_with_id.call_args.args[0])["data"]
        self.assertEqual(data["appName"], "onnxruntime-genai")
        self.assertEqual(data["LibraryVersion"], "1.0")
        self.assertEqual(data["AppSessionGuid"], telemetry._app_session_guid)
        self.assertNotIn("appVersion", data)
        self.assertNotIn("appSessionGuid", data)
        self.assertEqual(data["durationMs"], 1.0)
        self.assertFalse(any("_" in key for key in data))

    def test_model_session_ids_are_monotonic_and_correlate_model_events(self):
        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        telemetry._enabled = True
        telemetry._store = object()
        telemetry._next_model_session_id = 1
        telemetry._emit = MagicMock()

        session_id = telemetry.allocate_model_session_id()
        self.assertEqual(session_id, 1)
        self.assertEqual(telemetry.allocate_model_session_id(), 2)

        emitters = (
            lambda: telemetry.log_model_load(session_id=session_id),
            lambda: telemetry.log_benchmark(session_id=session_id),
            lambda: telemetry.log_inference(session_id=session_id),
            lambda: telemetry.log_error("RuntimeError", "boom", session_id=session_id),
        )
        for emit in emitters:
            with self.subTest(emit=emit):
                telemetry._emit.reset_mock()
                emit()
                self.assertEqual(telemetry._emit.call_args.args[1]["sessionId"], session_id)

    def test_heartbeat_uses_process_scope_session_id_and_omits_process_name(self):
        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        with (
            patch(
                "telemetry.telemetry.get_hashed_device_id_and_status",
                return_value=("c:device", MagicMock(value="existing")),
            ),
            patch("telemetry.telemetry.get_system_info", return_value={"process_name": "python.exe"}),
            patch("telemetry.telemetry.get_execution_provider_info", return_value={}),
        ):
            attributes = telemetry._build_heartbeat_attributes()

        self.assertEqual(attributes["sessionId"], 0)
        self.assertNotIn("processName", attributes)

    def test_all_core_event_fields_are_camel_case(self):
        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        telemetry._enabled = True
        telemetry._store = object()
        telemetry._next_model_session_id = 1
        telemetry._emit = MagicMock()
        emitters = (
            lambda: telemetry.log_model_build("build", 1.0, True),
            telemetry.log_benchmark,
            telemetry.log_model_load,
            telemetry.log_inference,
            lambda: telemetry.log_error("RuntimeError", "boom"),
        )

        for emit in emitters:
            telemetry._emit.reset_mock()
            emit()
            attributes = telemetry._emit.call_args.args[1]
            self.assertFalse(any("_" in key for key in attributes), attributes)


class TestForkLifecycle(_HermeticTelemetryTestCase):
    def test_after_fork_discards_inherited_resources(self):
        from telemetry.telemetry import GenAITelemetry

        old_instance = object.__new__(GenAITelemetry)
        uploader = MagicMock()
        store = MagicMock()
        old_instance._uploader = uploader
        old_instance._store = store
        old_instance._enabled = True
        old_instance._initialized = True
        GenAITelemetry._instance = old_instance

        GenAITelemetry._after_fork_child()

        uploader.discard_after_fork.assert_called_once()
        store.discard_after_fork.assert_called_once()
        self.assertFalse(old_instance._enabled)
        self.assertFalse(old_instance._initialized)
        self.assertIsNone(GenAITelemetry._instance)

    def test_after_fork_preserves_runtime_opt_out(self):
        from telemetry.telemetry import GenAITelemetry

        old_instance = object.__new__(GenAITelemetry)
        old_instance._uploader = MagicMock()
        old_instance._store = MagicMock()
        old_instance._enabled = False
        old_instance._initialized = True
        old_instance._telemetry_disabled = True
        GenAITelemetry._instance = old_instance

        GenAITelemetry._after_fork_child()
        child = GenAITelemetry()

        self.assertIs(child, old_instance)
        self.assertFalse(child._enabled)
        self.assertTrue(child._telemetry_disabled)
        self.assertIsNone(child._store)
        self.assertIsNotNone(child._heartbeat_thread)

    @unittest.skipUnless(hasattr(os, "fork"), "POSIX fork lifecycle")
    def test_forked_child_reinitializes_process_state(self):
        from telemetry.telemetry import GenAITelemetry

        parent = GenAITelemetry()
        self._join_heartbeat()
        parent_guid = parent._app_session_guid
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                child = GenAITelemetry()
                result = "|".join(
                    (
                        str(child is not parent),
                        str(child._app_session_guid != parent_guid),
                        str(child._uploader is not None and child._uploader._thread is not None),
                    )
                )
                child.shutdown(0)
                os.write(write_fd, result.encode("ascii"))
            finally:
                os.close(write_fd)
                os._exit(0)

        os.close(write_fd)
        result = os.read(read_fd, 128).decode("ascii")
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)
        self.assertEqual(status, 0)
        self.assertEqual(result, "True|True|True")

    @unittest.skipUnless(hasattr(os, "fork"), "POSIX fork lifecycle")
    def test_forked_child_keeps_runtime_opt_out(self):
        from telemetry.telemetry import GenAITelemetry

        parent = GenAITelemetry()
        self._join_heartbeat()
        parent.disable_telemetry()
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                child = GenAITelemetry()
                result = "|".join(
                    (
                        str(child._enabled),
                        str(child._store is None),
                        str(child._heartbeat_thread is not None),
                    )
                )
                os.write(write_fd, result.encode("ascii"))
            finally:
                os.close(write_fd)
                os._exit(0)

        os.close(write_fd)
        result = os.read(read_fd, 128).decode("ascii")
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)
        self.assertEqual(status, 0)
        self.assertEqual(result, "False|True|True")


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


class TestHttpTransport(unittest.TestCase):
    def test_success_does_not_read_response_body(self):
        import urllib.request

        from telemetry.library.transport import HttpJsonPostTransport

        response = MagicMock(status=204)
        response.getcode.return_value = 204
        context = MagicMock()
        context.__enter__.return_value = response
        request = urllib.request.Request("https://example.invalid", data=b"{}", method="POST")

        with patch("telemetry.library.transport.urllib.request.urlopen", return_value=context) as urlopen:
            self.assertEqual(HttpJsonPostTransport._do_request(request, 1.0), (True, 204))

        urlopen.assert_called_once()
        response.read.assert_not_called()

    def test_retry_shares_one_timeout_budget(self):
        import urllib.error
        import urllib.request

        from telemetry.library.transport import HttpJsonPostTransport

        request = urllib.request.Request("https://example.invalid", data=b"{}", method="POST")
        with (
            patch(
                "telemetry.library.transport.urllib.request.urlopen",
                side_effect=urllib.error.URLError("offline"),
            ) as urlopen,
            patch("telemetry.library.transport.time.monotonic", side_effect=[10.0, 10.0, 10.75]),
        ):
            self.assertEqual(HttpJsonPostTransport._do_request(request, 1.0), (False, None))

        self.assertEqual(urlopen.call_count, 2)
        self.assertAlmostEqual(urlopen.call_args_list[0].kwargs["timeout"], 1.0)
        self.assertAlmostEqual(urlopen.call_args_list[1].kwargs["timeout"], 0.25)

    def test_all_server_errors_are_retryable(self):
        from telemetry.library.transport import HttpJsonPostTransport

        self.assertTrue(HttpJsonPostTransport.is_retryable(507))
        self.assertTrue(HttpJsonPostTransport.is_retryable(520))


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

    def test_deferred_row_is_durable_but_unavailable_until_released(self):
        s = self._new_store()
        with patch("telemetry.offline_store.time.time", return_value=100.0):
            row_id = s.store_with_id(b'{"minimal":1}', available_after_seconds=60.0)
            s.store(b'{"ready":1}')
            self.assertEqual(s.count(), 2)
            self.assertEqual([payload for _, payload in s.get_batch(10)], [b'{"ready":1}'])
            self.assertTrue(s.replace(row_id, b'{"enriched":1}'))
            self.assertEqual(
                [payload for _, payload in s.get_batch(10)],
                [b'{"enriched":1}', b'{"ready":1}'],
            )

    def test_version_one_store_is_migrated_without_losing_events(self):
        import sqlite3
        import tempfile

        import telemetry.offline_store as store_module

        db = os.path.join(tempfile.mkdtemp(), "genai_telemetry.db")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, payload BLOB NOT NULL)")
        conn.execute("INSERT INTO events (payload) VALUES (?)", (sqlite3.Binary(b'{"legacy":1}'),))
        conn.execute("PRAGMA user_version=1")
        conn.commit()
        conn.close()

        store = store_module.OfflineEventStore(db)
        self.addCleanup(store.close)

        self.assertTrue(store.is_open)
        self.assertEqual(store.get_batch(1)[0][1], b'{"legacy":1}')
        self.assertEqual(
            {row[1] for row in store._conn.execute("PRAGMA table_info(events)").fetchall()},
            {"id", "payload", "available_at"},
        )

    def test_store_connection_is_closed_before_fork_and_reopened_lazily(self):
        store = self._new_store()
        store.prepare_for_fork()

        self.assertIsNone(store._conn)
        self.assertTrue(store.store(b'{"after_fork":1}'))
        self.assertTrue(store.is_open)

        store.discard_after_fork()
        self.assertIsNone(store._conn)

    def test_failed_lazy_reconnect_is_rate_limited_and_retried(self):
        store = self._new_store()
        store.prepare_for_fork()

        with (
            patch.object(store, "_initialize") as initialize,
            patch(
                "telemetry.offline_store.time.monotonic",
                side_effect=[100.0, 101.0, 106.0],
            ),
        ):
            self.assertFalse(store.store(b'{"attempt":1}'))
            self.assertFalse(store.store(b'{"attempt":2}'))
            self.assertFalse(store.store(b'{"attempt":3}'))

        self.assertEqual(initialize.call_count, 2)

    def test_delete(self):
        s = self._new_store()
        s.store(b'{"a":1}')
        s.store(b'{"b":2}')
        ids = [i for i, _ in s.get_batch(10)]
        self.assertTrue(s.delete(ids[:1]))
        self.assertEqual(s.count(), 1)

    def test_failed_delete_rolls_back(self):
        s = self._new_store()
        s.store(b'{"a":1}')
        row_id = s.get_batch(1)[0][0]
        s._conn.execute(
            "CREATE TRIGGER fail_delete BEFORE DELETE ON events "
            "BEGIN SELECT RAISE(FAIL, 'blocked'); END"
        )
        s._conn.commit()

        self.assertFalse(s.delete([row_id]))
        self.assertEqual(s.count(), 1)
        s._conn.execute("DROP TRIGGER fail_delete")
        s._conn.commit()
        self.assertTrue(s.delete([row_id]))
        self.assertEqual(s.count(), 0)

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

    def test_delete_failure_retries_without_reposting(self):
        store, uploader = self._setup()
        store.store(b'{"ok":1}')
        uploader._transport.send = MagicMock(return_value=(True, 204))
        original_delete = store.delete
        delete_attempts = 0

        def flaky_delete(ids):
            nonlocal delete_attempts
            delete_attempts += 1
            return False if delete_attempts == 1 else original_delete(ids)

        store.delete = flaky_delete

        self.assertEqual(uploader.drain_once(), (0, 1))
        self.assertEqual(store.count(), 1)
        self.assertEqual(uploader.drain_once(), (1, 0))
        self.assertEqual(store.count(), 0)
        uploader._transport.send.assert_called_once()

    def test_drain_uses_only_remaining_deadline(self):
        store, uploader = self._setup()
        store.store(b'{"ok":1}')
        uploader._transport.send = MagicMock(return_value=(False, None))

        with patch("telemetry.uploader.time.monotonic", return_value=100.75):
            self.assertEqual(uploader.drain_once(deadline=101.0), (0, 1))

        self.assertAlmostEqual(uploader._transport.send.call_args.args[1], 0.25)

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

    def test_uncommon_5xx_responses_are_retained(self):
        for status in (507, 520):
            with self.subTest(status=status):
                store, uploader = self._setup()
                store.store(b'{"later":1}')
                uploader._transport.send = lambda *a, status=status, **k: (False, status)
                self.assertEqual(uploader.drain_once(), (0, 1))
                self.assertEqual(store.count(), 1)

    def test_content_rejection_isolates_bad_event(self):
        store, uploader = self._setup()
        store.store(b'{"bad":1}')
        store.store(b'{"valid":1}')

        def send(payload, timeout, item_count=1):
            if item_count > 1 or b'"bad"' in payload:
                return (False, 400)
            return (True, 204)

        uploader._transport.send = MagicMock(side_effect=send)

        self.assertEqual(uploader.drain_once(), (0, 2))
        self.assertEqual(uploader.drain_once(), (1, 0))
        self.assertEqual(uploader.drain_once(), (1, 0))
        self.assertEqual(store.count(), 0)
        self.assertEqual(uploader._transport.send.call_count, 3)

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

    def test_runtime_disable_keeps_live_uploader_until_it_stops(self):
        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        telemetry._telemetry_disabled = False
        telemetry._enabled = True
        telemetry._store = MagicMock()
        telemetry._uploader = MagicMock()
        telemetry._uploader.stop_loop.return_value = False
        old_uploader = telemetry._uploader

        telemetry.disable_telemetry()

        self.assertIs(telemetry._uploader, old_uploader)
        self.assertFalse(telemetry._enabled)
        self.assertTrue(telemetry._telemetry_disabled)
        old_uploader.signal_stop.assert_called_once()
        old_uploader.stop_loop.assert_called_once_with(0)

    def test_shutdown_does_not_flush_after_runtime_disable(self):
        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        telemetry._initialized = True
        telemetry._telemetry_disabled = True
        telemetry._heartbeat_thread = None
        telemetry._uploader = MagicMock()
        telemetry._uploader.stop_loop.return_value = True
        telemetry._store = MagicMock()
        uploader = telemetry._uploader

        telemetry.shutdown(5.0)

        uploader.stop_loop.assert_called_once()
        uploader.flush.assert_not_called()
        uploader.close.assert_called_once()
        self.assertIsNone(telemetry._store)

    def test_shutdown_does_not_wait_for_opt_out_heartbeat(self):
        from telemetry.telemetry import GenAITelemetry

        telemetry = object.__new__(GenAITelemetry)
        telemetry._initialized = True
        telemetry._telemetry_disabled = True
        telemetry._heartbeat_thread = MagicMock()
        telemetry._heartbeat_thread.is_alive.return_value = True
        telemetry._uploader = None
        telemetry._store = None
        heartbeat = telemetry._heartbeat_thread

        telemetry.shutdown(5.0)

        heartbeat.join.assert_not_called()


if __name__ == "__main__":
    unittest.main()
