# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

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
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the telemetry source to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "python", "py"))


class TestOptOut(unittest.TestCase):
    """Test telemetry opt-out mechanisms."""

    def setUp(self):
        """Reset singleton state between tests."""
        from telemetry.telemetry import GenAITelemetry
        GenAITelemetry._instance = None

    def tearDown(self):
        from telemetry.telemetry import GenAITelemetry
        GenAITelemetry._instance = None

    @patch.dict(os.environ, {"ORTGENAI_DISABLE_TELEMETRY": "1"})
    def test_env_var_disables_telemetry(self):
        from telemetry.telemetry import GenAITelemetry
        t = GenAITelemetry()
        self.assertFalse(t._enabled)

    @patch.dict(os.environ, {"CI": "true"}, clear=False)
    def test_ci_env_disables_telemetry(self):
        from telemetry.telemetry import GenAITelemetry
        # Remove ORTGENAI_DISABLE_TELEMETRY if set
        env = os.environ.copy()
        env.pop("ORTGENAI_DISABLE_TELEMETRY", None)
        with patch.dict(os.environ, env, clear=True):
            os.environ["CI"] = "true"
            t = GenAITelemetry()
            self.assertFalse(t._enabled)

    @patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}, clear=False)
    def test_github_actions_disables_telemetry(self):
        from telemetry.telemetry import GenAITelemetry
        env = os.environ.copy()
        env.pop("ORTGENAI_DISABLE_TELEMETRY", None)
        with patch.dict(os.environ, env, clear=True):
            os.environ["GITHUB_ACTIONS"] = "true"
            t = GenAITelemetry()
            self.assertFalse(t._enabled)

    def test_disable_enable_api(self):
        from telemetry.telemetry import GenAITelemetry
        t = GenAITelemetry()
        t.disable_telemetry()
        self.assertFalse(t._enabled)
        t.enable_telemetry()
        self.assertTrue(t._enabled)


class TestDeviceId(unittest.TestCase):
    """Test device ID generation."""

    def test_get_encrypted_device_id(self):
        from telemetry.deviceid import get_encrypted_device_id_and_status, DeviceIdStatus
        device_id, status = get_encrypted_device_id_and_status()
        # Should return a non-empty hex string (SHA256 = 64 hex chars)
        if status != DeviceIdStatus.FAILED:
            self.assertEqual(len(device_id), 64)
            # Should be uppercase hex
            self.assertTrue(all(c in "0123456789ABCDEF" for c in device_id))
        self.assertIn(status, list(DeviceIdStatus))

    def test_device_id_consistent(self):
        from telemetry.deviceid import get_encrypted_device_id_and_status
        id1, _ = get_encrypted_device_id_and_status()
        id2, _ = get_encrypted_device_id_and_status()
        self.assertEqual(id1, id2)


class TestSystemInfo(unittest.TestCase):
    """Test system information collection."""

    def test_get_system_info(self):
        from telemetry.system_info import get_system_info
        info = get_system_info()

        # Should have all expected keys
        expected_keys = [
            "os", "os_version", "os_arch", "processor_count",
            "python_version", "gpu_name", "total_memory_mb",
        ]
        for key in expected_keys:
            self.assertIn(key, info, f"Missing key: {key}")

        # OS should be a known value
        self.assertIn(info["os"], ["Windows", "Linux", "Darwin", ""])

        # Processor count should be positive
        self.assertGreater(info["processor_count"], 0)

        # Python version should match
        self.assertTrue(info["python_version"].startswith(str(sys.version_info.major)))

    def test_system_info_cached(self):
        from telemetry.system_info import get_system_info
        info1 = get_system_info()
        info2 = get_system_info()
        self.assertIs(info1, info2)

    def test_execution_provider_info(self):
        from telemetry.system_info import get_execution_provider_info
        info = get_execution_provider_info()
        self.assertIn("available_providers", info)
        self.assertIsInstance(info["available_providers"], list)


class TestTelemetryEvents(unittest.TestCase):
    """Test telemetry event emission (with telemetry disabled to avoid network calls)."""

    def setUp(self):
        from telemetry.telemetry import GenAITelemetry
        GenAITelemetry._instance = None

    def tearDown(self):
        from telemetry.telemetry import GenAITelemetry
        GenAITelemetry._instance = None

    @patch.dict(os.environ, {"ORTGENAI_DISABLE_TELEMETRY": "1"})
    def test_log_model_build_when_disabled(self):
        """Ensure log_model_build doesn't crash when telemetry is disabled."""
        from telemetry.telemetry import GenAITelemetry
        t = GenAITelemetry()
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

    @patch.dict(os.environ, {"ORTGENAI_DISABLE_TELEMETRY": "1"})
    def test_log_benchmark_when_disabled(self):
        """Ensure log_benchmark doesn't crash when telemetry is disabled."""
        from telemetry.telemetry import GenAITelemetry
        t = GenAITelemetry()
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

    @patch.dict(os.environ, {"ORTGENAI_DISABLE_TELEMETRY": "1"})
    def test_log_model_load_when_disabled(self):
        from telemetry.telemetry import GenAITelemetry
        t = GenAITelemetry()
        t.log_model_load(
            model_name="test-model",
            model_type="phi3",
            execution_provider="cuda",
            total_load_time_ms=5000.0,
            num_sessions=3,
        )

    @patch.dict(os.environ, {"ORTGENAI_DISABLE_TELEMETRY": "1"})
    def test_log_inference_when_disabled(self):
        from telemetry.telemetry import GenAITelemetry
        t = GenAITelemetry()
        t.log_inference(
            model_name="test-model",
            time_to_first_token_ms=45.0,
            total_generation_time_ms=2000.0,
            total_tokens_generated=200,
            input_token_count=50,
        )

    @patch.dict(os.environ, {"ORTGENAI_DISABLE_TELEMETRY": "1"})
    def test_log_error_when_disabled(self):
        from telemetry.telemetry import GenAITelemetry
        t = GenAITelemetry()
        t.log_error(
            exception_type="RuntimeError",
            exception_message="Test error",
            action="test_action",
        )


class TestActionDecorator(unittest.TestCase):
    """Test the @action decorator and ActionContext."""

    def setUp(self):
        from telemetry.telemetry import GenAITelemetry
        GenAITelemetry._instance = None

    def tearDown(self):
        from telemetry.telemetry import GenAITelemetry
        GenAITelemetry._instance = None

    @patch.dict(os.environ, {"ORTGENAI_DISABLE_TELEMETRY": "1"})
    def test_action_decorator_success(self):
        from telemetry.telemetry_extensions import action

        @action
        def my_function():
            return 42

        result = my_function()
        self.assertEqual(result, 42)

    @patch.dict(os.environ, {"ORTGENAI_DISABLE_TELEMETRY": "1"})
    def test_action_decorator_exception(self):
        from telemetry.telemetry_extensions import action

        @action
        def my_failing_function():
            raise ValueError("test error")

        with self.assertRaises(ValueError):
            my_failing_function()

    @patch.dict(os.environ, {"ORTGENAI_DISABLE_TELEMETRY": "1"})
    def test_action_context_manager(self):
        from telemetry.telemetry_extensions import ActionContext

        with ActionContext("test_operation") as ctx:
            ctx.add_metadata("key", "value")
            result = 1 + 1

        self.assertEqual(result, 2)

    @patch.dict(os.environ, {"ORTGENAI_DISABLE_TELEMETRY": "1"})
    def test_action_context_manager_exception(self):
        from telemetry.telemetry_extensions import ActionContext

        with self.assertRaises(RuntimeError):
            with ActionContext("test_operation") as ctx:
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
        from telemetry.offline_store import OfflineEventStore
        db = os.path.join(tempfile.mkdtemp(), "genai_telemetry.db")
        return OfflineEventStore(db, **kw)

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
        from telemetry.offline_store import SCHEMA_VERSION
        s = self._new_store()
        v = sqlite3.connect(s.db_path).execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(v, SCHEMA_VERSION)


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
        from telemetry.offline_store import OfflineEventStore
        from telemetry.uploader import EventUploader
        db = os.path.join(tempfile.mkdtemp(), "genai_telemetry.db")
        store = OfflineEventStore(db)
        uploader = EventUploader(store, instrumentation_key="abc-def")
        return store, uploader

    def test_success_deletes(self):
        store, uploader = self._setup()
        store.store(b'{"ok":1}')
        uploader._transport.send = lambda *a, **k: (True, 204)
        delivered, left = uploader.drain_once()
        self.assertEqual((delivered, left), (1, 0))
        self.assertEqual(store.count(), 0)

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


if __name__ == "__main__":
    unittest.main()
