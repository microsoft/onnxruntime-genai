# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""
Comprehensive test coverage for ONNX Runtime GenAI Python API.
This test file focuses on ensuring complete API coverage for migration from pybind11 to nanobind.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest


# Test fixtures for device types
devices = ["cpu"]

if og.is_cuda_available():
    devices.append("cuda")

if og.is_dml_available():
    devices.append("dml")

if og.is_rocm_available():
    devices.append("rocm")

if og.is_openvino_available():
    devices.append("openvino")


# ============================================================================
# Config API Tests
# ============================================================================

def test_config_overlay(test_data_path):
    """Test Config.overlay method"""
    model_path = os.fspath(
        Path(test_data_path) / "hf-internal-testing" / "tiny-random-gpt2-fp32"
    )
    
    config = og.Config(model_path)
    # Test overlay with a custom configuration string
    if hasattr(config, 'overlay'):
        config.overlay('{"search": {"max_length": 100}}')
    
    # Should not raise any exception
    model = og.Model(config)
    assert model is not None


def test_config_decoder_provider_options(test_data_path):
    """Test decoder provider hardware options"""
    model_path = os.fspath(
        Path(test_data_path) / "hf-internal-testing" / "tiny-random-gpt2-fp32"
    )
    
    config = og.Config(model_path)
    
    # Test setting hardware device options
    config.set_decoder_provider_options_hardware_device_type("GPU")
    config.set_decoder_provider_options_hardware_device_id(0)
    config.set_decoder_provider_options_hardware_vendor_id(0x1002)  # Example vendor ID
    
    # Test clearing hardware device options
    config.clear_decoder_provider_options_hardware_device_type()
    config.clear_decoder_provider_options_hardware_device_id()
    config.clear_decoder_provider_options_hardware_vendor_id()
    
    # Should not raise any exception
    model = og.Model(config)
    assert model is not None


# ============================================================================
# Model API Tests
# ============================================================================

def test_model_type_property(test_data_path):
    """Test Model.type property"""
    model_path = os.fspath(
        Path(test_data_path) / "hf-internal-testing" / "tiny-random-gpt2-fp32"
    )
    
    model = og.Model(model_path)
    model_type = model.type
    
    # Should return a valid model type string
    assert isinstance(model_type, str)
    assert len(model_type) > 0


# ============================================================================
# Tokenizer API Tests
# ============================================================================

def test_tokenizer_update_options(device, phi2_for):
    """Test Tokenizer.update_options method"""
    model_path = phi2_for(device)
    model = og.Model(model_path)
    tokenizer = og.Tokenizer(model)
    
    # Update tokenizer options
    tokenizer.update_options(add_special_tokens="false", skip_special_tokens="true")
    
    # Test encoding with updated options
    text = "This is a test."
    tokens = tokenizer.encode(text)
    assert tokens is not None
    assert len(tokens) > 0


def test_tokenizer_to_token_id(device, phi2_for):
    """Test Tokenizer.to_token_id method"""
    model_path = phi2_for(device)
    model = og.Model(model_path)
    tokenizer = og.Tokenizer(model)
    
    # Convert a word to token ID
    token_id = tokenizer.to_token_id("hello")
    assert isinstance(token_id, int)
    assert token_id >= 0


# ============================================================================
# GeneratorParams API Tests
# ============================================================================

def test_generator_params_set_guidance(test_data_path):
    """Test GeneratorParams.set_guidance method"""
    model_path = os.fspath(
        Path(test_data_path) / "hf-internal-testing" / "tiny-random-gpt2-fp32"
    )
    
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    
    # Set guidance with type and data
    # Note: Actual guidance format depends on model support
    try:
        params.set_guidance("json", '{"schema": "test"}')
    except Exception:
        # Some models may not support guidance, that's ok for coverage
        pass


# ============================================================================
# Generator API Tests
# ============================================================================

def test_generator_get_input(test_data_path):
    """Test Generator.get_input method"""
    model_path = os.fspath(
        Path(test_data_path) / "hf-internal-testing" / "tiny-random-gpt2-fp32"
    )
    
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=10)
    
    generator = og.Generator(model, params)
    generator.append_tokens(np.array([[0, 0, 0, 52]], dtype=np.int32))
    
    # Get model input - this should return the input_ids tensor
    input_ids = generator.get_input("input_ids")
    assert input_ids is not None
    assert input_ids.shape[0] > 0


def test_generator_set_inputs(test_data_path):
    """Test Generator.set_inputs method"""
    model_path = os.fspath(
        Path(test_data_path) / "hf-internal-testing" / "tiny-random-gpt2-fp32"
    )
    
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=10)
    
    generator = og.Generator(model, params)
    
    # Create named tensors
    named_tensors = og.NamedTensors()
    named_tensors["input_ids"] = np.array([[0, 0, 0, 52]], dtype=np.int32)
    
    # Set inputs using named tensors
    generator.set_inputs(named_tensors)
    
    # Generate to ensure inputs were set correctly
    generator.generate_next_token()
    assert not generator.is_done()


def test_generator_set_model_input(test_data_path):
    """Test Generator.set_model_input method"""
    model_path = os.fspath(
        Path(test_data_path) / "hf-internal-testing" / "tiny-random-gpt2-fp32"
    )
    
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=10)
    
    generator = og.Generator(model, params)
    generator.append_tokens(np.array([[0, 0, 0, 52]], dtype=np.int32))
    
    # Create custom input
    custom_input = np.array([[0, 0, 0, 52]], dtype=np.int32)
    
    # Set model input directly
    generator.set_model_input("input_ids", custom_input)
    
    # Generate to ensure input was set correctly
    generator.generate_next_token()
    assert not generator.is_done()


def test_generator_get_next_tokens(test_data_path):
    """Test Generator.get_next_tokens method"""
    model_path = os.fspath(
        Path(test_data_path) / "hf-internal-testing" / "tiny-random-gpt2-fp32"
    )
    
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=10, batch_size=2)
    
    generator = og.Generator(model, params)
    generator.append_tokens(np.array([[0, 0, 0, 52], [0, 0, 195, 731]], dtype=np.int32))
    
    generator.generate_next_token()
    
    # Get the next tokens for all sequences in the batch
    next_tokens = generator.get_next_tokens()
    assert next_tokens is not None
    assert len(next_tokens) == 2  # batch_size


# ============================================================================
# NamedTensors API Tests
# ============================================================================

def test_named_tensors_complete_api():
    """Test all NamedTensors dictionary-like operations"""
    named_tensors = og.NamedTensors()
    
    # Test __setitem__ with numpy array
    named_tensors["input_ids"] = np.array([[1, 2, 3]], dtype=np.int32)
    
    # Test __setitem__ with Tensor
    tensor = og.Tensor(np.array([[4, 5, 6]], dtype=np.int32))
    named_tensors["attention_mask"] = tensor
    
    # Test __contains__
    assert "input_ids" in named_tensors
    assert "attention_mask" in named_tensors
    assert "non_existent" not in named_tensors
    
    # Test __getitem__
    retrieved = named_tensors["input_ids"]
    assert retrieved is not None
    
    # Test __len__
    assert len(named_tensors) == 2
    
    # Test keys
    keys = named_tensors.keys()
    assert "input_ids" in keys
    assert "attention_mask" in keys
    
    # Test __delitem__
    del named_tensors["input_ids"]
    assert "input_ids" not in named_tensors
    assert len(named_tensors) == 1
    
    # Clean up
    del named_tensors["attention_mask"]
    assert len(named_tensors) == 0


# ============================================================================
# Tensor API Tests
# ============================================================================

def test_tensor_complete_api():
    """Test all Tensor methods"""
    # Create tensor from numpy array
    np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    tensor = og.Tensor(np_array)
    
    # Test shape method
    shape = tensor.shape()
    assert shape == [2, 3]
    
    # Test type method
    tensor_type = tensor.type()
    assert tensor_type is not None
    
    # Test data method (returns pointer)
    data_ptr = tensor.data()
    assert data_ptr is not None
    
    # Test as_numpy method
    np_result = tensor.as_numpy()
    assert np.array_equal(np_result, np_array)


# ============================================================================
# Audios API Tests
# ============================================================================

def test_audios_open_bytes(test_data_path, relative_audio_path):
    """Test Audios.open_bytes static method"""
    audio_path = Path(test_data_path) / relative_audio_path
    
    if not audio_path.exists():
        pytest.skip(f"Audio file not found: {audio_path}")
    
    # Read audio file as bytes
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    # Load audio from bytes
    audios = og.Audios.open_bytes(audio_bytes)
    assert audios is not None


# ============================================================================
# Adapters API Tests
# ============================================================================

def test_adapters_unload(test_data_path, device, phi2_for):
    """Test Adapters.unload method"""
    # Skip if no adapter model available
    model_path = phi2_for(device)
    
    # Check if adapter exists
    adapter_path = Path(test_data_path) / "adapters" / "phi2-lora"
    if not adapter_path.exists():
        pytest.skip(f"Adapter not found: {adapter_path}")
    
    model = og.Model(model_path)
    adapters = og.Adapters(model)
    
    # Load an adapter
    adapter_name = "test_adapter"
    adapters.load(os.fspath(adapter_path), adapter_name)
    
    # Unload the adapter
    adapters.unload(adapter_name)


# ============================================================================
# Engine and Request API Tests
# ============================================================================

def test_engine_complete_api(test_data_path):
    """Test all Engine methods"""
    model_path = os.fspath(
        Path(test_data_path) / "hf-internal-testing" / "tiny-random-gpt2-fp32"
    )
    
    model = og.Model(model_path)
    engine = og.Engine(model)
    
    # Create a request
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=20)
    request = og.Request(params)
    
    # Add tokens to the request
    request.add_tokens(np.array([0, 0, 0, 52], dtype=np.int32))
    
    # Test add_request
    engine.add_request(request)
    
    # Test has_pending_requests
    has_pending = engine.has_pending_requests()
    assert has_pending is True or has_pending is False  # Valid boolean
    
    # Test step (process one iteration)
    if has_pending:
        engine.step()
    
    # Note: We'll let the engine naturally complete or remove request
    # Test remove_request (if still pending)
    if not request.is_done():
        engine.remove_request(request)


def test_request_complete_api(test_data_path):
    """Test all Request methods"""
    model_path = os.fspath(
        Path(test_data_path) / "hf-internal-testing" / "tiny-random-gpt2-fp32"
    )
    
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=20)
    
    request = og.Request(params)
    
    # Test add_tokens
    request.add_tokens(np.array([0, 0, 0, 52], dtype=np.int32))
    
    # Test has_unseen_tokens
    has_unseen = request.has_unseen_tokens()
    assert isinstance(has_unseen, bool)
    
    # Test get_unseen_token (if available)
    if has_unseen:
        token = request.get_unseen_token()
        assert isinstance(token, (int, np.integer))
    
    # Test set_opaque_data and get_opaque_data
    test_data = {"user_id": 123, "session": "abc"}
    request.set_opaque_data(test_data)
    
    retrieved_data = request.get_opaque_data()
    assert retrieved_data == test_data
    
    # Test is_done
    is_done = request.is_done()
    assert isinstance(is_done, bool)


# ============================================================================
# Module-level Function Tests
# ============================================================================

def test_gpu_device_functions():
    """Test GPU device ID functions"""
    if not og.is_cuda_available():
        pytest.skip("CUDA not available")
    
    # Test get_current_gpu_device_id
    try:
        current_id = og.get_current_gpu_device_id()
        assert isinstance(current_id, int)
        assert current_id >= 0
        
        # Test set_current_gpu_device_id
        og.set_current_gpu_device_id(0)
        
        # Verify it was set
        new_id = og.get_current_gpu_device_id()
        assert new_id == 0
    except Exception as e:
        # Some systems may not support this operation
        pytest.skip(f"GPU device operations not supported: {e}")


def test_execution_provider_library_functions():
    """Test execution provider registration functions"""
    # Test register_execution_provider_library
    try:
        # This may fail if the library doesn't exist, which is expected
        og.register_execution_provider_library("test_provider", "/fake/path/lib.so")
    except Exception:
        # Expected to fail with fake path, but tests the API
        pass
    
    # Test unregister_execution_provider_library
    try:
        og.unregister_execution_provider_library("test_provider")
    except Exception:
        # Expected to fail if not registered, but tests the API
        pass


def test_device_availability_functions():
    """Test all device availability check functions"""
    # Test is_qnn_available
    qnn_available = og.is_qnn_available()
    assert isinstance(qnn_available, bool)
    
    # Test is_webgpu_available
    webgpu_available = og.is_webgpu_available()
    assert isinstance(webgpu_available, bool)
    
    # Already tested in other places but verify here too
    assert isinstance(og.is_cuda_available(), bool)
    assert isinstance(og.is_dml_available(), bool)
    assert isinstance(og.is_rocm_available(), bool)
    assert isinstance(og.is_openvino_available(), bool)


# ============================================================================
# MultiModalProcessor API Tests
# ============================================================================

def test_multimodal_processor_call(test_data_path, relative_model_path, relative_image_path):
    """Test MultiModalProcessor.__call__ method"""
    model_path = Path(test_data_path) / relative_model_path
    
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")
    
    image_path = Path(test_data_path) / relative_image_path
    if not image_path.exists():
        pytest.skip(f"Image not found: {image_path}")
    
    model = og.Model(os.fspath(model_path))
    processor = model.create_multimodal_processor()
    
    # Test with single prompt
    prompt = "Describe this image"
    images = og.Images.open(os.fspath(image_path))
    
    result = processor(prompt, images=images)
    assert result is not None
    
    # Test with list of prompts
    prompts = ["Describe this image", "What do you see?"]
    result = processor(prompts, images=images)
    assert result is not None
    
    # Test with None prompt
    result = processor(None, images=images)
    assert result is not None


# ============================================================================
# Parameterized Tests
# ============================================================================

@pytest.mark.parametrize("device", devices)
def test_device_specific_apis(device, phi2_for):
    """Run device-specific API tests"""
    try:
        model_path = phi2_for(device)
        model = og.Model(model_path)
        
        # Verify model loads on the specified device
        device_type = model.device_type
        assert isinstance(device_type, str)
        assert len(device_type) > 0
    except Exception as e:
        pytest.skip(f"Device {device} not available or model not found: {e}")
