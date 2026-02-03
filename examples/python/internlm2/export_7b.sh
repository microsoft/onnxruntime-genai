#!/bin/bash
# Export InternLM2-7B model to ONNX format

# Set environment
export HF_HOME="./hf_cache"

# Model configuration
MODEL_ID="internlm/internlm2-7b"
OUTPUT_DIR="./internlm2-7b-onnx-cpu-int4-awq"

# Export with INT4 AWQ quantization (recommended for 7B)
python -m onnxruntime_genai.models.builder \
    --input ${MODEL_ID} \
    --output ${OUTPUT_DIR} \
    --precision int4 \
    --execution_provider cpu \
    --extra_options int4_accuracy_level=4

echo "Export complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo ""
echo "To run inference:"
echo "  python -c \"import onnxruntime_genai as og; model = og.Model('${OUTPUT_DIR}'); print('Model loaded successfully!')\""
