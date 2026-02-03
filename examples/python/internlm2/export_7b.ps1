# Export InternLM2-7B model to ONNX format (PowerShell version)

# Set environment
$env:HF_HOME = "./hf_cache"

# Model configuration
$MODEL_ID = "internlm/internlm2-7b"
$OUTPUT_DIR = "./internlm2-7b-onnx-cpu-int4-awq"

Write-Host "Exporting InternLM2-7B to ONNX with INT4 AWQ quantization..." -ForegroundColor Cyan

# Export with INT4 AWQ quantization (recommended for 7B)
python -m onnxruntime_genai.models.builder `
    --input $MODEL_ID `
    --output $OUTPUT_DIR `
    --precision int4 `
    --execution_provider cpu `
    --extra_options int4_accuracy_level=4

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nExport complete!" -ForegroundColor Green
    Write-Host "Model saved to: $OUTPUT_DIR" -ForegroundColor Green
    Write-Host ""
    Write-Host "To run inference:" -ForegroundColor Yellow
    Write-Host "  python -c `"import onnxruntime_genai as og; model = og.Model('$OUTPUT_DIR'); print('Model loaded successfully!')`"" -ForegroundColor Yellow
} else {
    Write-Host "`nExport failed!" -ForegroundColor Red
    exit 1
}
