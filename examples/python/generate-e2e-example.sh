# Description: Example of generate end-to-end usage, including model building and running.
pip install numpy
pip install transformers
pip install torch
pip install onnx
pip install onnxruntime-gpu
python3 -m onnxruntime_genai.models.builder -m microsoft/phi-2 -o genai_models/phi2-int4-cpu -p int4 -e cpu -c hf_cache
python3 model-generate.py -m genai_models/phi2-int4-cpu -pr "my favorite movie is" "write a function that always returns True" "I am very happy" -ep cpu -p 0.0 -k 1 -v 