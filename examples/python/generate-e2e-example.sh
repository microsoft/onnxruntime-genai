# Description: Example of generate end-to-end usage, including model building and running
pip install numpy transformers torch onnx onnxruntime
python3 -m onnxruntime_genai.models.builder -m microsoft/phi-2 -o genai_models/phi2-int4-cpu -p int4 -e cpu -c hf_cache
python3 model-generate.py -m genai_models/phi2-int4-cpu -e cpu -pr "my favorite movie is" "write a function that always returns True" "I am very happy" -p 0.0 -k 1 -v 
