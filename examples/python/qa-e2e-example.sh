# Description: Example of chatbot end-to-end usage, including model building and running.
python3 -m onnxruntime_genai.models.builder -m microsoft/phi-2 -o genai_models/phi2-int4-cpu -p int4 -e cpu -c hf_cache
python3 model-qa.py -m genai_models/phi2-int4-cpu -e cpu -p 0.0 -k 1
