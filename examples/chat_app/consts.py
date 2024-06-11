import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)

default_prompt = "<|user|>\n<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n"
