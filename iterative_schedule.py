from collections import deque
from pathlib import Path
import time
from typing import List

import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoConfig

model_path = Path("/devdisk0/yingxiong/projects/onnxruntime-genai/models/llama7b_fp16/")


class Engine:
    def __init__(self):
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        # sess_options.log_verbosity_level = 0
        # sess_options.log_severity_level = 0
        ep = (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
            },
        )
        self.model = ort.InferenceSession(
            model_path / "model.onnx", sess_options=sess_options, providers=[ep]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_batch_size = 16
        self.model_config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.device = torch.device(0)
        self.logits_name = "logits"
        self.input_ids_name = "input_ids"
        self.attention_mask_name = "attention_mask"
        self.pt_to_np = {
            "torch.int32": np.int32,
            "torch.float32": np.float32,
            "torch.float16": np.float16,
        }
        self._set_up_kv_cache()

    def _set_up_kv_cache(self):
        max_sequence_length = self.model_config.max_position_embeddings // 2
        num_heads, head_size = (
            self.model_config.num_attention_heads,
            self.model_config.hidden_size // self.model_config.num_attention_heads,
        )
        self.kv_cache = [
            [
                [
                    ort.OrtValue.ortvalue_from_shape_and_type(
                        [
                            self.max_batch_size,
                            num_heads,
                            max_sequence_length,
                            head_size,
                        ],
                        np.float16,
                        "cuda",
                        self.device.index,
                    ),
                    ort.OrtValue.ortvalue_from_shape_and_type(
                        [
                            self.max_batch_size,
                            num_heads,
                            max_sequence_length,
                            head_size,
                        ],
                        np.float16,
                        "cuda",
                        self.device.index,
                    ),
                ]
                for _ in range(self.model_config.num_hidden_layers)
            ]
            for _ in range(2)
        ]

    def _get_next_token(self, lhs_logits, temperature, top_p):
        next_token = None
        if temperature > 0:
            probs = torch.softmax(lhs_logits / temperature, dim=-1)
            next_token = self._sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(lhs_logits, dim=-1, keepdim=True)
        return next_token

    def generate(self, prompts, max_length, temperature=0, top_p=0.7):
        model = self.model
        tokenizer = self.tokenizer
        max_sequence_length = self.model_config.max_position_embeddings // 2
        num_heads, head_size = (
            self.model_config.num_attention_heads,
            self.model_config.hidden_size // self.model_config.num_attention_heads,
        )
        #    prompts, max_length = ['''```python
        # def print_prime(n):
        # """
        # Print all primes between 1 and n
        # """'''], 128

        # print(f"prompts:{prompts}")
        generated = None

        with torch.no_grad():
            # Get model and its initial inputs/outputs
            tokenizer.pad_token = "[PAD]"
            encodings_dict = tokenizer(prompts, padding=True)

            input_ids = encodings_dict["input_ids"]
            attention_mask = encodings_dict["attention_mask"]
            batch_size, sequence_length = np.shape(input_ids)

            num_valid_tokens = np.sum(attention_mask, axis=-1)

            past_seq_len = 0
            present_seq_len = sequence_length
            if sequence_length + max_length > max_sequence_length:
                return [""] * batch_size
            for len_idx in range(max_length):
                # Bind inputs/outputs to IO binding
                io_binding = self.model.io_binding()

                # input
                input_ids = np.asarray(input_ids, dtype=np.int64)
                attention_mask = np.asarray(attention_mask, dtype=np.int64)
                input_ids_ort_value = ort.OrtValue.ortvalue_from_numpy(
                    np.array(input_ids), self.device.type, self.device.index
                )
                attention_mask_ort_value = ort.OrtValue.ortvalue_from_numpy(
                    np.array(attention_mask), self.device.type, self.device.index
                )
                device_type = input_ids_ort_value.device_name()
                io_binding.bind_input(
                    name=self.input_ids_name,
                    device_type=device_type,
                    device_id=self.device.index,
                    element_type=input_ids.dtype,
                    shape=input_ids_ort_value.shape(),
                    buffer_ptr=input_ids_ort_value.data_ptr(),
                )
                io_binding.bind_input(
                    name=self.attention_mask_name,
                    device_type=device_type,
                    device_id=self.device.index,
                    element_type=attention_mask.dtype,
                    shape=attention_mask_ort_value.shape(),
                    buffer_ptr=attention_mask_ort_value.data_ptr(),
                )
                for i in range(self.model_config.num_hidden_layers):
                    io_binding.bind_input(
                        name=f"past_key_values.{i}.key",
                        device_type=device_type,
                        device_id=self.device.index,
                        element_type=np.float16,
                        shape=[
                            batch_size,
                            self.model_config.num_hidden_layers,
                            past_seq_len,
                            head_size,
                        ],
                        buffer_ptr=self.kv_cache[len_idx % 2][i][0].data_ptr(),
                    )
                    io_binding.bind_input(
                        name=f"past_key_values.{i}.value",
                        device_type=device_type,
                        device_id=self.device.index,
                        element_type=np.float16,
                        shape=[
                            batch_size,
                            self.model_config.num_hidden_layers,
                            past_seq_len,
                            head_size,
                        ],
                        buffer_ptr=self.kv_cache[len_idx % 2][i][1].data_ptr(),
                    )

                # output
                logits = ort.OrtValue.ortvalue_from_shape_and_type(
                    [
                        batch_size,
                        sequence_length if len_idx == 0 else 1,
                        self.model_config.vocab_size,
                    ],
                    np.float16,
                    self.device.type,
                    self.device.index,
                )
                io_binding.bind_output(
                    name=self.logits_name,
                    device_type=device_type,
                    device_id=self.device.index,
                    element_type=np.float16,
                    shape=logits.shape(),
                    buffer_ptr=logits.data_ptr(),
                )
                for i in range(self.model_config.num_hidden_layers):
                    io_binding.bind_output(
                        name=f"present.{i}.key",
                        device_type=device_type,
                        device_id=self.device.index,
                        element_type=np.float16,
                        shape=[
                            batch_size,
                            self.model_config.num_hidden_layers,
                            present_seq_len,
                            head_size,
                        ],
                        buffer_ptr=self.kv_cache[(len_idx + 1) % 2][i][0].data_ptr(),
                    )
                    io_binding.bind_output(
                        name=f"present.{i}.value",
                        device_type=device_type,
                        device_id=self.device.index,
                        element_type=np.float16,
                        shape=[
                            batch_size,
                            self.model_config.num_hidden_layers,
                            present_seq_len,
                            head_size,
                        ],
                        buffer_ptr=self.kv_cache[(len_idx + 1) % 2][i][1].data_ptr(),
                    )

                io_binding.synchronize_inputs()
                model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

                # Sample with argmax (greedy search)
                next_token_logits = torch.from_numpy(
                    io_binding.get_outputs()[0].numpy()
                ).to(self.device)
                # print(next_token_logits)

                if generated is None:
                    next_tokens = []
                    for i in range(batch_size):
                        next_tokens.append(
                            self._get_next_token(
                                next_token_logits[i, num_valid_tokens[i] - 1, :],
                                temperature,
                                top_p,
                            ).item()
                        )
                    next_tokens = torch.tensor(next_tokens).unsqueeze(1).to(self.device)
                    # next_tokens = torch.argmax(next_token_logits[:, -1, :], dim=-1).unsqueeze(1)
                    generated = next_tokens
                else:
                    next_tokens = self._get_next_token(
                        next_token_logits[:, -1, :], temperature, top_p
                    )
                    generated = torch.cat([generated, next_tokens], dim=-1)

                past_seq_len = present_seq_len
                present_seq_len = present_seq_len + 1

                input_ids = next_tokens.cpu().numpy()
                attention_mask = np.concatenate(
                    (attention_mask, np.ones((batch_size, 1), dtype=np.int32)), 1
                )

        # gc.collect()
        # torch.cuda.empty_cache()
        results = [
            tokenizer.decode([a for a in tokens if a != 0]) for tokens in generated
        ]
        # print(f"results:{results}")
        return results

    def schedule(self):
        if not self.unscheduled_requests:
            return None
        scheduled = []
        i = 0
        while i < len(self.unscheduled_requests) and i < self.max_batch_size:
            scheduled.append(self.unscheduled_requests[i])
            i += 1

        for _ in range(i):
            self.unscheduled_requests.popleft()
        return scheduled

    def generate_schedule(self, prompts, max_length, temperature=0, top_p=0.7):
        results = []
        self.unscheduled_requests = deque(prompts)
        while self.unscheduled_requests:
            scheduled = self.schedule()
            if scheduled:
                results.extend(self.generate(scheduled, max_length, temperature, top_p))
        return results

    def generate_naive(self, prompts, max_length, temperature=0, top_p=0.7):
        results = []
        for prompt in prompts:
            results.extend(self.generate([prompt], max_length, temperature, top_p))
        return results


def main():
    engine = Engine()
    test_prompt = ["Hello World. " * 33] * 100

    # warmup
    for _ in range(10):
        result = engine.generate(test_prompt, max_length=15)
    print(result)

    TOKENS = 115 * 100

    # naive for loop
    start = time.time()
    result = engine.generate_naive(test_prompt, max_length=15)
    print(f"Naive: {time.time()-start}")
    print(f"Naive token throughput: {TOKENS/(time.time()-start):.2f} tokens/s")

    # schedule
    start = time.time()
    result = engine.generate_schedule(test_prompt, max_length=15)
    print(f"Schedule: {time.time()-start}")
    print(f"Schedule token throughput: {TOKENS/(time.time()-start):.2f} tokens/s")


if __name__ == "__main__":
    main()
