import gc
import os
import sys
import onnxruntime_genai as og
from app_modules.utils import convert_to_markdown, is_stop_word_or_prefix, shared_state

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, "..", "..", ".."))

class ONNXModel():
    """A wrapper for OnnxRuntime-GenAI to run ONNX LLM model."""

    def __init__(self, model_path, execution_provider):
        self.og = og

        logging.info("Loading model...")
        self.config = og.Config(model_path)
        self.config.clear_providers()
        if execution_provider != "cpu":
            self.config.append_provider(execution_provider)
        self.model = og.Model(self.config)
        logging.info("Loaded model...")

        self.tokenizer = og.Tokenizer(self.model)
        self.tokenizer_stream = self.tokenizer.create_stream()
        self.model_path = model_path

        if "phi" in self.model_path:
            self.template_header = ""
            self.enable_history_max = 10 if "mini" in self.model_path else 2
            self.history_template = "<|user|>{input}<|end|><|assistant|>{response}<|end|>"
            self.chat_template = "<|user|>{input}<|end|><|assistant|>"
        elif "Llama-3" in self.model_path:
            self.enable_history_max = 2
            self.template_header =  """<|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant.<|eot_id|>"""
            self.history_template = """<|start_header_id|>user<|end_header_id|>
{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{response}<|eot_id|>"""
            
            self.chat_template = """<|start_header_id|>user<|end_header_id|>
{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            
            #self.chat_template = llama3_template
        else:
            self.enable_history_max = 2
            self.template_header = "<s>"
            self.history_template = "[INST] {input} [/INST]{response}</s>"
            self.chat_template = "[INST] {input} [/INST]"

    def generate_prompt_with_history(self, text, history, max_length=2048):
        prompt = ""

        for dialog in history[-self.enable_history_max:]:
            prompt += f'{self.history_template.format(input=dialog[0], response=dialog[1])}'

        prompt = self.template_header + prompt

        prompt += f'{self.chat_template.format(input=text)}'

        input_ids = self.tokenizer.encode(prompt)

        if len(input_ids) <= max_length:
            return input_ids
        else:
            history.clear()
            if "Llama-3" in self.model_path:
                prompt = self.template_header
            prompt += f'{self.chat_template.format(input=text)}'
            return self.tokenizer.encode(prompt)

    def search(
        self,
        input_ids,
        max_length: int,
        token_printing_step: int = 4,
    ):
        output_tokens = []

        params = og.GeneratorParams(self.model)
        search_options = {"max_length" : max_length}
        params.set_search_options(**search_options)

        generator = og.Generator(self.model, params)
        generator.append_tokens(input_ids)

        idx = 0
        while not generator.is_done():
            idx += 1
            generator.generate_next_token()
            next_token = generator.get_next_tokens()[0]
            output_tokens.append(next_token)

            if idx % token_printing_step == 0:
                yield self.tokenizer.decode(output_tokens)

    def predict(
        self,
        text,
        chatbot,
        history,
        max_length_tokens,
        max_context_length_tokens,
        token_printing_step,
        *args
    ):
        if text == "":
            yield chatbot, history, "Empty context."
            return

        inputs = self.generate_prompt_with_history(
            text, history, max_length=max_context_length_tokens
        )

        if inputs is None:
            yield chatbot, history, "Input too long."
            return

        input_ids = inputs[-max_context_length_tokens:]

        human_tokens = [
            "[|Human|]",
            "Human:",
            "### HUMAN:",
            "### User:",
            "USER:",
            "<|im_start|>user",
            "<|user|>",
            "### Instruction:",
            "GPT4 Correct User:",
        ]

        ai_tokens = [
            "[|AI|]",
            "AI:",
            "### RESPONSE:",
            "### Response:",
            "ASSISTANT:",
            "<|im_start|>assistant",
            "<|assistant|>",
            "GPT4 Correct Assistant:",
            "### Assistant:",
        ]

        for x in self.search(
            input_ids,
            max_length=max_length_tokens,
            token_printing_step=token_printing_step,
        ):
            sentence = x

            if is_stop_word_or_prefix(sentence, ["[|Human|]", "[|AI|]", "Human:", "AIL"]) is False:
                for human_token in human_tokens:
                    if human_token in sentence:
                        sentence = sentence[: sentence.index(human_token)].strip()
                        break

                for ai_token in ai_tokens:
                    if ai_token in sentence:
                        sentence = sentence[: sentence.index(ai_token)].strip()
                        break
                sentence = sentence.strip()
                a, b = [[y[0], convert_to_markdown(y[1])] for y in history] + [[text, convert_to_markdown(sentence)]], [
                    *history,
                    [text, sentence],
                ]
                yield a, b, "Generating..."

            if shared_state.interrupted:
                shared_state.recover()
                try:
                    yield a, b, "Stop: Success"
                    return
                except Exception as e:
                    print(type(e).__name__, e)

        del input_ids
        gc.collect()

        try:
            yield a, b, "Generate: Success"
        except Exception as e:
            print(type(e).__name__, e)

        return
    
    def shutdown(self):
        pass
    
    def retry(self, chatbot, history, max_length_tokens, max_context_length_tokens, token_printing_step):
        if len(history) == 0:
            yield chatbot, history, "Empty context"
            return
        chatbot.pop()
        inputs = history.pop()[0]
        yield from self.predict(
            inputs,
            chatbot,
            history,
            max_length_tokens,
            max_context_length_tokens,
            token_printing_step,
        )