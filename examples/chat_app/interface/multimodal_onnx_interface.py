import gc
import onnxruntime_genai as og
from consts import default_prompt, logging
from app_modules.utils import convert_to_markdown, shared_state

logging.getLogger("interface")


class MultiModal_ONNXModel():

    """A wrapper for ONNXRuntime GenAI to run ONNX Multimodal model"""

    def __init__(self, model_path, execution_provider):
        self.og = og

        logging.info("Loading model...")
        self.config = og.Config(model_path)
        self.config.clear_providers()
        if execution_provider != "cpu":
            self.config.append_provider(execution_provider)
        self.model = og.Model(self.config)
        logging.info("Loaded model ...")

        self.processor = self.model.create_multimodal_processor()
        self.tokenizer = self.processor.create_stream()

        self.enable_history_max = 2
        self.template_header = "<s>"
        self.history_template = "[INST] {input} [/INST]{response}</s>"
        self.chat_template = "<|user|>\n{tags}\n{input}<|end|>\n<|assistant|>\n"

    def generate_prompt_with_history(self, images, history, text=default_prompt, max_length=3072):
        prompt = ""

        for dialog in history[-self.enable_history_max:]:
            prompt += f'{self.history_template.format(input=dialog[0], response=dialog[1])}'

        prompt = self.template_header + prompt

        image_tags = ""
        for i in range(len(images)):
            image_tags += f"<|image_{i+1}|>\n"

        prompt += f'{self.chat_template.format(input=text, tags=image_tags)}'
        if len(prompt) > max_length:
            history.clear()
            prompt = f'{self.chat_template.format(input=text, tags=image_tags)}'

        self.images = og.Images.open(*images)

        logging.info("Preprocessing images and prompt ...")
        inputs = self.processor(prompt, images=self.images)
        return inputs


    def search(self, inputs, max_length: int = 3072, token_printing_step: int = 1):
        output = ""
        params = og.GeneratorParams(self.model)
        params.set_inputs(inputs)

        search_options = {"max_length": max_length}
        params.set_search_options(**search_options)
        generator = og.Generator(self.model, params)

        idx = 0
        while not generator.is_done():
            idx += 1
            generator.generate_next_token()
            next_token = generator.get_next_tokens()[0]
            output += self.tokenizer.decode(next_token)

        return output

    def predict(self, text, chatbot, history, max_length_tokens, max_context_length_tokens, token_printing_step, *args):

        if text == "":
            yield chatbot, history, "Empty context"
            return

        inputs = self.generate_prompt_with_history(
            text=text,
            history=history,
            images=args[0],
            max_length=max_context_length_tokens
        )

        sentence = self.search(
            inputs,
            max_length=max_length_tokens,
            token_printing_step=token_printing_step,
        )

        sentence = sentence.strip()
        a, b = [[y[0], convert_to_markdown(y[1])] for y in history] + [[text, convert_to_markdown(sentence)]], [
            *history,
            [ text, sentence],
        ]
        yield a, b, "Generating ... "


        if shared_state.interrupted:
            shared_state.recover()
            try:
                yield a, b, "Stop: Success"
                return
            except Exception as e:
                print(type(e).__name__, e)

        del inputs
        gc.collect()

        try:
            yield a, b, "Generate: Success"

        except Exception as e:
            print(type(e).__name__, e)

        return

    def shutdown(self):
        pass

    def retry(self, chatbot, history, max_length_tokens, max_context_length_tokens, token_printing_step, *args):
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
            args[0]
        )

