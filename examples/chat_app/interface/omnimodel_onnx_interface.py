import gc
import onnxruntime_genai as og
from consts import default_prompt, logging
from app_modules.utils import convert_to_markdown, shared_state
import os

logging.getLogger("interface")


class Omni_ONNXModel():

    """A wrapper for ONNXRuntime GenAI to run ONNX Omni model"""

    def __init__(self, model_path, execution_provider):
        self.og = og

        logging.info("Loading model...")
        self.config = og.Config(model_path)
        self.config.clear_providers()
        if execution_provider != "cpu":
            print(f"Setting model to {execution_provider}...")
            self.config.append_provider(execution_provider)
        self.model = og.Model(self.config)
        logging.info("Loaded model ...")

        # define image and audio instances for omni model
        self.images = None
        self.audios = None

        self.processor = self.model.create_multimodal_processor()
        self.tokenizer = self.processor.create_stream()

        self.enable_history_max = 2
        self.history_template = "[INST] {input} [/INST]{response}</s>"


    def generate_prompt_with_history(self, images_paths, history, text=default_prompt, audios_paths=None, max_length=3072):
        
        prompt = "<|user|>\n"

        for dialog in history[-self.enable_history_max:]:
            print(dialog, '......#####')
            prompt += f'{self.history_template.format(input=dialog[0], response=dialog[1])}'

        image_tags = ""
        # process the image files
        if not images_paths:
            logging.info("No image provided ... ")
        else:
            for i, image_path in enumerate(images_paths):
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                image_tags += f"<|image_{i+1}|>\n"
            
                prompt += image_tags
            self.images = og.Images.open(*images_paths)
            logging.info("Image file generated ....")

            
        # process the audio files
        audio_tags = ""
        if not audios_paths:
            logging.info("No image provided ... ")
        else:
            for i, audio_path in enumerate(audios_paths):
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                audio_tags += f"<|audio_{i+1}|>\n"
                prompt += audio_tags
            self.audios = og.Audios.open(*audios_paths)
            logging.info("Audio file generated ....")

        # process the text input
        prompt += f"{text}<|end|>\n<|assistant|>\n"

        logging.info("Preprocessing images and prompt ...")
        
        logging.info("Processing inputs ...")
        # TODO: rui-ren Test prompt
        logging.info(prompt)
        inputs = self.processor(prompt, images=self.images, audios=self.audios)
        
        if len(inputs) > max_length:
            logging.info("Inputs is larger than the max_length, trim history...")
            history.clear()
            prompt = f'{self.chat_template.format(text=text, image_tags=image_tags, audio_tags=audio_tags)}'
            inputs = self.processor(prompt, images=self.images, audios=self.audios)

        logging.info("The inputs {}".format(inputs))

        return inputs


    def search(self, inputs, max_length: int = 7680, token_printing_step: int = 1):
        output = ""
        params = og.GeneratorParams(self.model)
        params.set_inputs(inputs)

        search_options = {"max_length": max_length}
        params.set_search_options(**search_options)
        generator = og.Generator(self.model, params)

        # TODO: rui-ren Test prompt
        logging.info("Generate the ouput")
        while not generator.is_done():
            generator.generate_next_token()
            next_token = generator.get_next_tokens()[0]
            output += self.tokenizer.decode(next_token)

        logging.info("Output - {}".format(output))
        return output


    def predict(self, text, chatbot, history, max_length_tokens, max_context_length_tokens, token_printing_step, *args):

        if text == "":
            yield chatbot, history, "Empty context"
            return

        inputs = self.generate_prompt_with_history(
            text=text,
            history=history,
            images_paths=args[0],
            audios_paths=args[1],
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
            args[0],
        )

