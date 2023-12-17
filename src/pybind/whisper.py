import onnxruntime_genai as og
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperConfig
import numpy as np

model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name)
config = WhisperConfig.from_pretrained(model_name)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
input_features = processor([ds[0]["audio"]["array"]], return_tensors="pt").input_features

forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
forced_decoder_ids = [config.decoder_start_token_id] + list(map(lambda token: token[1], forced_decoder_ids))

batch_size, max_length, min_length, num_beams, num_return_sequences = 1, 26, 0, 5, 1
length_penalty, repetition_penalty = 1.0, 1.0

inputs = {
    "input_features": input_features.detach().cpu().numpy(),
    "max_length": np.array([max_length], dtype=np.int32),
    "min_length": np.array([min_length], dtype=np.int32),
    "num_beams": np.array([num_beams], dtype=np.int32),
    "num_return_sequences": np.array([num_return_sequences], dtype=np.int32),
    "length_penalty": np.array([length_penalty], dtype=np.float32),
    "repetition_penalty": np.array([repetition_penalty], dtype=np.float32),
    "decoder_input_ids": np.array([forced_decoder_ids], dtype=np.int32)
}

device_type = og.DeviceType.CPU
# device_type = og.DeviceType.CUDA

print("Loading model...")
model=og.Model("../../test_models/whisper-tiny", device_type)
print("Model loaded")

params=og.SearchParams(model)
params.num_beams = inputs["num_beams"]
params.max_length = inputs["max_length"]
params.length_penalty = inputs["length_penalty"]
params.whisper_input_features=inputs["input_features"]
params.whisper_decoder_input_ids=inputs["decoder_input_ids"]

search=params.CreateSearch()
state=model.CreateState(search.GetSequenceLengths(), params)

print("Processing")

while not search.IsDone():
    logits=state.Run(search.GetSequenceLength(), search.GetNextTokens(), search.GetNextIndices())
    # Note, if using beam search the logic below only affects the first beam entry
    # logits.GetArray()[50257]=-999 # Force ignore EOS token
    search.SetLogits(logits)

    # search.Apply_Repetition_Penalty(input.repetition_penalty)
    # time_stampe=state.Whisper().GetTimeStamps() ? Hypothetical
    # do_stuff_with_time_stamps()
    search.SelectTop()

ort_expected_transcription = (" Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.")

og_transcription = processor.decode(search.GetSequence(0).GetArray(), skip_special_tokens=True)
print(og_transcription)
print("Expected transcription:")
print(ort_expected_transcription)