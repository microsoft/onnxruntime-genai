// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntimeGenAI;

if (args.Length < 2) {
  Console.WriteLine("Usage: StreamingASR <model_path> <audio_file.wav>");
  return;
}

string modelPath = args[0];
string audioFile = args[1];

// Load raw PCM audio
float[] audio = LoadWavAudio(audioFile);
Console.WriteLine($"Audio: {audio.Length / 16000.0:F1}s ({audio.Length} samples)");

using var model = new Model(modelPath);
using var processor = new StreamingProcessor(model);
using var tokenizer = new Tokenizer(model);
using var tokenizerStream = tokenizer.CreateStream();
using var genParams = new GeneratorParams(model);
using var generator = new Generator(model, genParams);

int chunkSize = 8960;  // 560ms chunks
Console.WriteLine(new string('-', 60));
string fullTranscript = "";

void DecodeChunk() {
  while (!generator.IsDone()) {
    generator.GenerateNextToken();
    var tokens = generator.GetNextTokens();
    if (tokens.Length > 0) {
      string text = tokenizerStream.Decode(tokens[0]);
      if (!string.IsNullOrEmpty(text)) {
        Console.Write(text);
        fullTranscript += text;
      }
    }
  }
}

for (int i = 0; i < audio.Length; i += chunkSize) {
  int remaining = Math.Min(chunkSize, audio.Length - i);
  float[] chunk = new float[remaining];
  Array.Copy(audio, i, chunk, 0, remaining);

  using var inputs = processor.Process(chunk);
  if (inputs != null) {
    generator.SetInputs(inputs);
    DecodeChunk();
  }
}

// Flush remaining buffered audio
using var flushInputs = processor.Flush();
if (flushInputs != null) {
  generator.SetInputs(flushInputs);
  DecodeChunk();
}

// Feed silence chunks for right context
for (int i = 0; i < 4; i++) {
  float[] silence = new float[chunkSize];
  using var silenceInputs = processor.Process(silence);
  if (silenceInputs != null) {
    generator.SetInputs(silenceInputs);
    DecodeChunk();
  }
}

Console.WriteLine($"\n{new string('=', 60)}");
Console.WriteLine($"  {fullTranscript.Trim()}");
Console.WriteLine(new string('=', 60));

static float[] LoadWavAudio(string path) {
  using var reader = new BinaryReader(File.OpenRead(path));

  // Skip RIFF header (44 bytes for standard WAV)
  reader.ReadBytes(44);

  var samples = new List<float>();
  while (reader.BaseStream.Position < reader.BaseStream.Length) {
    try {
      short sample = reader.ReadInt16();
      samples.Add(sample / 32768.0f);
    } catch (EndOfStreamException) {
      break;
    }
  }
  return samples.ToArray();
}
