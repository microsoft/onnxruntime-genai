// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntimeGenAI;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using System.Text.Json;

if (args.Length < 2) {
  Console.WriteLine("Usage: NemotronSpeech <model_path> <audio_file.wav>");
  return;
}

string modelPath = args[0];
string audioFile = args[1];

// Read sample_rate and chunk_samples from genai_config.json
var configJson = JsonDocument.Parse(File.ReadAllText(Path.Combine(modelPath, "genai_config.json")));
var modelConfig = configJson.RootElement.GetProperty("model");
int sampleRate = modelConfig.GetProperty("sample_rate").GetInt32();
int chunkSize = modelConfig.GetProperty("chunk_samples").GetInt32();

// Load audio, convert to mono, and resample to match the model's expected sample rate
float[] audio = LoadAudio(audioFile, sampleRate);
Console.WriteLine($"Audio: {audio.Length / (double)sampleRate:F1}s ({audio.Length} samples)");

using var model = new Model(modelPath);
using var processor = new StreamingProcessor(model);
using var tokenizer = new Tokenizer(model);
using var tokenizerStream = tokenizer.CreateStream();
using var genParams = new GeneratorParams(model);
using var generator = new Generator(model, genParams);
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

Console.WriteLine($"\n{new string('=', 60)}");
Console.WriteLine($"  {fullTranscript.Trim()}");
Console.WriteLine(new string('=', 60));

static float[] LoadAudio(string path, int targetSampleRate) {
  using var reader = new AudioFileReader(path);

  // Convert to mono if needed
  ISampleProvider source = reader;
  if (reader.WaveFormat.Channels > 1) {
    source = new StereoToMonoSampleProvider(source);
  }

  // Resample if needed
  if (reader.WaveFormat.SampleRate != targetSampleRate) {
    source = new WdlResamplingSampleProvider(source, targetSampleRate);
  }

  var samples = new List<float>();
  float[] buffer = new float[4096];
  int read;
  while ((read = source.Read(buffer, 0, buffer.Length)) > 0) {
    for (int i = 0; i < read; i++)
      samples.Add(buffer[i]);
  }
  return samples.ToArray();
}
