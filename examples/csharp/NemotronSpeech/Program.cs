// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using CommonUtils;
using Microsoft.ML.OnnxRuntimeGenAI;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using System.Text.Json;

if (args.Length < 2) {
  Console.WriteLine("Usage: NemotronSpeech <model_path> <audio_file.wav> [execution_provider]");
  return;
}

string modelPath = args[0];
string audioFile = args[1];
string executionProvider = "follow_config";
bool enableVad = false;

for (int i = 2; i < args.Length; i++) {
  if (args[i] == "--enable_vad") {
    enableVad = true;
  } else {
    executionProvider = args[i];
  }
}

// Read sample_rate and chunk_samples from genai_config.json
var configJson = JsonDocument.Parse(File.ReadAllText(Path.Combine(modelPath, "genai_config.json")));
var modelConfig = configJson.RootElement.GetProperty("model");
int sampleRate = modelConfig.GetProperty("sample_rate").GetInt32();
int chunkSize = modelConfig.GetProperty("chunk_samples").GetInt32();

// Load audio, convert to mono, and resample to match the model's expected sample rate
float[] audio = LoadAudio(audioFile, sampleRate);
Console.WriteLine($"Audio: {audio.Length / (double)sampleRate:F1}s ({audio.Length} samples)");

using var config = Common.GetConfig(path: modelPath, ep: executionProvider, null, new GeneratorParamsArgs());
using var model = new Model(config);
using var processor = new StreamingProcessor(model);

// VAD is disabled by default. Enable via --enable_vad.
if (!enableVad) {
    processor.SetOption("use_vad", "false");
}
var useVad = processor.GetOption("use_vad");
Console.WriteLine("  Use VAD: " + useVad);
if (useVad == "true") {
    Console.WriteLine("  VAD threshold: " + processor.GetOption("vad_threshold"));
}

using var tokenizer = new Tokenizer(model);
using var tokenizerStream = tokenizer.CreateStream();
using var genParams = new GeneratorParams(model);
using var generator = new Generator(model, genParams);
Console.WriteLine(new string('-', 60));
string fullTranscript = "";
int chunksTotal = 0;
int chunksProcessed = 0;
int chunksSkipped = 0;

for (int i = 0; i < audio.Length; i += chunkSize) {
  int remaining = Math.Min(chunkSize, audio.Length - i);
  float[] chunk = new float[remaining];
  Array.Copy(audio, i, chunk, 0, remaining);

  using var inputs = processor.Process(chunk);
  chunksTotal++;
  if (inputs != null) {
    chunksProcessed++;
    generator.SetInputs(inputs);
    fullTranscript += DecodeTokens(generator, tokenizerStream);
  } else {
    chunksSkipped++;
  }
}

// Flush remaining buffered audio
using var flushInputs = processor.Flush();
if (flushInputs != null) {
  generator.SetInputs(flushInputs);
  fullTranscript += DecodeTokens(generator, tokenizerStream);
}

Console.WriteLine($"\n{new string('=', 60)}");
Console.WriteLine($"  {fullTranscript.Trim()}");
Console.WriteLine(new string('=', 60));
if (useVad == "true") {
  double pctSaved = chunksTotal > 0 ? (double)chunksSkipped / chunksTotal * 100.0 : 0.0;
  Console.WriteLine($"  VAD Metrics: {chunksTotal} total chunks, {chunksProcessed} processed, " +
                    $"{chunksSkipped} skipped ({pctSaved:F1}% compute saved)");
}

static string DecodeTokens(Generator generator, TokenizerStream tokenizerStream) {
  string text = "";
  while (!generator.IsDone()) {
    generator.GenerateNextToken();
    var tokens = generator.GetNextTokens();
    if (tokens.Length > 0) {
      string tokenText = tokenizerStream.Decode(tokens[0]);
      if (!string.IsNullOrEmpty(tokenText)) {
        Console.Write(tokenText);
        text += tokenText;
      }
    }
  }
  return text;
}

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
  // Allocate memory to read, any num works.
  float[] buffer = new float[4096];
  int read;
  while ((read = source.Read(buffer, 0, buffer.Length)) > 0) {
    for (int i = 0; i < read; i++)
      samples.Add(buffer[i]);
  }
  return samples.ToArray();
}
