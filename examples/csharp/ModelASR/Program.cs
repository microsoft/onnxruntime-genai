// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Streaming ASR example using the onnxruntime-genai StreamingProcessor API.
//
// Drives any streaming encoder/decoder ASR model exposed through onnxruntime-genai:
//   * NVIDIA Nemotron streaming RNN-T (`nemotron_speech`, multilingual)
//   * Moonshine streaming encoder-decoder (`streaming_enc_dec_asr`, English only)
//
// The runtime pipeline is identical for both: the StreamingProcessor factory
// dispatches per `model.type`, and the Generator routes both through the
// TransducerState path under the hood.
//
// The `--use_vad` flag is Nemotron-specific; Moonshine has no VAD runtime knob.

using CommonUtils;
using Microsoft.ML.OnnxRuntimeGenAI;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using System.Text;
using System.Text.Json;

if (args.Length < 2) {
  Console.WriteLine("Usage: ModelASR <model_path> <audio_file.wav> [execution_provider]");
  return;
}

string modelPath = args[0];
string audioFile = args[1];
string executionProvider = "follow_config";
string useVadOverride = "";

for (int i = 2; i < args.Length; i++) {
  if (args[i] == "--use_vad" && i + 1 < args.Length) {
    useVadOverride = args[++i];
  } else {
    executionProvider = args[i];
  }
}

// Read sample_rate and chunk_samples from genai_config.json
using var configJson = JsonDocument.Parse(File.ReadAllText(Path.Combine(modelPath, "genai_config.json")));
var modelConfig = configJson.RootElement.GetProperty("model");
int sampleRate = modelConfig.GetProperty("sample_rate").GetInt32();
int chunkSize = modelConfig.GetProperty("chunk_samples").GetInt32();
string timestampLevel = modelConfig.TryGetProperty("timestamp_level", out var timestampLevelElement)
  ? timestampLevelElement.GetString() ?? "off"
  : "off";

// Load audio, convert to mono, and resample to match the model's expected sample rate
float[] audio = LoadAudio(audioFile, sampleRate);
Console.WriteLine($"Audio: {audio.Length / (double)sampleRate:F1}s ({audio.Length} samples)");

using var config = Common.GetConfig(path: modelPath, ep: executionProvider, null, new GeneratorParamsArgs());
using var model = new Model(config);
using var processor = new StreamingProcessor(model);

// VAD is off by default. Use --use_vad true to enable (requires "vad" section in genai_config.json).
processor.SetOption("use_vad", "false");
if (useVadOverride == "true") {
    try {
        processor.SetOption("use_vad", "true");
    } catch (Exception e) {
        Console.WriteLine($"  VAD: disabled (no VAD config in genai_config.json: {e.Message})");
    }
}
var useVad = processor.GetOption("use_vad");
Console.WriteLine("  Use VAD: " + useVad);
if (useVad == "true") {
    Console.WriteLine("  VAD threshold: " + processor.GetOption("vad_threshold"));
}

using var genParams = new GeneratorParams(model);
using var generator = new Generator(model, genParams);
using var postprocessor = new TokenPostProcessor(generator);
Console.WriteLine(new string('-', 60));
string fullTranscript = "";
int chunksTotal = 0;
int chunksProcessed = 0;

for (int i = 0; i < audio.Length; i += chunkSize) {
  int remaining = Math.Min(chunkSize, audio.Length - i);
  float[] chunk = new float[remaining];
  Array.Copy(audio, i, chunk, 0, remaining);

  using var inputs = processor.Process(chunk);
  chunksTotal++;
  if (inputs != null) {
    chunksProcessed++;
    generator.SetInputs(inputs);
    fullTranscript += DecodeTokens(generator, postprocessor, timestampLevel);
  }
}

// Flush remaining buffered audio
while (true) {
  using var flushInputs = processor.Flush();
  if (flushInputs == null) break;
  chunksProcessed++;
  generator.SetInputs(flushInputs);
  fullTranscript += DecodeTokens(generator, postprocessor, timestampLevel);
}

postprocessor.FinalizeProcessing();
foreach (var result in postprocessor.GetNextTokenResults()) {
  Console.Write(FormatTimestampRecords(result, timestampLevel != "word"));
}
int chunksSkipped = chunksTotal - chunksProcessed;

Console.WriteLine($"\n{new string('=', 60)}");
Console.WriteLine($"  {fullTranscript.Trim()}");
Console.WriteLine(new string('=', 60));
if (useVad == "true") {
  double pctSaved = chunksTotal > 0 ? (double)chunksSkipped / chunksTotal * 100.0 : 0.0;
  Console.WriteLine($"  VAD Metrics: {chunksTotal} total chunks, {chunksProcessed} processed, " +
                    $"{chunksSkipped} skipped ({pctSaved:F1}% compute saved)");
}

static string DecodeTokens(Generator generator, TokenPostProcessor postprocessor, string timestampLevel) {
  string text = "";
  while (!generator.IsDone()) {
    generator.GenerateNextToken();
    postprocessor.ProcessNextTokens();
    foreach (var result in postprocessor.GetNextTokenResults()) {
      text += result.GetText();
      if (result.GetTimestamps() == null) {
        string tokenText = result.GetText();
        Console.Write(tokenText);
      } else {
        string timestampedText = FormatTimestampRecords(result, timestampLevel != "word");
        Console.Write(timestampedText);
      }
    }
  }
  return text;
}

static string FormatTimestampRecords(TokenPostProcessorResult result, bool useSegments) {
  var timestamps = result.GetTimestamps();
  if (timestamps == null) return "";
  var records = useSegments ? timestamps.GetSegments() : timestamps.GetWords();
  return string.Concat(records.Select(record => {
    string text = useSegments ? record.GetText() : record.GetText().Trim();
    string separator = useSegments && (text.Length == 0 || !char.IsWhiteSpace(text[0])) ? " " : "";
    string suffix = useSegments ? "" : " ";
    string prefix = record.HasTimestamp() ? $"[{record.GetStartTime():F2} - {record.GetStopTime():F2}]" : "";
    if (record.GetSpeakerLabel() is string speaker) prefix += $"[{speaker}]";
    return $"{prefix}{separator}{text}{suffix}";
  }));
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
