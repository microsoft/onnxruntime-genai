// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using CommonUtils;
using Microsoft.ML.OnnxRuntimeGenAI;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using System.Text;
using System.Text.Json;

if (args.Length < 2) {
  Console.WriteLine("Usage: NemotronSpeech <model_path> <audio_file.wav> [execution_provider]");
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
var configJson = JsonDocument.Parse(File.ReadAllText(Path.Combine(modelPath, "genai_config.json")));
var modelConfig = configJson.RootElement.GetProperty("model");
int sampleRate = modelConfig.GetProperty("sample_rate").GetInt32();
int chunkSize = modelConfig.GetProperty("chunk_samples").GetInt32();
string timestampLevel = modelConfig.TryGetProperty("timestamp_level", out var timestampLevelElement)
  ? timestampLevelElement.GetString() ?? "off"
  : "off";
bool timestampsEnabled = timestampLevel != "off";

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

using var tokenizer = new Tokenizer(model);
using var tokenizerStream = tokenizer.CreateStream();
using var genParams = new GeneratorParams(model);
using var generator = new Generator(model, genParams);
Console.WriteLine(new string('-', 60));
string fullTranscript = "";
var allWordTranscript = new StringBuilder();
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
    fullTranscript += DecodeTokens(generator, tokenizerStream, timestampLevel, allWordTranscript);
  } else {
    chunksSkipped++;
  }
}

// Flush remaining buffered audio
using var flushInputs = processor.Flush();
if (flushInputs != null) {
  generator.SetInputs(flushInputs);
  fullTranscript += DecodeTokens(generator, tokenizerStream, timestampLevel, allWordTranscript);
}

if (!timestampsEnabled) {
  // Ordinary decoding has no pending timestamp records.
} else {
  var result = tokenizerStream.FinalizeTimestamps();
  fullTranscript += FormatTimestampRecords(result, timestampLevel is "segment" or "all");
  if (timestampLevel == "all")
    allWordTranscript.Append(FormatTimestampRecords(result, false));
}

Console.WriteLine($"\n{new string('=', 60)}");
Console.WriteLine($"  {fullTranscript.Trim()}");
if (timestampLevel == "all")
  Console.WriteLine($"  Word timestamps: {allWordTranscript}");
Console.WriteLine(new string('=', 60));
if (useVad == "true") {
  double pctSaved = chunksTotal > 0 ? (double)chunksSkipped / chunksTotal * 100.0 : 0.0;
  Console.WriteLine($"  VAD Metrics: {chunksTotal} total chunks, {chunksProcessed} processed, " +
                    $"{chunksSkipped} skipped ({pctSaved:F1}% compute saved)");
}

static string DecodeTokens(Generator generator, TokenizerStream tokenizerStream, string timestampLevel,
                           StringBuilder allWordTranscript) {
  string text = "";
  bool timestampsEnabled = timestampLevel != "off";
  while (!generator.IsDone()) {
    generator.GenerateNextToken();
    if (!timestampsEnabled) {
      var tokens = generator.GetNextTokens();
      if (tokens.Length == 0)
        continue;
      string tokenText = tokenizerStream.Decode(tokens[0]);
      if (!string.IsNullOrEmpty(tokenText)) {
        Console.Write(tokenText);
        text += tokenText;
      }
    } else {
      foreach (var token in generator.GetNextTokensWithTimings()) {
        var result = tokenizerStream.DecodeWithTimestamps(token);
        string timestampedText = FormatTimestampRecords(result, timestampLevel is "segment" or "all");
        if (timestampLevel == "all")
          allWordTranscript.Append(FormatTimestampRecords(result, false));
        Console.Write(timestampedText);
        text += timestampedText;
      }
    }
  }
  return text;
}

static string FormatTimestampRecords(TimestampDecodeResult result, bool useSegments) {
  var records = useSegments ? result.Segments : result.Words;
  return string.Concat(records.Select(record => {
    string text = useSegments ? record.Text : record.Text.Trim();
    string separator = useSegments && (text.Length == 0 || !char.IsWhiteSpace(text[0])) ? " " : "";
    string suffix = useSegments ? "" : " ";
    return $"[{record.StartTime:F2} - {record.StopTime:F2}]{separator}{text}{suffix}";
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
