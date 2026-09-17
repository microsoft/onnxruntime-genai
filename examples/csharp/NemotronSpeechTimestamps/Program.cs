// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using CommonUtils;
using Microsoft.ML.OnnxRuntimeGenAI;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using System.Text;
using System.Text.Json;

if (args.Length < 2) {
  Console.WriteLine("Usage: NemotronSpeechTimestamps <model_path> <audio_file.wav> [execution_provider] [--use_vad true]");
  return;
}

string modelPath = args[0];
string audioFile = args[1];
string executionProvider = "follow_config";
string useVadOverride = "";

for (int index = 2; index < args.Length; index++) {
  if (args[index] == "--use_vad" && index + 1 < args.Length) {
    useVadOverride = args[++index];
  } else {
    executionProvider = args[index];
  }
}

using var configJson = JsonDocument.Parse(File.ReadAllText(Path.Combine(modelPath, "genai_config.json")));
var modelConfig = configJson.RootElement.GetProperty("model");
int sampleRate = modelConfig.GetProperty("sample_rate").GetInt32();
int chunkSize = modelConfig.GetProperty("chunk_samples").GetInt32();
float[] audio = LoadAudio(audioFile, sampleRate);

using var config = Common.GetConfig(modelPath, executionProvider, null, new GeneratorParamsArgs());
config.Overlay("{\"model\":{\"timestamp_level\":\"segment\"}}");
using var model = new Model(config);
using var processor = new StreamingProcessor(model);
processor.SetOption("use_vad", "false");
if (useVadOverride == "true") {
  processor.SetOption("use_vad", "true");
}

using var tokenizer = new Tokenizer(model);
using var tokenizerStream = tokenizer.CreateStream();
using var generatorParams = new GeneratorParams(model);
using var generator = new Generator(model, generatorParams);
var timestampedTranscript = new StringBuilder();

for (int offset = 0; offset < audio.Length; offset += chunkSize) {
  int sampleCount = Math.Min(chunkSize, audio.Length - offset);
  float[] chunk = new float[sampleCount];
  Array.Copy(audio, offset, chunk, 0, sampleCount);

  using var inputs = processor.Process(chunk);
  if (inputs != null) {
    generator.SetInputs(inputs);
    DecodeSegments(generator, tokenizerStream, timestampedTranscript);
  }
}

using var flushInputs = processor.Flush();
if (flushInputs != null) {
  generator.SetInputs(flushInputs);
  DecodeSegments(generator, tokenizerStream, timestampedTranscript);
}

AppendSegments(tokenizerStream.FinalizeTimestamps(), timestampedTranscript);

Console.WriteLine();
Console.WriteLine(new string('=', 60));
Console.WriteLine(timestampedTranscript.ToString());
Console.WriteLine(new string('=', 60));

static void DecodeSegments(Generator generator, TokenizerStream tokenizerStream,
                           StringBuilder timestampedTranscript) {
  while (!generator.IsDone()) {
    generator.GenerateNextToken();
    foreach (var token in generator.GetNextTokensWithTimings()) {
      AppendSegments(tokenizerStream.DecodeWithTimestamps(token), timestampedTranscript);
    }
  }
}

static void AppendSegments(TimestampDecodeResult result, StringBuilder timestampedTranscript) {
  foreach (var segment in result.Segments) {
    string separator = segment.Text.Length > 0 && char.IsWhiteSpace(segment.Text[0]) ? "" : " ";
    string timestampedSegment = $"[{segment.StartTime:F2} - {segment.StopTime:F2}]{separator}{segment.Text}";
    timestampedTranscript.Append(timestampedSegment);
    Console.Write(timestampedSegment);
  }
}

static float[] LoadAudio(string path, int targetSampleRate) {
  using var reader = new AudioFileReader(path);
  ISampleProvider source = reader;
  if (reader.WaveFormat.Channels > 1) {
    source = new StereoToMonoSampleProvider(source);
  }
  if (reader.WaveFormat.SampleRate != targetSampleRate) {
    source = new WdlResamplingSampleProvider(source, targetSampleRate);
  }

  var samples = new List<float>();
  float[] buffer = new float[4096];
  int read;
  while ((read = source.Read(buffer, 0, buffer.Length)) > 0) {
    for (int index = 0; index < read; index++) {
      samples.Add(buffer[index]);
    }
  }
  return samples.ToArray();
}