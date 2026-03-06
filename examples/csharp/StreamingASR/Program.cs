// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntimeGenAI;

if (args.Length < 2)
{
    Console.WriteLine("Usage: StreamingASR <model_path> <audio_file.wav>");
    return;
}

string modelPath = args[0];
string audioFile = args[1];

// Load raw PCM audio (expects 16kHz mono float32 WAV)
float[] audio = LoadWavAudio(audioFile);
Console.WriteLine($"Audio: {audio.Length / 16000.0:F1}s ({audio.Length} samples)");

using var model = new Model(modelPath);
using var asr = new StreamingASR(model);

int chunkSize = 8960; // 560ms chunks
Console.WriteLine(new string('-', 60));

for (int i = 0; i < audio.Length; i += chunkSize)
{
    int remaining = Math.Min(chunkSize, audio.Length - i);
    float[] chunk = new float[chunkSize];
    Array.Copy(audio, i, chunk, 0, remaining);

    string text = asr.TranscribeChunk(chunk);
    if (!string.IsNullOrEmpty(text))
    {
        Console.Write(text);
    }
}

// Flush remaining audio
string flushText = asr.Flush();
if (!string.IsNullOrEmpty(flushText))
{
    Console.Write(flushText);
}

// Final transcript
string transcript = asr.GetTranscript();
Console.WriteLine($"\n{new string('=', 60)}");
Console.WriteLine($"  {transcript.Trim()}");
Console.WriteLine(new string('=', 60));

static float[] LoadWavAudio(string path)
{
    using var reader = new BinaryReader(File.OpenRead(path));

    // Skip RIFF header (44 bytes for standard WAV)
    reader.ReadBytes(44);

    var samples = new List<float>();
    while (reader.BaseStream.Position < reader.BaseStream.Length)
    {
        try
        {
            short sample = reader.ReadInt16();
            samples.Add(sample / 32768.0f);
        }
        catch (EndOfStreamException)
        {
            break;
        }
    }
    return samples.ToArray();
}
