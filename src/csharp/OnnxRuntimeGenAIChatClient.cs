﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;

#nullable enable

namespace Microsoft.ML.OnnxRuntimeGenAI;

/// <summary>Provides an <see cref="IChatClient"/> implementation for interacting with an ONNX Runtime GenAI <see cref="Model"/>.</summary>
public sealed class OnnxRuntimeGenAIChatClient : IChatClient
{
    /// <summary>Options used to configure the instance's behavior.</summary>
    private readonly OnnxRuntimeGenAIChatClientOptions? _options;
    /// <summary>The wrapped <see cref="Model"/>.</summary>
    private readonly Model _model;
    /// <summary>The wrapped <see cref="Tokenizer"/>.</summary>
    private readonly Tokenizer _tokenizer;
    /// <summary>Whether to dispose of <see cref="_model"/> when this instance is disposed.</summary>
    private readonly bool _ownsModel;
    /// <summary>Metadata for the chat client.</summary>
    private readonly ChatClientMetadata _metadata;

    /// <summary>Cached information about the last generation to speed up a subsequent generation.</summary>
    /// <remarks>Only one is cached. Interlocked operations are used to take and return an instance from this cache.</remarks>
    private CachedGenerator? _cachedGenerator;

    /// <summary>Initializes an instance of the <see cref="OnnxRuntimeGenAIChatClient"/> class.</summary>
    /// <param name="modelPath">The file path to the model to load.</param>
    /// <param name="options">Options used to configure the client instance.</param>
    /// <exception cref="ArgumentNullException"><paramref name="modelPath"/> is <see langword="null"/>.</exception>
    public OnnxRuntimeGenAIChatClient(string modelPath, OnnxRuntimeGenAIChatClientOptions? options = null)
    {
        if (modelPath is null)
        {
            throw new ArgumentNullException(nameof(modelPath));
        }

        _ownsModel = true;
        _model = new Model(modelPath);
        _tokenizer = new Tokenizer(_model);
        _options = options;

        _metadata = new("onnx", new Uri($"file://{modelPath}"), modelPath);
    }

    /// <summary>Initializes an instance of the <see cref="OnnxRuntimeGenAIChatClient"/> class.</summary>
    /// <param name="model">The model to employ.</param>
    /// <param name="ownsModel">
    /// <see langword="true"/> if this <see cref="IChatClient"/> owns the <paramref name="model"/> and should
    /// dispose of it when this <see cref="IChatClient"/> is disposed; otherwise, <see langword="false"/>.
    /// The default is <see langword="true"/>.
    /// </param>
    /// <param name="options">Options used to configure the client instance.</param>
    /// <exception cref="ArgumentNullException"><paramref name="model"/> is <see langword="null"/>.</exception>
    public OnnxRuntimeGenAIChatClient(Model model, bool ownsModel = true, OnnxRuntimeGenAIChatClientOptions? options = null)
    {
        if (model is null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        _ownsModel = ownsModel;
        _model = model;
        _tokenizer = new Tokenizer(_model);
        _options = options;

        _metadata = new("onnx");
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _cachedGenerator, null) is CachedGenerator cachedGenerator)
        {
            cachedGenerator.Dispose();
        }

        _tokenizer.Dispose();

        if (_ownsModel)
        {
            _model.Dispose();
        }
    }

    /// <inheritdoc/>
    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
        GetStreamingResponseAsync(messages, options, cancellationToken).ToChatResponseAsync(cancellationToken: cancellationToken);

    /// <inheritdoc/>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages, ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (messages is null)
        {
            throw new ArgumentNullException(nameof(messages));
        }

        // Check to see whether there's a cached generator. If there is, and if its id matches what we got from the client,
        // we can use it; otherwise, we need to create a new one.
        CachedGenerator? generator = Interlocked.Exchange(ref _cachedGenerator, null);
        if (generator is null ||
            generator.ChatThreadId is null ||
            generator.ChatThreadId != options?.ChatThreadId)
        {
            generator?.Dispose();

            using GeneratorParams p = new(_model); // we can dispose of this after we create the generator
            UpdateGeneratorParamsFromOptions(p, options);
            generator = new(new Generator(_model, p));
        }

        // If caching is enabled, generate a new ID to represent the state of the generator when we finish this response.
        generator.ChatThreadId = _options?.EnableCaching is true ? Guid.NewGuid().ToString("N") : null;

        // Format and tokenize the message.
        string formattedPrompt = _options?.PromptFormatter is { } formatter ?
            formatter(messages, options) :
            FormatPromptDefault(messages, options);

        using Sequences tokens = _tokenizer.Encode(formattedPrompt);
        try
        {
            string responseId = Guid.NewGuid().ToString("N");

            generator.Generator.AppendTokenSequences(tokens);
            int inputTokens = tokens[0].Length, outputTokens = 0;

            // Loop while we still want to produce more tokens.
            using var tokenizerStream = _tokenizer.CreateStream();
            while (!generator.Generator.IsDone())
            {
                // If we've reached a max output token limit, stop.
                if (options?.MaxOutputTokens is int maxOutputTokens &&
                    outputTokens >= maxOutputTokens)
                {
                    break;
                }

                // Avoid blocking calling thread with expensive compute
                cancellationToken.ThrowIfCancellationRequested();
                await YieldAwaiter.Instance;

                // Generate the next token.
                generator.Generator.GenerateNextToken();
                string next = tokenizerStream.Decode(GetLastToken(generator.Generator.GetSequence(0)));

                // workaround until C# 13 is adopted and ref locals are usable in async methods
                static int GetLastToken(ReadOnlySpan<int> span) => span[span.Length - 1];

                // If this token is a stop token, bail.
                if (IsStop(next, options))
                {
                    break;
                }

                // Yield the next token in the stream.
                outputTokens++;
                yield return new(ChatRole.Assistant, next)
                {
                    CreatedAt = DateTimeOffset.UtcNow,
                    ResponseId = responseId,
                };
            }

            // Yield a final update containing metadata.
            yield return new()
            {
                ChatThreadId = generator.ChatThreadId,
                Contents = [new UsageContent(new()
                {
                    InputTokenCount = inputTokens,
                    OutputTokenCount = outputTokens,
                    TotalTokenCount = inputTokens + outputTokens,
                })],
                CreatedAt = DateTimeOffset.UtcNow,
                FinishReason = options is not null && options.MaxOutputTokens <= outputTokens ? ChatFinishReason.Length : ChatFinishReason.Stop,
                ModelId = _metadata.ModelId,
                ResponseId = responseId,
                Role = ChatRole.Assistant,
            };
        }
        finally
        {
            // Cache the generator for subsequent use if it's cachable and there isn't already a generator cached.
            if (generator.ChatThreadId is null ||
                Interlocked.CompareExchange(ref _cachedGenerator, generator, null) != null)
            {
                generator.Dispose();
            }
        }
    }

    /// <inheritdoc/>
    object? IChatClient.GetService(Type serviceType, object? serviceKey)
    {
        if (serviceType is null)
        {
            throw new ArgumentNullException(nameof(serviceType));
        }

        return 
            serviceKey is not null ? null :
            serviceType == typeof(ChatClientMetadata) ? _metadata :
            serviceType == typeof(Model) ? _model :
            serviceType == typeof(Tokenizer) ? _tokenizer :
            serviceType?.IsInstanceOfType(this) is true ? this :
            null;
    }

    /// <summary>Gets whether the specified token is a stop sequence.</summary>
    private bool IsStop(string token, ChatOptions? options) =>
        options?.StopSequences?.Contains(token) is true ||
        _options?.StopSequences?.Contains(token) is true;

    /// <summary>Formats messages into a prompt using a default format.</summary>
    private static string FormatPromptDefault(IEnumerable<ChatMessage> messages, ChatOptions? options)
    {
        StringBuilder sb = new();
        foreach (var message in messages)
        {
            sb.Append(message).AppendLine();
}

        return sb.ToString();
    }

    /// <summary>Updates the <paramref name="generatorParams"/> based on the supplied <paramref name="options"/>.</summary>
    private static void UpdateGeneratorParamsFromOptions(GeneratorParams generatorParams, ChatOptions? options)
    {
        if (options is null)
        {
            return;
        }

        if (options.Temperature.HasValue)
        {
            generatorParams.SetSearchOption("temperature", options.Temperature.Value);
        }

        if (options.PresencePenalty.HasValue)
        {
            generatorParams.SetSearchOption("repetition_penalty", options.PresencePenalty.Value);
        }

        if (options.TopP.HasValue || options.TopK.HasValue)
        {
            if (options.TopP.HasValue)
            {
                generatorParams.SetSearchOption("top_p", options.TopP.Value);
            }

            if (options.TopK.HasValue)
            {
                generatorParams.SetSearchOption("top_k", options.TopK.Value);
            }

            generatorParams.SetSearchOption("do_sample", true);
        }

        if (options.Seed.HasValue)
        {
            generatorParams.SetSearchOption("random_seed", options.Seed.Value);
        }

        if (options.AdditionalProperties is { } props)
        {
            foreach (var entry in props)
            {
                if (entry.Value is bool b)
                {
                    generatorParams.SetSearchOption(entry.Key, b);
                }
                else if (entry.Value is not null)
                {
                    try
                    {
                        double d = Convert.ToDouble(entry.Value);
                        generatorParams.SetSearchOption(entry.Key, d);
                    }
                    catch
                    {
                        // Ignore values we can't convert
                    }
                }
            }
        }
    }

    private sealed class CachedGenerator(Generator generator) : IDisposable
    {
        public Generator Generator { get; } = generator;

        public string? ChatThreadId { get; set; }

        public void Dispose() => Generator?.Dispose();
    }

    /// <summary>Polyfill for Task.CompletedTask.ConfigureAwait(ConfigureAwaitOptions.ForceYielding);</summary>
    private sealed class YieldAwaiter : INotifyCompletion
    {
        public static YieldAwaiter Instance { get; } = new();
        public YieldAwaiter GetAwaiter() => this;
        public bool IsCompleted => false;
        public void OnCompleted(Action continuation) => Task.Run(continuation);
        public void UnsafeOnCompleted(Action continuation) => Task.Run(continuation);
        public void GetResult() { }
    }
}