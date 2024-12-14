using Microsoft.Extensions.AI;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.OnnxRuntimeGenAI;

/// <summary>Provides an <see cref="IChatClient"/> implementation for interacting with a <see cref="Model"/>.</summary>
public sealed partial class ChatClient : IChatClient
{
    /// <summary>The options used to configure the instance.</summary>
    private readonly ChatClientConfiguration _config;
    /// <summary>The wrapped <see cref="Model"/>.</summary>
    private readonly Model _model;
    /// <summary>The wrapped <see cref="Tokenizer"/>.</summary>
    private readonly Tokenizer _tokenizer;
    /// <summary>Whether to dispose of <see cref="_model"/> when this instance is disposed.</summary>
    private readonly bool _ownsModel;

    /// <summary>Initializes an instance of the <see cref="ChatClient"/> class.</summary>
    /// <param name="configuration">Options used to configure the client instance.</param>
    /// <param name="modelPath">The file path to the model to load.</param>
    /// <exception cref="ArgumentNullException"><paramref name="modelPath"/> is null.</exception>
    public ChatClient(ChatClientConfiguration configuration, string modelPath)
    {
        if (configuration is null)
        {
            throw new ArgumentNullException(nameof(configuration));
        }

        if (modelPath is null)
        {
            throw new ArgumentNullException(nameof(modelPath));
        }

        _config = configuration;

        _ownsModel = true;
        _model = new Model(modelPath);
        _tokenizer = new Tokenizer(_model);

        Metadata = new("onnxruntime-genai", new Uri($"file://{modelPath}"), modelPath);
    }

    /// <summary>Initializes an instance of the <see cref="ChatClient"/> class.</summary>
    /// <param name="configuration">Options used to configure the client instance.</param>
    /// <param name="model">The model to employ.</param>
    /// <param name="ownsModel">
    /// <see langword="true"/> if this <see cref="IChatClient"/> owns the <paramref name="model"/> and should
    /// dispose of it when this <see cref="IChatClient"/> is disposed; otherwise, <see langword="false"/>.
    /// The default is <see langword="true"/>.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="model"/> is null.</exception>
    public ChatClient(ChatClientConfiguration configuration, Model model, bool ownsModel = true)
    {
        if (configuration is null)
        {
            throw new ArgumentNullException(nameof(configuration));
        }

        if (model is null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        _config = configuration;

        _ownsModel = ownsModel;
        _model = model;
        _tokenizer = new Tokenizer(_model);

        Metadata = new("onnxruntime-genai");
    }

    /// <inheritdoc/>
    public ChatClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public void Dispose()
    {
        _tokenizer.Dispose();

        if (_ownsModel)
        {
            _model.Dispose();
        }
    }

    /// <inheritdoc/>
    public async Task<ChatCompletion> CompleteAsync(IList<ChatMessage> chatMessages, ChatOptions options = null, CancellationToken cancellationToken = default)
    {
        if (chatMessages is null)
        {
            throw new ArgumentNullException(nameof(chatMessages));
        }

        StringBuilder text = new();
        await Task.Run(() =>
        {
            using Sequences tokens = _tokenizer.Encode(_config.PromptFormatter(chatMessages));
            using GeneratorParams generatorParams = new(_model);
            UpdateGeneratorParamsFromOptions(tokens[0].Length, generatorParams, options);

            using Generator generator = new(_model, generatorParams);
            generator.AppendTokenSequences(tokens);

            using var tokenizerStream = _tokenizer.CreateStream();

            var completionId = Guid.NewGuid().ToString();
            while (!generator.IsDone())
            {
                cancellationToken.ThrowIfCancellationRequested();

                generator.GenerateNextToken();

                ReadOnlySpan<int> outputSequence = generator.GetSequence(0);
                string next = tokenizerStream.Decode(outputSequence[outputSequence.Length - 1]);

                if (IsStop(next, options))
                {
                    break;
                }

                text.Append(next);
            }
        }, cancellationToken);

        return new ChatCompletion(new ChatMessage(ChatRole.Assistant, text.ToString()))
        {
            CompletionId = Guid.NewGuid().ToString(),
            CreatedAt = DateTimeOffset.UtcNow,
            ModelId = Metadata.ModelId,
        };
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<StreamingChatCompletionUpdate> CompleteStreamingAsync(
        IList<ChatMessage> chatMessages, ChatOptions options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (chatMessages is null)
        {
            throw new ArgumentNullException(nameof(chatMessages));
        }

        using Sequences tokens = _tokenizer.Encode(_config.PromptFormatter(chatMessages));
        using GeneratorParams generatorParams = new(_model);
        UpdateGeneratorParamsFromOptions(tokens[0].Length, generatorParams, options);

        using Generator generator = new(_model, generatorParams);
        generator.AppendTokenSequences(tokens);

        using var tokenizerStream = _tokenizer.CreateStream();

        var completionId = Guid.NewGuid().ToString();
        while (!generator.IsDone())
        {
            string next = await Task.Run(() =>
            {
                generator.GenerateNextToken();

                ReadOnlySpan<int> outputSequence = generator.GetSequence(0);
                return tokenizerStream.Decode(outputSequence[outputSequence.Length - 1]);
            }, cancellationToken);

            if (IsStop(next, options))
            {
                break;
            }

            yield return new StreamingChatCompletionUpdate
            {
                CompletionId = completionId,
                CreatedAt = DateTimeOffset.UtcNow,
                Role = ChatRole.Assistant,
                Text = next,
            };
        }
    }

    /// <inheritdoc/>
    public object GetService(Type serviceType, object key = null) =>
        key is not null ? null :
        serviceType == typeof(Model) ? _model :
        serviceType == typeof(Tokenizer) ? _tokenizer :
        serviceType?.IsInstanceOfType(this) is true ? this :
        null;

    /// <summary>Gets whether the specified token is a stop sequence.</summary>
    private bool IsStop(string token, ChatOptions options) =>
        options?.StopSequences?.Contains(token) is true ||
        Array.IndexOf(_config.StopSequences, token) >= 0;

    /// <summary>Updates the <paramref name="generatorParams"/> based on the supplied <paramref name="options"/>.</summary>
    private static void UpdateGeneratorParamsFromOptions(int numInputTokens, GeneratorParams generatorParams, ChatOptions options)
    {
        if (options is null)
        {
            return;
        }

        if (options.MaxOutputTokens.HasValue)
        {
            generatorParams.SetSearchOption("max_length", numInputTokens + options.MaxOutputTokens.Value);
        }

        if (options.Temperature.HasValue)
        {
            generatorParams.SetSearchOption("temperature", options.Temperature.Value);
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
        }

        if (options.Seed.HasValue)
        {
            generatorParams.SetSearchOption("random_seed", options.Seed.Value);
        }

        if (options.AdditionalProperties is { } props)
        {
            foreach (var entry in props)
            {
                switch (entry.Value)
                {
                    case int i: generatorParams.SetSearchOption(entry.Key, i); break;
                    case long l: generatorParams.SetSearchOption(entry.Key, l); break;
                    case float f: generatorParams.SetSearchOption(entry.Key, f); break;
                    case double d: generatorParams.SetSearchOption(entry.Key, d); break;
                    case bool b: generatorParams.SetSearchOption(entry.Key, b); break;
                }
            }
        }
    }
}