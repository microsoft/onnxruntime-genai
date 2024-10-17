using Microsoft.Extensions.AI;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.OnnxRuntimeGenAI;

/// <summary>An <see cref="IChatClient"/> implementation based on ONNX Runtime GenAI.</summary>
public sealed class ChatClient : IChatClient, IDisposable
{
    /// <summary>The wrapped <see cref="Model"/>.</summary>
    private readonly Model _model;
    /// <summary>The wrapped <see cref="Tokenizer"/>.</summary>
    private readonly Tokenizer _tokenizer;
    /// <summary>Whether to dispose of <see cref="_model"/> when this instance is disposed.</summary>
    private readonly bool _ownsModel;

    /// <summary>Initializes an instance of the <see cref="ChatClient"/> class.</summary>
    /// <param name="modelPath">The file path to the model to load.</param>
    /// <exception cref="ArgumentNullException"><paramref name="modelPath"/> is null.</exception>
    public ChatClient(string modelPath)
    {
        if (modelPath is null)
        {
            throw new ArgumentNullException(nameof(modelPath));
        }

        _ownsModel = true;
        _model = new Model(modelPath);
        _tokenizer = new Tokenizer(_model);

        Metadata = new(typeof(ChatClient).Namespace, new Uri($"file://{modelPath}"), modelPath);
    }

    /// <summary>Initializes an instance of the <see cref="ChatClient"/> class.</summary>
    /// <param name="model">The model to employ.</param>
    /// <param name="ownsModel">
    /// <see langword="true"/> if this <see cref="IChatClient"/> owns the <paramref name="model"/> and should
    /// dispose of it when this <see cref="IChatClient"/> is disposed; otherwise, <see langword="false"/>.
    /// The default is <see langword="true"/>.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="model"/> is null.</exception>
    public ChatClient(Model model, bool ownsModel = true)
    {
        if (model is null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        _ownsModel = ownsModel;
        _model = model;
        _tokenizer = new Tokenizer(_model);

        Metadata = new("Microsoft.ML.OnnxRuntimeGenAI");
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
    public Task<ChatCompletion> CompleteAsync(IList<ChatMessage> chatMessages, ChatOptions options = null, CancellationToken cancellationToken = default)
    {
        if (chatMessages is null)
        {
            throw new ArgumentNullException(nameof(chatMessages));
        }

        return Task.Run(() =>
        {
            using Sequences tokens = _tokenizer.Encode(CreatePrompt(chatMessages));
            using GeneratorParams generatorParams = new(_model);
            UpdateGeneratorParamsFromOptions(tokens[0].Length, generatorParams, options);
            generatorParams.SetInputSequences(tokens);

            using Generator generator = new(_model, generatorParams);
            using Sequences outputSequences = _model.Generate(generatorParams);

            return new ChatCompletion(new ChatMessage(ChatRole.Assistant, _tokenizer.Decode(outputSequences[0])))
            {
                CompletionId = Guid.NewGuid().ToString(),
                CreatedAt = DateTimeOffset.UtcNow,
                ModelId = Metadata.ModelId,
            };
        }, cancellationToken);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<StreamingChatCompletionUpdate> CompleteStreamingAsync(
        IList<ChatMessage> chatMessages, ChatOptions options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (chatMessages is null)
        {
            throw new ArgumentNullException(nameof(chatMessages));
        }

        using Sequences tokens = _tokenizer.Encode(CreatePrompt(chatMessages));
        using GeneratorParams generatorParams = new(_model);
        UpdateGeneratorParamsFromOptions(tokens[0].Length, generatorParams, options);
        generatorParams.SetInputSequences(tokens);

        using Generator generator = new(_model, generatorParams);
        using var tokenizerStream = _tokenizer.CreateStream();

        var completionId = Guid.NewGuid().ToString();
        while (!generator.IsDone())
        {
            string next = await Task.Run(() =>
            {
                generator.ComputeLogits();
                generator.GenerateNextToken();

                ReadOnlySpan<int> outputSequence = generator.GetSequence(0);
                return tokenizerStream.Decode(outputSequence[outputSequence.Length - 1]);
            }, cancellationToken);

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
    public TService GetService<TService>(object key = null) where TService : class =>
        typeof(TService) == typeof(Model) ? (TService)(object)_model :
        typeof(TService) == typeof(Tokenizer) ? (TService)(object)_tokenizer :
        this as TService;

    /// <summary>Creates a prompt string from the supplied chat history.</summary>
    private string CreatePrompt(IEnumerable<ChatMessage> messages)
    {
        StringBuilder prompt = new();

        foreach (var message in messages)
        {
            foreach (var content in message.Contents)
            {
                switch (content)
                {
                    case TextContent tc when !string.IsNullOrWhiteSpace(tc.Text):
                        prompt.Append("<|").Append(message.Role.Value).Append("|>\n").Append(tc.Text);
                        break;
                }
            }
        }

        return prompt.Append("<|end|>\n<|assistant|>").ToString();
    }

    /// <summary>Updates the <paramref name="generatorParams"/> based on the supplied <paramref name="options"/>.</summary>
    private static void UpdateGeneratorParamsFromOptions(int numInputTokens, GeneratorParams generatorParams, ChatOptions options)
    {
        if (options is null)
        {
            return;
        }

        if (options.Temperature.HasValue)
        {
            generatorParams.SetSearchOption("temperature", options.Temperature.Value);
        }

        if (options.TopP.HasValue)
        {
            generatorParams.SetSearchOption("top_p", options.TopP.Value);
        }

        if (options.MaxOutputTokens.HasValue)
        {
            generatorParams.SetSearchOption("max_length", numInputTokens + options.MaxOutputTokens.Value);
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