using Microsoft.Extensions.AI;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.OnnxRuntimeGenAI;

/// <summary>Provides configuration options used when constructing a <see cref="ChatClient"/>.</summary>
/// <remarks>
/// Every model has different requirements for stop sequences and prompt formatting. For best results,
/// the configuration should be tailored to the exact nature of the model being used. For example,
/// when using a Phi3 model, a configuration like the following may be used:
/// <code>
/// static ChatClientConfiguration CreateForPhi3() =&gt;
///     new(["&lt;|system|&gt;", "&lt;|user|&gt;", "&lt;|assistant|&gt;", "&lt;|end|&gt;"],
///         (IEnumerable&lt;ChatMessage&gt; messages) =&gt;
///         {
///             StringBuilder prompt = new();
///             
///             foreach (var message in messages)
///                 foreach (var content in message.Contents.OfType&lt;TextContent&gt;())
///                     prompt.Append("&lt;|").Append(message.Role.Value).Append("|&gt;\n").Append(tc.Text).Append("&lt;|end|&gt;\n");
///             
///             return prompt.Append("&lt;|assistant|&gt;\n").ToString();
///         });
/// </code>
/// </remarks>
public sealed class ChatClientConfiguration
{
    private string[] _stopSequences;
    private Func<IEnumerable<ChatMessage>, string> _promptFormatter;

    /// <summary>Initializes a new instance of the <see cref="ChatClientConfiguration"/> class.</summary>
    /// <param name="stopSequences">The stop sequences used by the model.</param>
    /// <param name="promptFormatter">The function to use to format a list of messages for input into the model.</param>
    /// <exception cref="ArgumentNullException"><paramref name="stopSequences"/> is null.</exception>
    /// <exception cref="ArgumentNullException"><paramref name="promptFormatter"/> is null.</exception>
    public ChatClientConfiguration(
        string[] stopSequences,
        Func<IEnumerable<ChatMessage>, string> promptFormatter)
    {
        if (stopSequences is null)
        {
            throw new ArgumentNullException(nameof(stopSequences));
        }

        if (promptFormatter is null)
        {
            throw new ArgumentNullException(nameof(promptFormatter));
        }

        StopSequences = stopSequences;
        PromptFormatter = promptFormatter;
    }

    /// <summary>
    /// Gets or sets stop sequences to use during generation.
    /// </summary>
    /// <remarks>
    /// These will apply in addition to any stop sequences that are a part of the <see cref="ChatOptions.StopSequences"/>.
    /// </remarks>
    public string[] StopSequences
    {
        get => _stopSequences;
        set => _stopSequences = value ?? throw new ArgumentNullException(nameof(value));
    }

    /// <summary>Gets the function that creates a prompt string from the chat history.</summary>
    public Func<IEnumerable<ChatMessage>, string> PromptFormatter
    {
        get => _promptFormatter;
        set => _promptFormatter = value ?? throw new ArgumentNullException(nameof(value));
    }
}