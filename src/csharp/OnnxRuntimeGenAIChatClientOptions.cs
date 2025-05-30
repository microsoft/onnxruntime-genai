﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using Microsoft.Extensions.AI;

#nullable enable

namespace Microsoft.ML.OnnxRuntimeGenAI;

/// <summary>Provides configuration options used when constructing a <see cref="OnnxRuntimeGenAIChatClient"/>.</summary>
/// <remarks>
/// Every model has different requirements for stop sequences and prompt formatting. For best results,
/// the configuration should be tailored to the exact nature of the model being used. For example,
/// when using a Phi3 model, a configuration like the following may be used:
/// <code>
/// static OnnxRuntimeGenAIChatClientOptions CreateForPhi3() =&gt; new()
/// {
///     StopSequences = ["&lt;|system|&gt;", "&lt;|user|&gt;", "&lt;|assistant|&gt;", "&lt;|end|&gt;"],
///     PromptFormatter = (IEnumerable&lt;ChatMessage&gt; messages, ChatOptions? options) =&gt;
///     {
///         StringBuilder prompt = new();
///
///         foreach (var message in messages)
///             foreach (var content in message.Contents.OfType&lt;TextContent&gt;())
///                 prompt.Append("&lt;|").Append(message.Role.Value).Append("|&gt;\n").Append(content.Text).Append("&lt;|end|&gt;\n");
///
///         return prompt.Append("&lt;|assistant|&gt;\n").ToString();
///     });
/// };
/// </code>
/// </remarks>
public sealed class OnnxRuntimeGenAIChatClientOptions
{
    /// <summary>Initializes a new instance of the <see cref="OnnxRuntimeGenAIChatClientOptions"/> class.</summary>
    /// <param name="stopSequences">The stop sequences used by the model.</param>
    /// <param name="promptFormatter">The function to use to format a list of messages for input into the model.</param>
    /// <exception cref="ArgumentNullException"><paramref name="stopSequences"/> is null.</exception>
    /// <exception cref="ArgumentNullException"><paramref name="promptFormatter"/> is null.</exception>
    public OnnxRuntimeGenAIChatClientOptions()
    {
    }

    /// <summary>
    /// Gets or sets stop sequences to use during generation.
    /// </summary>
    /// <remarks>
    /// These will apply in addition to any stop sequences that are a part of the <see cref="ChatOptions.StopSequences"/>
    /// provided to the <see cref="IChatClient.GetResponseAsync"/> and <see cref="IChatClient.GetStreamingResponseAsync"/>
    /// methods. If <see langword="null"/>, this will not contribute any additional stop sequences.
    /// </remarks>
    public IList<string>? StopSequences { get; set; }

    /// <summary>Gets or sets a delegate that formats a prompt string from a list of chat messages.</summary>
    /// <remarks>
    /// Each time <see cref="IChatClient.GetResponseAsync"/> or <see cref="IChatClient.GetStreamingResponseAsync"/>
    /// is invoked, this delegate will be invoked with the supplied list of messages to produce a string that
    /// will be tokenized and provided to the underlying <see cref="Generator"/>. If <see langword="null"/>,
    /// the <see cref="OnnxRuntimeGenAIChatClient"/> will choose a default prompt formatter to employ.
    /// </remarks>
    public Func<IEnumerable<ChatMessage>, ChatOptions?, string>? PromptFormatter { get; set; }

    /// <summary>Gets or sets whether to cache the most recent conversation.</summary>
    /// <remarks>
    /// This should only be set to <see langword="true"/> when the <see cref="OnnxRuntimeGenAIChatClient"/>
    /// is not shared across multiple uses. Only one thread of conversation is cached at any given time.
    /// If multiple consumers interleave their use, the implementation may lose the context of the previous
    /// conversation and produce less ideal responses.
    /// </remarks>
    public bool EnableCaching { get; set; }
}