using System;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace SignalProcessorRunner
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Running SignalProcessor.TestSplitSignalSegments() ...");
            SignalProcessor.TestSplitSignalSegments();
        }
    }
}
