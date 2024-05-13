package ai.onnxruntime_genai;

/**
 * The `SimpleGenAI` class provides a simple usage example of the GenAI API.
 * 
 * Usage:
 *  - Create an instance of the class with the path to the model.
 *  - Call createGeneratorParams with the prompt text.
 *  - Set any other search options via the GeneratorParams object as needed using `setSearchOption`.
 *  - Call generate with the GeneratorParams object and an optional listener.
 * 
 * The listener is used as a callback mechanism so that tokens can be used as they are generated.
 * Create a class that implements the TokenUpdateListener interface and provide an instance of that class as the 
 * `listener` argument.
 */
public class SimpleGenAI {
    private Model model;

    public SimpleGenAI(String modelPath) throws GenAIException {
        model = new Model(modelPath);
    }

    /**
     * This interface represents a listener for token updates.
     * When a new token is generated, the listener will be called with the token.
     * <p>
     * WARNING: Generation of the next token will be blocked until the listener returns.
     */
    public interface TokenUpdateListener {
        /**
         * Called when a new token is generated.
         *
         * @param token The new token.
         */
        void onTokenGenerate(String token);
    }

    /**
     * Create the generator parameters and add the prompt text.
     * The user can set other search options via the GeneratorParams object prior to running `generate`.
     *
     * @param prompt The prompt text to encode.
     * @return The encoded sequences.
     * @throws GenAIException
     */
    GeneratorParams createGeneratorParams(String prompt) throws GenAIException {
        try (Tokenizer tokenizer = new Tokenizer(model);
                Sequences prompt_sequences = tokenizer.Encode(prompt);
                GeneratorParams generatorParams = new GeneratorParams(model)) {
            // add the prompt and set max_length as example usage
            generatorParams.setInput(prompt_sequences);
            generatorParams.setSearchOption("max_length", 200);

            return generatorParams;
        } catch (Exception e) {
            e.printStackTrace();
            throw new GenAIException("Prompt encoding failed", e);
        }
    }

    /**
     * Generate text based on the prompt and settings in GeneratorParams.
     * <p>
     * NOTE: This only handles a single sequence of input (i.e. a single prompt which equates to batch size of 1)
     * @param generatorParams The prompt and settings to run the model with.
     * @param listener Optional callback for tokens to be provided as they are generated.
     * @return The generated text.
     * @throws GenAIException
     */
    public String generate(GeneratorParams generatorParams, TokenUpdateListener listener) throws GenAIException {
        String result = null;
        try (Tokenizer tokenizer = new Tokenizer(model)) {
            int[] output_ids = null;

            if (listener != null) {
                try (TokenizerStream stream = tokenizer.CreateStream();
                        Generator generator = new Generator(model, generatorParams)) {
                    while (!generator.isDone()) {
                        // generate next token
                        generator.computeLogits();
                        generator.generateNextToken();

                        // decode and call listener
                        int token_id = generator.getLastTokenInSequence(0);
                        String token = stream.decode(token_id);
                        listener.onTokenGenerate(token);
                    }

                    output_ids = generator.getSequence(0);
                } catch (Exception e) {
                    handleException("Token generation loop failed.", e);
                }
            } else {
                Sequences output_sequences = model.generate(generatorParams);
                output_ids = output_sequences.getSequence(0);
            }

            result = tokenizer.Decode(output_ids);
        } catch (Exception e) {
            handleException("Failed to create Tokenizer", e);
        }

        return result;
    }

    private static void handleException(String reason, Exception e) throws GenAIException {
        e.printStackTrace();
        throw new GenAIException(reason, e);
    }
}
