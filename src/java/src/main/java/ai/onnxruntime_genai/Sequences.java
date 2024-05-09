/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime_genai;

public class Sequences {
    private long sequencesHandle;
    private long numSequences;

    public Sequences(long sequencesHandle) {
        this.sequencesHandle = sequencesHandle;
        numSequences = getSequencesCount(sequencesHandle);
    }

    public long NumSequences() {
        return numSequences;
    }

    static {
        try {
            GenAI.init();
        } catch (Exception e) {
            throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
        }
    }
    private native long getSequencesCount(long sequencesHandle);
}