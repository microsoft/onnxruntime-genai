/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime_genai;

public class Sequences implements AutoCloseable {
    private long sequencesHandle;
    private long numSequences;

    // sequencesHandle is created by Toeknzier
    protected Sequences(long sequencesHandle) {
        assert(sequencesHandle != 0);  //internal usage should never pass an invalid handle
        
        this.sequencesHandle = sequencesHandle;
        numSequences = getSequencesCount(sequencesHandle);
    }

    public long NumSequences() {
        return numSequences;
    }

    int[] GetSequence(long sequenceIndex) {
        return getSequence(sequencesHandle, sequenceIndex);
    }

    @Override
    public void close() throws Exception {
        if (sequencesHandle != 0) {
            destroySequences(sequencesHandle);
            sequencesHandle = 0;
        }
    }

    static {
        try {
            GenAI.init();
        } catch (Exception e) {
            throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
        }
    }

    private native long getSequencesCount(long sequencesHandle);
    private native int[] getSequence(long sequencesHandle, long sequenceIndex);
    private native void destroySequences(long sequencesHandle);
}