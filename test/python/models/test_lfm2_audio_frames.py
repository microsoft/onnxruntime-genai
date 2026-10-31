# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
LFM2-Audio frame and token accounting tests.

Three pieces have to agree on how many decoder tokens a clip is worth, or the encoder output and the
audio placeholders in the prompt will not line up:

  * NeMo's `AudioToMelSpectrogramPreprocessor`, which the reference implementation wraps: its STFT
    decides how many mel frames a clip has, and its `get_seq_len` decides how many of those count as
    valid for the per-feature normalization,
  * the FastConformer encoder, whose three stride-2 convolutions subsample those frames by 8,
  * `Lfm2AudioProcessor` in C++, which writes one placeholder per encoder frame.

These tests are pure arithmetic reimplementations of all three, so they run without a model or a
built runtime. The real C++ processor is exercised against the tiny test model in
test_lfm2_audio_models.py, and `test_matches_liquid_audio` below checks this arithmetic against the
reference implementation itself whenever it is installed.

Run with:
    python -m pytest test/python/models/test_lfm2_audio_frames.py -v
"""

import math

import pytest

# Defaults from LiquidAI/LFM2.5-Audio-1.5B's config.json ("preprocessor" and "encoder"), shared
# byte for byte by LFM2-Audio-1.5B and LFM2.5-Audio-1.5B-JP.
SAMPLE_RATE = 16000
WINDOW_SIZE = 0.025  # seconds
WINDOW_STRIDE = 0.01  # seconds
WIN_LENGTH = int(WINDOW_SIZE * SAMPLE_RATE)  # 400
HOP_LENGTH = int(WINDOW_STRIDE * SAMPLE_RATE)  # 160
FFT_SIZE = 512
NUM_MELS = 128
# Three stride-2 depthwise convolutions in the encoder's pre-encode stage.
SUBSAMPLING_FACTOR = 8
# The shortest clip the per-feature normalization can handle: its standard deviation divides by
# (valid frames - 1), so one valid frame is not enough.
MIN_VALID_FRAMES = 2


def num_mel_frames(num_samples: int) -> int:
    """Frames out of `torch.stft(..., center=True)`: one per hop, plus one for the centre padding."""
    return num_samples // HOP_LENGTH + 1


def num_valid_mel_frames(num_samples: int) -> int:
    """NeMo's `get_seq_len`: the frames the per-feature normalization averages over.

    One fewer than the STFT produces. The extra frame is still handed to the encoder and still
    counts towards the decoder tokens; it is just zeroed instead of normalized.
    """
    return num_samples // HOP_LENGTH


def num_audio_tokens(num_frames: int) -> int:
    """Decoder tokens for a clip: one per encoder frame, the tail frame rounded up."""
    return math.ceil(num_frames / SUBSAMPLING_FACTOR)


def seconds_to_tokens(seconds: float) -> int:
    return num_audio_tokens(num_mel_frames(int(seconds * SAMPLE_RATE)))


class TestMelFrames:
    @pytest.mark.parametrize(
        "num_samples,frames,valid",
        [
            (0, 1, 0),  # centre padding alone still yields a frame
            (1, 1, 0),
            (159, 1, 0),
            (160, 2, 1),  # exactly one hop
            (16000, 101, 100),  # one second
            (16159, 101, 100),  # a partial hop is truncated
            (16160, 102, 101),
        ],
    )
    def test_known_lengths(self, num_samples, frames, valid):
        assert num_mel_frames(num_samples) == frames
        assert num_valid_mel_frames(num_samples) == valid

    @pytest.mark.parametrize("num_samples", [0, 1, 160, 4000, 16000, 44100, 160000])
    def test_valid_frames_are_one_short_of_the_stft(self, num_samples):
        # This off-by-one is the whole reason the two counts exist separately: the last frame is
        # zeroed by the normalization but still reaches the encoder.
        assert num_valid_mel_frames(num_samples) == num_mel_frames(num_samples) - 1

    def test_one_second_is_one_hundred_hops(self):
        assert num_valid_mel_frames(SAMPLE_RATE) == int(1.0 / WINDOW_STRIDE)

    @pytest.mark.parametrize("num_samples", [320, 1000, 16000, 100000])
    def test_frame_count_is_monotonic_in_the_clip_length(self, num_samples):
        assert num_mel_frames(num_samples) <= num_mel_frames(num_samples + 1)
        assert num_mel_frames(num_samples) < num_mel_frames(num_samples + HOP_LENGTH)

    def test_shortest_normalizable_clip(self):
        # Two valid frames need two whole hops; anything shorter the processor rejects.
        shortest = MIN_VALID_FRAMES * HOP_LENGTH
        assert num_valid_mel_frames(shortest) == MIN_VALID_FRAMES
        assert num_valid_mel_frames(shortest - 1) < MIN_VALID_FRAMES

    def test_window_fits_inside_the_fft(self):
        # A window longer than the FFT size cannot be centred in it; the processor refuses that.
        assert WIN_LENGTH <= FFT_SIZE
        assert (FFT_SIZE - WIN_LENGTH) % 2 == 0, "the window is centred, so the padding has to split evenly"


class TestAudioTokens:
    @pytest.mark.parametrize(
        "frames,tokens",
        [(0, 0), (1, 1), (8, 1), (9, 2), (16, 2), (101, 13), (126, 16), (1001, 126)],
    )
    def test_known_frame_counts(self, frames, tokens):
        assert num_audio_tokens(frames) == tokens

    @pytest.mark.parametrize("frames", [1, 7, 8, 9, 63, 64, 65, 1000])
    def test_tail_frames_round_up(self, frames):
        # Rounding down would drop the tail of every clip whose length is not a multiple of 8, and
        # the placeholders would then outnumber the features.
        assert num_audio_tokens(frames) == -(-frames // SUBSAMPLING_FACTOR)
        assert num_audio_tokens(frames) * SUBSAMPLING_FACTOR >= frames

    @pytest.mark.parametrize("seconds", [0.1, 0.5, 1.0, 2.5, 10.0, 60.0])
    def test_one_token_per_eighty_milliseconds(self, seconds):
        # 8 frames x 10 ms: the figure the tutorial quotes for prompt budgeting.
        assert seconds_to_tokens(seconds) == pytest.approx(seconds / 0.08, abs=1)

    def test_one_minute_of_audio(self):
        assert seconds_to_tokens(60.0) == 751


class TestPlaceholders:
    @pytest.mark.parametrize("clips", [[1.0], [0.5, 1.7], [0.2, 0.2, 0.2]])
    def test_placeholders_match_the_encoder_output(self, clips):
        # The processor writes this many placeholders; the speech state concatenates exactly this
        # many feature rows. A mismatch is what the scatter in the embedding model would trip over.
        per_clip = [seconds_to_tokens(seconds) for seconds in clips]
        assert sum(per_clip) == sum(num_audio_tokens(num_mel_frames(int(s * SAMPLE_RATE))) for s in clips)
        assert all(count > 0 for count in per_clip)

    def test_padding_a_batch_does_not_change_any_clip(self):
        # Clips are staged in one padded tensor, but each is encoded on its own frames, so a clip's
        # token count never depends on what it is batched with.
        short, long = 0.5, 1.7
        assert seconds_to_tokens(short) == num_audio_tokens(num_mel_frames(int(short * SAMPLE_RATE)))
        padded_frames = num_mel_frames(int(long * SAMPLE_RATE))
        assert num_audio_tokens(padded_frames) != seconds_to_tokens(short), "the test lengths must differ"


@pytest.mark.parametrize("seconds", [0.25, 1.0, 2.5])
def test_matches_liquid_audio(seconds):
    """The reference implementation itself, when the liquid-audio package is installed."""
    torch = pytest.importorskip("torch")
    processor_module = pytest.importorskip("liquid_audio.model.conformer.processor")

    preprocessor = processor_module.AudioToMelSpectrogramPreprocessor(
        sample_rate=SAMPLE_RATE,
        normalize="per_feature",
        window_size=WINDOW_SIZE,
        window_stride=WINDOW_STRIDE,
        window="hann",
        features=NUM_MELS,
        n_fft=FFT_SIZE,
        log=True,
        frame_splicing=1,
        dither=1.0e-05,
        pad_to=0,
        pad_value=0.0,
    ).eval()

    num_samples = int(seconds * SAMPLE_RATE)
    with torch.no_grad():
        mel, lengths = preprocessor(torch.zeros(1, num_samples), torch.tensor([num_samples]))

    assert mel.shape[1] == NUM_MELS
    assert mel.shape[2] == num_mel_frames(num_samples)
    assert int(lengths[0]) == num_valid_mel_frames(num_samples)
