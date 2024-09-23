# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Compute input examples for VGGish from audio waveform."""

import numpy as np
import resampy

import dataset.mel_features as mel_features

# Hyperparameters used in feature and example generation.
SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = 64 # 480
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 4.8  # Each example contains 10,000 10ms frames
EXAMPLE_HOP_SECONDS = 4.8  # with zero overlap.

try:
    import soundfile as sf

    def wav_read(wav_file):
        wav_data, sr = sf.read(wav_file, dtype="int16")
        return wav_data, sr

except ImportError:

    def wav_read(wav_file):
        raise NotImplementedError("WAV file reading requires soundfile package.")


def waveform_to_examples(data, sample_rate, num_mel_bins):
    """Converts audio waveform into an array of examples for VGGish.

    Args:
      data: np.array of either one dimension (mono) or two dimensions
        (multi-channel, with the outer dimension representing channels).
        Each sample is generally expected to lie in the range [-1.0, +1.0],
        although this is not required.
      sample_rate: Sample rate of data.

    Returns:
      3-D np.array of shape [num_examples, num_frames, num_bands] which represents
      a sequence of examples, each of which contains a patch of log mel
      spectrogram, covering num_frames frames of audio and num_bands mel frequency
      bands, where the frame length is STFT_HOP_LENGTH_SECONDS.
    """
    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, SAMPLE_RATE)

    print("data.shape after resampling", data.shape)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=SAMPLE_RATE,
        log_offset=LOG_OFFSET,
        window_length_secs=STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=num_mel_bins,
        lower_edge_hertz=MEL_MIN_HZ,
        upper_edge_hertz=MEL_MAX_HZ,
    )

    print("log_mel.shape", log_mel.shape)

    # Frame features into examples.
    features_sample_rate = 1.0 / STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = mel_features.frame(
        log_mel, window_length=example_window_length, hop_length=example_hop_length
    )

    print("log_mel_examples.shape", log_mel_examples.shape)
    
    return log_mel_examples


def wavfile_to_examples(wav_file, num_mel_bins):
    """Convenience wrapper around waveform_to_examples() for a common WAV format.

    Args:
      wav_file: String path to a file, or a file-like object. The file
      is assumed to contain WAV audio data with signed 16-bit PCM samples.

    Returns:
      See waveform_to_examples.
    """
    wav_data, sr = wav_read(wav_file)
    assert wav_data.dtype == np.int16, "Bad sample type: %r" % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    return waveform_to_examples(samples, sr, num_mel_bins)
