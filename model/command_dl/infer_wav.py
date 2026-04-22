import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np


def read_wav_mono(path: Path):
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV is supported")

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    return samples, sample_rate


def resample_linear(samples, input_rate, output_rate):
    if input_rate == output_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.shape[0] / float(input_rate)
    output_length = max(1, int(round(duration * output_rate)))
    source_positions = np.linspace(0.0, samples.shape[0] - 1, output_length, dtype=np.float32)
    left = np.floor(source_positions).astype(np.int32)
    right = np.minimum(left + 1, samples.shape[0] - 1)
    alpha = source_positions - left
    return (samples[left] * (1.0 - alpha) + samples[right] * alpha).astype(np.float32)


def trim_and_window(samples, sample_rate, window_samples):
    samples = resample_linear(samples, sample_rate, 16000)
    peak = np.max(np.abs(samples)) if samples.size else 0.0
    if peak > 1e-5:
        samples = samples / peak
    if samples.shape[0] >= window_samples:
        center = samples.shape[0] // 2
        half = window_samples // 2
        start = max(0, center - half)
        end = min(samples.shape[0], start + window_samples)
        start = end - window_samples
        return samples[start:end]
    output = np.zeros(window_samples, dtype=np.float32)
    offset = (window_samples - samples.shape[0]) // 2
    output[offset:offset + samples.shape[0]] = samples
    return output


def hz_to_mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def build_mel_bank(fft_size, mel_bins):
    min_mel = hz_to_mel(20.0)
    max_mel = hz_to_mel(8000.0)
    mel_points = np.linspace(min_mel, max_mel, mel_bins + 2, dtype=np.float32)
    hz_points = np.array([mel_to_hz(value) for value in mel_points], dtype=np.float32)
    bins = np.floor((fft_size // 2 + 1) * hz_points / 8000.0).astype(np.int32)
    bins = np.clip(bins, 0, fft_size // 2)
    bank = np.zeros((mel_bins, fft_size // 2 + 1), dtype=np.float32)
    for mel_index in range(mel_bins):
        left = bins[mel_index]
        center = max(left + 1, bins[mel_index + 1])
        right = max(center + 1, bins[mel_index + 2])
        for bin_index in range(left, min(center, bank.shape[1])):
            bank[mel_index, bin_index] = (bin_index - left) / max(1, center - left)
        for bin_index in range(center, min(right, bank.shape[1])):
            bank[mel_index, bin_index] = (right - bin_index) / max(1, right - center)
    return bank


def log_mel(samples, frame_size, frame_hop, fft_size, mel_bins, target_frames):
    window = np.hanning(frame_size).astype(np.float32)
    frame_count = 1 + max(0, (samples.shape[0] - frame_size) // frame_hop)
    bank = build_mel_bank(fft_size, mel_bins)
    frames = []
    for index in range(frame_count):
        start = index * frame_hop
        frame = np.zeros(fft_size, dtype=np.float32)
        frame[:frame_size] = samples[start:start + frame_size] * window
        spectrum = np.fft.rfft(frame)
        power = np.abs(spectrum) ** 2
        mel = bank @ power
        frames.append(np.log(mel + 1e-5))
    features = np.stack(frames, axis=0).astype(np.float32)
    positions = np.linspace(0, features.shape[0] - 1, target_frames, dtype=np.float32)
    left = np.floor(positions).astype(np.int32)
    right = np.minimum(left + 1, features.shape[0] - 1)
    alpha = (positions - left).reshape(-1, 1)
    return features[left] * (1.0 - alpha) + features[right] * alpha


def relu(values):
    return np.maximum(values, 0.0)


def softmax(values):
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def conv2d_single_channel(inputs, weights, bias, padding_h, padding_w):
    time_steps, mel_bins = inputs.shape
    out_channels = len(weights)
    kernel_h = len(weights[0][0])
    kernel_w = len(weights[0][0][0])
    outputs = np.zeros((out_channels, time_steps, mel_bins), dtype=np.float32)
    for out_channel in range(out_channels):
        for time_index in range(time_steps):
            for mel_index in range(mel_bins):
                acc = bias[out_channel]
                for kernel_time in range(kernel_h):
                    for kernel_mel in range(kernel_w):
                        source_time = time_index + kernel_time - padding_h
                        source_mel = mel_index + kernel_mel - padding_w
                        if 0 <= source_time < time_steps and 0 <= source_mel < mel_bins:
                            acc += weights[out_channel][0][kernel_time][kernel_mel] * inputs[source_time][source_mel]
                outputs[out_channel][time_index][mel_index] = acc
    return outputs


def conv1d(inputs, weights, bias, padding):
    in_channels, time_steps = inputs.shape
    out_channels = len(weights)
    kernel_size = len(weights[0][0])
    outputs = np.zeros((out_channels, time_steps), dtype=np.float32)
    for out_channel in range(out_channels):
        for time_index in range(time_steps):
            acc = bias[out_channel]
            for in_channel in range(in_channels):
                for kernel_index in range(kernel_size):
                    source_time = time_index + kernel_index - padding
                    if 0 <= source_time < time_steps:
                        acc += weights[out_channel][in_channel][kernel_index] * inputs[in_channel][source_time]
            outputs[out_channel][time_index] = acc
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-json", required=True)
    parser.add_argument("--wav", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.model_json).read_text(encoding="utf-8"))
    samples, sample_rate = read_wav_mono(Path(args.wav))
    samples = trim_and_window(samples, sample_rate, payload["window_samples"])
    features = log_mel(
        samples,
        payload["frame_size"],
        payload["frame_hop"],
        payload["fft_size"],
        payload["mel_bins"],
        payload["target_frames"],
    )
    mean = np.array(payload["feature_mean"], dtype=np.float32)
    std = np.array(payload["feature_std"], dtype=np.float32)
    features = (features - mean.reshape(1, -1)) / std.reshape(1, -1)

    conv = payload["conv2d"]
    temporal = payload["temporal_conv"]
    fc = payload["fc"]

    hidden = relu(
        conv2d_single_channel(
            features,
            conv["weights"],
            conv["bias"],
            conv["padding_h"],
            conv["padding_w"],
        )
    )
    hidden = hidden.mean(axis=2)
    hidden = relu(conv1d(hidden, temporal["weights"], temporal["bias"], temporal["padding"]))
    pooled = hidden.mean(axis=1)
    logits = np.dot(np.array(fc["weights"], dtype=np.float32), pooled) + np.array(fc["bias"], dtype=np.float32)
    probabilities = softmax(logits)
    best_index = int(np.argmax(probabilities))
    print(json.dumps({
        "label": payload["labels"][best_index],
        "confidence": float(probabilities[best_index]),
        "probabilities": {
            label: float(probabilities[index]) for index, label in enumerate(payload["labels"])
        },
    }, indent=2))


if __name__ == "__main__":
    main()
