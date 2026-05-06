import argparse
import importlib.util
import json
import time
import wave
from pathlib import Path

import numpy as np

if not importlib.util.find_spec("tensorflow"):
    raise SystemExit(
        "tensorflow is required for TFLite inference. Install TensorFlow in your Python environment first."
    )

import tensorflow as tf


def calibrate_probabilities(probabilities, meta):
    calibrated = np.array(probabilities, dtype=np.float32, copy=True)
    labels = meta["labels"]
    on_score_scale = float(meta.get("on_score_scale", 1.08))
    off_score_scale = float(meta.get("off_score_scale", 0.94))
    for index, label in enumerate(labels):
        if label == "on":
            calibrated[index] *= on_score_scale
        elif label == "off":
            calibrated[index] *= off_score_scale
    total = float(np.sum(calibrated))
    if total > 1e-6:
        calibrated /= total
    return calibrated


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
    elif channels != 1:
        raise ValueError("Only mono or stereo WAV is supported")
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


def normalize_amplitude(samples):
    peak = np.max(np.abs(samples)) if samples.size else 0.0
    if peak < 1e-5:
        return samples.astype(np.float32, copy=False)
    return (samples / peak).astype(np.float32)


def trim_silence(samples, frame_size, frame_hop, window_samples):
    if samples.size <= frame_size:
        return samples
    frame_count = 1 + max(0, (samples.shape[0] - frame_size) // frame_hop)
    rms = []
    for index in range(frame_count):
        start = index * frame_hop
        frame = samples[start:start + frame_size]
        rms.append(float(np.sqrt(np.mean(frame * frame) + 1e-8)))
    rms = np.array(rms, dtype=np.float32)
    threshold = max(float(rms.max()) * 0.12, 0.01)
    active = np.where(rms >= threshold)[0]
    if active.size == 0:
        return samples[: min(samples.shape[0], window_samples)]
    start_frame = max(0, int(active[0]) - 2)
    end_frame = min(frame_count - 1, int(active[-1]) + 2)
    start = start_frame * frame_hop
    end = min(samples.shape[0], end_frame * frame_hop + frame_size)
    return samples[start:end]


def fit_to_window(samples, window_samples):
    if samples.shape[0] >= window_samples:
        center = samples.shape[0] // 2
        half = window_samples // 2
        start = max(0, center - half)
        end = min(samples.shape[0], start + window_samples)
        start = end - window_samples
        return samples[start:end].astype(np.float32)
    output = np.zeros(window_samples, dtype=np.float32)
    offset = (window_samples - samples.shape[0]) // 2
    output[offset:offset + samples.shape[0]] = samples
    return output


def preprocess_resampled(samples, meta):
    samples = normalize_amplitude(samples)
    samples = trim_silence(samples, meta["frame_size"], meta["frame_hop"], meta["window_samples"])
    samples = fit_to_window(samples, meta["window_samples"])
    return samples


def preprocess(samples, sample_rate, meta):
    samples = resample_linear(samples, sample_rate, meta["sample_rate"])
    return preprocess_resampled(samples, meta)


def compute_spectrograms(samples, meta):
    stft = tf.signal.stft(
        tf.convert_to_tensor(samples, dtype=tf.float32),
        frame_length=meta["frame_size"],
        frame_step=meta["frame_hop"],
        fft_length=meta["fft_size"],
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    power = (tf.abs(stft) ** 2).numpy().astype(np.float32)
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=meta["mel_bins"],
        num_spectrogram_bins=(meta["fft_size"] // 2) + 1,
        sample_rate=meta["sample_rate"],
        lower_edge_hertz=20.0,
        upper_edge_hertz=meta["sample_rate"] / 2.0,
    )
    mel = tf.matmul(tf.convert_to_tensor(power, dtype=tf.float32), mel_matrix)
    log_mel = tf.math.log(mel + 1e-5)
    log_mel = tf.image.resize(
        log_mel[..., tf.newaxis],
        [meta["target_frames"], meta["mel_bins"]],
    ).numpy().astype(np.float32)

    mean = np.array(meta["feature_mean"], dtype=np.float32).reshape(1, meta["mel_bins"], 1)
    std = np.array(meta["feature_std"], dtype=np.float32).reshape(1, meta["mel_bins"], 1)
    normalized_log_mel = (log_mel - mean) / std
    return power, log_mel, normalized_log_mel


def render_visualizations(
    raw_samples,
    raw_sample_rate,
    processed_samples,
    processed_sample_rate,
    power_spectrogram,
    log_mel,
    wav_path,
    output_path=None,
    show=False,
):
    if not importlib.util.find_spec("matplotlib"):
        raise SystemExit(
            "matplotlib is required for visualization. Install matplotlib or run without visualization flags."
        )

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 1, figsize=(14, 13))

    raw_waveform_time = np.arange(raw_samples.shape[0], dtype=np.float32) / float(raw_sample_rate)
    axes[0].plot(raw_waveform_time, raw_samples, color="#0f766e", linewidth=0.8)
    axes[0].set_title("Input WAV Waveform (Before Preprocessing)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0.0, raw_waveform_time[-1] if raw_waveform_time.size else 0.0)
    axes[0].grid(True, alpha=0.25)

    processed_waveform_time = np.arange(processed_samples.shape[0], dtype=np.float32) / float(processed_sample_rate)
    axes[1].plot(processed_waveform_time, processed_samples, color="#2563eb", linewidth=0.8)
    axes[1].set_title("Preprocessed WAV Waveform (After Preprocessing)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlim(0.0, processed_waveform_time[-1] if processed_waveform_time.size else 0.0)
    axes[1].grid(True, alpha=0.25)

    stft_image = axes[2].imshow(
        np.log10(np.maximum(power_spectrogram, 1e-8)).T,
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    axes[2].set_title("STFT Log-Power Spectrogram")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Frequency Bin")
    figure.colorbar(stft_image, ax=axes[2], fraction=0.046, pad=0.04)

    log_mel_image = axes[3].imshow(
        np.squeeze(log_mel, axis=-1).T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[3].set_title("Log-Mel Spectrogram")
    axes[3].set_xlabel("Frame")
    axes[3].set_ylabel("Mel Bin")
    figure.colorbar(log_mel_image, ax=axes[3], fraction=0.046, pad=0.04)

    figure.suptitle(f"Audio Features: {Path(wav_path).name}")
    figure.tight_layout()

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tflite", required=True)
    parser.add_argument("--model-meta", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--show-visualizations", action="store_true")
    parser.add_argument("--save-visualizations")
    args = parser.parse_args()

    meta = json.loads(Path(args.model_meta).read_text(encoding="utf-8"))

    preprocess_started_at = time.perf_counter()
    raw_samples, raw_sample_rate = read_wav_mono(Path(args.wav))
    samples = preprocess(raw_samples, raw_sample_rate, meta)
    power_spectrogram, log_mel, features = compute_spectrograms(samples, meta)
    inputs = features[np.newaxis, ...]
    preprocess_ms = (time.perf_counter() - preprocess_started_at) * 1000.0

    if args.show_visualizations or args.save_visualizations:
        render_visualizations(
            raw_samples,
            raw_sample_rate,
            samples,
            meta["sample_rate"],
            power_spectrogram,
            log_mel,
            args.wav,
            output_path=args.save_visualizations,
            show=args.show_visualizations,
        )

    interpreter = tf.lite.Interpreter(model_path=args.model_tflite)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], inputs)
    inference_started_at = time.perf_counter()
    interpreter.invoke()
    inference_ms = (time.perf_counter() - inference_started_at) * 1000.0
    probabilities = calibrate_probabilities(
        interpreter.get_tensor(output_details[0]["index"])[0],
        meta,
    )
    total_ms = preprocess_ms + inference_ms

    best_index = int(np.argmax(probabilities))
    second_index = 0 if best_index != 0 else 1
    for index in range(len(probabilities)):
        if index == best_index:
            continue
        if probabilities[index] > probabilities[second_index]:
            second_index = index
    label = meta["labels"][best_index]
    confidence = float(probabilities[best_index])
    second_label = meta["labels"][second_index]
    second_confidence = float(probabilities[second_index])
    margin = confidence - second_confidence
    threshold = float(meta.get("on_threshold" if label == "on" else "off_threshold", meta.get("threshold", 0.75)))
    margin_threshold = float(
        meta.get(
            "on_margin_threshold" if label == "on" else "off_margin_threshold",
            meta.get("margin_threshold", 0.12),
        )
    )
    routed_label = label
    if label in ("on", "off"):
        if confidence < threshold or margin < margin_threshold:
            routed_label = "unknown"
    print(
        json.dumps(
            {
                "label": label,
                "confidence": confidence,
                "second_label": second_label,
                "second_confidence": second_confidence,
                "margin": margin,
                "routed_label": routed_label,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "total_ms": total_ms,
                "probabilities": {
                    class_label: float(probabilities[index])
                    for index, class_label in enumerate(meta["labels"])
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
