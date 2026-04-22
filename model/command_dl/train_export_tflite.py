import argparse
import importlib.util
import json
import math
import random
import wave
from pathlib import Path

import numpy as np

if not importlib.util.find_spec("tensorflow"):
    raise SystemExit(
        "tensorflow is required for TFLite export. Install TensorFlow in your Python environment first."
    )

import tensorflow as tf

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 16_000
FRAME_SIZE = 400
FRAME_HOP = 160
FFT_SIZE = 512
MEL_BINS = 24
TARGET_FRAMES = 64
LABELS = ["on", "off", "unknown", "silence"]


def read_wav_mono(path: Path):
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"{path} must be 16-bit PCM WAV")
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    elif channels != 1:
        raise ValueError(f"{path} must be mono or stereo WAV")
    return samples, sample_rate


def resample_linear(samples: np.ndarray, input_rate: int, output_rate: int):
    if input_rate == output_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.shape[0] / float(input_rate)
    output_length = max(1, int(round(duration * output_rate)))
    source_positions = np.linspace(0.0, samples.shape[0] - 1, output_length, dtype=np.float32)
    left = np.floor(source_positions).astype(np.int32)
    right = np.minimum(left + 1, samples.shape[0] - 1)
    alpha = source_positions - left
    return (samples[left] * (1.0 - alpha) + samples[right] * alpha).astype(np.float32)


def normalize_amplitude(samples: np.ndarray):
    peak = np.max(np.abs(samples)) if samples.size else 0.0
    if peak < 1e-5:
        return samples.astype(np.float32, copy=False)
    return (samples / peak).astype(np.float32)


def trim_silence(samples: np.ndarray):
    if samples.size <= FRAME_SIZE:
        return samples
    frame_count = 1 + max(0, (samples.shape[0] - FRAME_SIZE) // FRAME_HOP)
    rms = []
    for index in range(frame_count):
        start = index * FRAME_HOP
        frame = samples[start:start + FRAME_SIZE]
        rms.append(float(np.sqrt(np.mean(frame * frame) + 1e-8)))
    rms = np.array(rms, dtype=np.float32)
    threshold = max(float(rms.max()) * 0.12, 0.01)
    active = np.where(rms >= threshold)[0]
    if active.size == 0:
        return samples[: min(samples.shape[0], WINDOW_SAMPLES)]
    start_frame = max(0, int(active[0]) - 2)
    end_frame = min(frame_count - 1, int(active[-1]) + 2)
    start = start_frame * FRAME_HOP
    end = min(samples.shape[0], end_frame * FRAME_HOP + FRAME_SIZE)
    return samples[start:end]


def fit_to_window(samples: np.ndarray):
    if samples.shape[0] >= WINDOW_SAMPLES:
        center = samples.shape[0] // 2
        half = WINDOW_SAMPLES // 2
        start = max(0, center - half)
        end = start + WINDOW_SAMPLES
        if end > samples.shape[0]:
            end = samples.shape[0]
            start = end - WINDOW_SAMPLES
        return samples[start:end].astype(np.float32)
    output = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    offset = (WINDOW_SAMPLES - samples.shape[0]) // 2
    output[offset:offset + samples.shape[0]] = samples
    return output


def preprocess_audio(samples: np.ndarray, sample_rate: int):
    return fit_to_window(trim_silence(normalize_amplitude(resample_linear(samples, sample_rate, SAMPLE_RATE))))


def synthesize_silence(seed: int):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.003, WINDOW_SAMPLES).astype(np.float32)
    if seed % 3 == 0:
        pulse_length = rng.integers(300, 900)
        start = rng.integers(0, WINDOW_SAMPLES - pulse_length)
        base[start:start + pulse_length] += rng.normal(0.0, 0.01, pulse_length).astype(np.float32)
    return np.clip(base, -1.0, 1.0)


def compute_log_mel(samples: np.ndarray):
    tensor = tf.convert_to_tensor(samples, dtype=tf.float32)
    stft = tf.signal.stft(
        tensor,
        frame_length=FRAME_SIZE,
        frame_step=FRAME_HOP,
        fft_length=FFT_SIZE,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    power = tf.abs(stft) ** 2
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=MEL_BINS,
        num_spectrogram_bins=(FFT_SIZE // 2) + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=20.0,
        upper_edge_hertz=SAMPLE_RATE / 2.0,
    )
    mel = tf.matmul(power, mel_matrix)
    log_mel = tf.math.log(mel + 1e-5)
    log_mel = tf.image.resize(log_mel[..., tf.newaxis], [TARGET_FRAMES, MEL_BINS]).numpy()
    return log_mel.astype(np.float32)


def collect_wavs(directory: Path):
    return sorted(path for path in directory.glob("*.wav"))


def build_features(paths, label_index):
    features = []
    labels = []
    for path in paths:
        samples, sample_rate = read_wav_mono(path)
        features.append(compute_log_mel(preprocess_audio(samples, sample_rate)))
        labels.append(label_index)
    return features, labels


def build_model():
    inputs = tf.keras.Input(shape=(TARGET_FRAMES, MEL_BINS, 1), name="log_mel")
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Lambda(lambda value: tf.reduce_mean(value, axis=2))(x)
    x = tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(len(LABELS), activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class BatchProgressLogger(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs: int, steps_per_epoch: int):
        super().__init__()
        self.total_epochs = total_epochs
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        print(f"\nEpoch {self.current_epoch}/{self.total_epochs}")

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        step = batch + 1
        progress = step / float(self.steps_per_epoch)
        bar_width = 24
        filled = min(bar_width, int(round(progress * bar_width)))
        bar = "#" * filled + "-" * (bar_width - filled)
        loss = float(logs.get("loss", 0.0))
        accuracy = float(logs.get("accuracy", 0.0))
        print(
            f"\r[{bar}] {step:>4}/{self.steps_per_epoch:<4} "
            f"loss={loss:.4f} acc={accuracy:.4f}",
            end="",
            flush=True,
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print()
        print(
            "epoch_summary "
            f"loss={float(logs.get('loss', 0.0)):.4f} "
            f"acc={float(logs.get('accuracy', 0.0)):.4f} "
            f"val_loss={float(logs.get('val_loss', 0.0)):.4f} "
            f"val_acc={float(logs.get('val_accuracy', 0.0)):.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--on-dir", required=True)
    parser.add_argument("--off-dir", required=True)
    parser.add_argument("--unknown-dir", action="append", required=True)
    parser.add_argument("--output-tflite", required=True)
    parser.add_argument("--output-meta", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    features = []
    labels = []

    on_features, on_labels = build_features(collect_wavs(Path(args.on_dir)), 0)
    off_features, off_labels = build_features(collect_wavs(Path(args.off_dir)), 1)
    unknown_paths = []
    for directory in args.unknown_dir:
        unknown_paths.extend(collect_wavs(Path(directory)))
    unknown_features, unknown_labels = build_features(sorted(unknown_paths), 2)

    silence_features = []
    silence_labels = []
    silence_count = max(1200, (len(on_features) + len(off_features)) // 2)
    for index in range(silence_count):
        silence_features.append(compute_log_mel(synthesize_silence(args.seed + index)))
        silence_labels.append(3)

    features.extend(on_features)
    labels.extend(on_labels)
    features.extend(off_features)
    labels.extend(off_labels)
    features.extend(unknown_features)
    labels.extend(unknown_labels)
    features.extend(silence_features)
    labels.extend(silence_labels)

    features = np.stack(features, axis=0)
    labels = np.array(labels, dtype=np.int32)
    permutation = np.random.permutation(features.shape[0])
    split_index = max(1, int(features.shape[0] * 0.85))
    train_indices = permutation[:split_index]
    val_indices = permutation[split_index:]

    train_x = features[train_indices]
    train_y = labels[train_indices]
    val_x = features[val_indices]
    val_y = labels[val_indices]

    feature_mean = train_x.mean(axis=(0, 1, 3)).astype(np.float32)
    feature_std = train_x.std(axis=(0, 1, 3)).astype(np.float32)
    feature_std = np.clip(feature_std, 1e-4, None)
    train_x = (train_x - feature_mean.reshape(1, 1, MEL_BINS, 1)) / feature_std.reshape(1, 1, MEL_BINS, 1)
    val_x = (val_x - feature_mean.reshape(1, 1, MEL_BINS, 1)) / feature_std.reshape(1, 1, MEL_BINS, 1)

    model = build_model()
    steps_per_epoch = int(math.ceil(train_x.shape[0] / float(max(1, args.batch_size))))
    print(
        "dataset_summary "
        f"train={train_x.shape[0]} val={val_x.shape[0]} "
        f"feature_shape=({TARGET_FRAMES},{MEL_BINS},1)"
    )
    model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=[BatchProgressLogger(args.epochs, steps_per_epoch)],
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    output_tflite = Path(args.output_tflite)
    output_meta = Path(args.output_meta)
    output_tflite.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    output_tflite.write_bytes(tflite_model)
    output_meta.write_text(
        json.dumps(
            {
                "version": 1,
                "sample_rate": SAMPLE_RATE,
                "window_samples": WINDOW_SAMPLES,
                "frame_size": FRAME_SIZE,
                "frame_hop": FRAME_HOP,
                "fft_size": FFT_SIZE,
                "mel_bins": MEL_BINS,
                "target_frames": TARGET_FRAMES,
                "stream_hop_samples": 4000,
                "labels": LABELS,
                "threshold": 0.80,
                "smoothing_windows": 3,
                "cooldown_ms": 1200,
                "feature_mean": feature_mean.tolist(),
                "feature_std": feature_std.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"exported TFLite model to {output_tflite}")
    print(f"exported metadata to {output_meta}")


if __name__ == "__main__":
    main()
