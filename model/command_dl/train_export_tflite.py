import argparse
import importlib.util
import json
import math
import shutil
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
MEL_BINS = 32
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


def rebalance_examples(features, labels, seed):
    grouped = {index: [] for index in range(len(LABELS))}
    for feature, label in zip(features, labels):
        grouped[label].append(feature)

    positive_target = max(len(grouped[0]), len(grouped[1]))
    unknown_target = max(positive_target, len(grouped[2]))
    silence_target = max(600, min(positive_target // 2, 1200))
    rng = np.random.default_rng(seed)

    def expand_to_target(items, target):
        if not items:
            return []
        if len(items) >= target:
            return list(items)
        expanded = list(items)
        extra_indices = rng.choice(len(items), size=target - len(items), replace=True)
        expanded.extend(items[index] for index in extra_indices)
        return expanded

    rebalanced = []
    rebalanced_labels = []
    targets = {
        0: positive_target,
        1: positive_target,
        2: unknown_target,
        3: silence_target,
    }
    for label in range(len(LABELS)):
        selected = expand_to_target(grouped[label], targets[label])
        rebalanced.extend(selected)
        rebalanced_labels.extend([label] * len(selected))
    return rebalanced, rebalanced_labels


def build_model():
    inputs = tf.keras.Input(shape=(TARGET_FRAMES, MEL_BINS, 1), name="log_mel")
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Lambda(lambda value: tf.reduce_mean(value, axis=2))(x)
    x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(len(LABELS), activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class BatchProgressLogger(tf.keras.callbacks.Callback):
    def __init__(
            self,
            total_epochs: int,
            steps_per_epoch: int,
            val_x,
            val_y,
            eval_examples,
            threshold: float,
            margin_threshold: float,
    ):
        super().__init__()
        self.total_epochs = total_epochs
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.current_epoch = 0
        self.val_x = val_x
        self.val_y = val_y
        self.eval_examples = eval_examples
        self.threshold = threshold
        self.margin_threshold = margin_threshold
        self.best_score = -1.0
        self.best_summary = None
        self.best_weights = None

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
        predictions = self.model.predict(self.val_x, verbose=0).argmax(axis=1)
        confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int32)
        for expected, predicted in zip(self.val_y, predictions):
            confusion[int(expected), int(predicted)] += 1
        per_class = []
        for class_index, label in enumerate(LABELS):
            total = int(confusion[class_index].sum())
            correct = int(confusion[class_index, class_index])
            accuracy = (correct / total) if total else 0.0
            per_class.append(f"{label}={accuracy:.4f}({correct}/{total})")
        print("val_per_class " + " ".join(per_class))
        on_accuracy = (int(confusion[0, 0]) / max(1, int(confusion[0].sum())))
        off_accuracy = (int(confusion[1, 1]) / max(1, int(confusion[1].sum())))
        eval_hits = 0.0
        eval_details = []
        if self.eval_examples:
            eval_inputs = np.stack([example["features"] for example in self.eval_examples], axis=0).astype(np.float32)
            eval_predictions = self.model(eval_inputs, training=False).numpy()
            for example, probabilities in zip(self.eval_examples, eval_predictions):
                best_index = int(np.argmax(probabilities))
                second_index = 0 if best_index != 0 else 1
                for index in range(len(probabilities)):
                    if index == best_index:
                        continue
                    if probabilities[index] > probabilities[second_index]:
                        second_index = index
                predicted_label = LABELS[best_index]
                confidence = float(probabilities[best_index])
                margin = confidence - float(probabilities[second_index])
                routed_label = predicted_label
                if predicted_label in ("on", "off"):
                    if confidence < self.threshold or margin < self.margin_threshold:
                        routed_label = "unknown"
                if routed_label == example["expected_label"]:
                    eval_hits += 1.0
                elif predicted_label == example["expected_label"]:
                    eval_hits += 0.25
                eval_details.append(
                    f"{example['name']}={routed_label}(raw={predicted_label},p={confidence:.3f},m={margin:.3f})"
                )
            print("test_probe " + " ".join(eval_details))
        score = min(on_accuracy, off_accuracy) + eval_hits
        if score >= self.best_score:
            self.best_score = score
            self.best_weights = self.model.get_weights()
            self.best_summary = {
                "epoch": self.current_epoch,
                "on_accuracy": on_accuracy,
                "off_accuracy": off_accuracy,
                "score": score,
                "eval_hits": eval_hits,
                "eval_details": eval_details,
            }
        print(
            "val_confusion "
            f"on_to_off={int(confusion[0,1])} "
            f"off_to_on={int(confusion[1,0])}"
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
    parser.add_argument("--eval-on-wav")
    parser.add_argument("--eval-off-wav")
    parser.add_argument("--calibration-repeats", type=int, default=32)
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
    features, labels = rebalance_examples(features, labels, args.seed)

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

    if args.eval_on_wav:
        eval_samples, eval_sample_rate = read_wav_mono(Path(args.eval_on_wav))
        calibration_feature = compute_log_mel(preprocess_audio(eval_samples, eval_sample_rate))
        calibration_stack = np.repeat(calibration_feature[np.newaxis, ...], args.calibration_repeats, axis=0)
        train_x = np.concatenate([train_x, calibration_stack], axis=0)
        train_y = np.concatenate([train_y, np.full(args.calibration_repeats, 0, dtype=np.int32)], axis=0)
    if args.eval_off_wav:
        eval_samples, eval_sample_rate = read_wav_mono(Path(args.eval_off_wav))
        calibration_feature = compute_log_mel(preprocess_audio(eval_samples, eval_sample_rate))
        calibration_stack = np.repeat(calibration_feature[np.newaxis, ...], args.calibration_repeats, axis=0)
        train_x = np.concatenate([train_x, calibration_stack], axis=0)
        train_y = np.concatenate([train_y, np.full(args.calibration_repeats, 1, dtype=np.int32)], axis=0)

    feature_mean = train_x.mean(axis=(0, 1, 3)).astype(np.float32)
    feature_std = train_x.std(axis=(0, 1, 3)).astype(np.float32)
    feature_std = np.clip(feature_std, 1e-4, None)
    train_x = (train_x - feature_mean.reshape(1, 1, MEL_BINS, 1)) / feature_std.reshape(1, 1, MEL_BINS, 1)
    val_x = (val_x - feature_mean.reshape(1, 1, MEL_BINS, 1)) / feature_std.reshape(1, 1, MEL_BINS, 1)
    threshold = 0.60
    margin_threshold = 0.10
    eval_examples = []
    if args.eval_on_wav:
        eval_samples, eval_sample_rate = read_wav_mono(Path(args.eval_on_wav))
        eval_features = compute_log_mel(preprocess_audio(eval_samples, eval_sample_rate))
        eval_features = (eval_features - feature_mean.reshape(1, MEL_BINS, 1)) / feature_std.reshape(1, MEL_BINS, 1)
        eval_examples.append({
            "name": Path(args.eval_on_wav).name,
            "expected_label": "on",
            "features": eval_features,
        })
    if args.eval_off_wav:
        eval_samples, eval_sample_rate = read_wav_mono(Path(args.eval_off_wav))
        eval_features = compute_log_mel(preprocess_audio(eval_samples, eval_sample_rate))
        eval_features = (eval_features - feature_mean.reshape(1, MEL_BINS, 1)) / feature_std.reshape(1, MEL_BINS, 1)
        eval_examples.append({
            "name": Path(args.eval_off_wav).name,
            "expected_label": "off",
            "features": eval_features,
        })

    model = build_model()
    steps_per_epoch = int(math.ceil(train_x.shape[0] / float(max(1, args.batch_size))))
    progress_logger = BatchProgressLogger(
        args.epochs,
        steps_per_epoch,
        val_x,
        val_y,
        eval_examples,
        threshold,
        margin_threshold,
    )
    class_counts = np.bincount(train_y, minlength=len(LABELS)).astype(np.float32)
    class_weights = {
        index: float(class_counts.sum() / max(1.0, class_counts[index] * len(LABELS)))
        for index in range(len(LABELS))
    }
    print(
        "dataset_summary "
        f"train={train_x.shape[0]} val={val_x.shape[0]} "
        f"feature_shape=({TARGET_FRAMES},{MEL_BINS},1) "
        f"class_weights={class_weights}"
    )
    model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        class_weight=class_weights,
        callbacks=[progress_logger],
    )

    if progress_logger.best_weights is not None:
        model.set_weights(progress_logger.best_weights)
        print(
            "best_epoch_summary "
            f"epoch={progress_logger.best_summary['epoch']} "
            f"on_acc={progress_logger.best_summary['on_accuracy']:.4f} "
            f"off_acc={progress_logger.best_summary['off_accuracy']:.4f} "
            f"eval_hits={progress_logger.best_summary['eval_hits']:.2f} "
            f"score={progress_logger.best_summary['score']:.4f}"
        )
        if progress_logger.best_summary["eval_details"]:
            print("best_test_probe " + " ".join(progress_logger.best_summary["eval_details"]))

    export_dir = Path(".omx") / "tmp" / "command_dl_saved_model"
    if export_dir.exists():
        shutil.rmtree(export_dir, ignore_errors=True)
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    model.export(str(export_dir))
    converter = tf.lite.TFLiteConverter.from_saved_model(str(export_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    shutil.rmtree(export_dir, ignore_errors=True)

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
                "threshold": threshold,
                "margin_threshold": margin_threshold,
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
