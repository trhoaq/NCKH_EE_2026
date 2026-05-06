import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from infer_wav import (
    calibrate_probabilities,
    compute_spectrograms,
    preprocess,
    read_wav_mono,
    tf,
)


DEFAULT_SEED = 2026
DEFAULT_SILENCE_COUNT = 600
COMMAND_LABELS = ("on", "off")
NON_COMMAND_LABELS = ("unknown", "silence")


def synthesize_silence(seed: int, window_samples: int):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.003, window_samples).astype(np.float32)
    if seed % 3 == 0 and window_samples > 900:
        pulse_length = int(rng.integers(300, 900))
        start = int(rng.integers(0, window_samples - pulse_length))
        base[start:start + pulse_length] += rng.normal(0.0, 0.01, pulse_length).astype(np.float32)
    return np.clip(base, -1.0, 1.0)


def find_latest_model_pair(tmp_dir: Path):
    pairs = []
    for tflite_path in tmp_dir.glob("*.tflite"):
        meta_path = tflite_path.with_suffix(".json")
        if meta_path.exists():
            timestamp = max(tflite_path.stat().st_mtime, meta_path.stat().st_mtime)
            pairs.append((timestamp, tflite_path, meta_path))
    if not pairs:
        raise FileNotFoundError(f"No matching .tflite/.json model pair found in {tmp_dir}")
    pairs.sort(key=lambda item: item[0], reverse=True)
    _, model_tflite, model_meta = pairs[0]
    return model_tflite, model_meta


def build_example_list(on_dir: Path, off_dir: Path, unknown_dirs, silence_count: int, seed: int):
    examples = []
    for wav_path in sorted(on_dir.glob("*.wav")):
        examples.append({"path": wav_path, "expected_label": "on", "kind": "wav"})
    for wav_path in sorted(off_dir.glob("*.wav")):
        examples.append({"path": wav_path, "expected_label": "off", "kind": "wav"})
    for unknown_dir in unknown_dirs:
        for wav_path in sorted(unknown_dir.glob("*.wav")):
            examples.append({"path": wav_path, "expected_label": "unknown", "kind": "wav"})
    for index in range(max(0, silence_count)):
        examples.append({"path": None, "expected_label": "silence", "kind": "synthetic_silence", "seed": seed + index})
    return examples


def prepare_input(example, meta):
    if example["kind"] == "wav":
        samples, sample_rate = read_wav_mono(example["path"])
        processed = preprocess(samples, sample_rate, meta)
    else:
        processed = synthesize_silence(example["seed"], int(meta["window_samples"]))
    _, _, features = compute_spectrograms(processed, meta)
    return features[np.newaxis, ...]


def run_inference(interpreter, input_details, output_details, inputs, meta):
    interpreter.set_tensor(input_details[0]["index"], inputs.astype(np.float32))
    interpreter.invoke()
    raw_probabilities = interpreter.get_tensor(output_details[0]["index"])[0]
    probabilities = calibrate_probabilities(raw_probabilities, meta)
    best_index = int(np.argmax(probabilities))
    second_index = 0 if best_index != 0 else 1
    for index in range(len(probabilities)):
        if index == best_index:
            continue
        if probabilities[index] > probabilities[second_index]:
            second_index = index
    return {
        "label": meta["labels"][best_index],
        "confidence": float(probabilities[best_index]),
        "second_label": meta["labels"][second_index],
        "second_confidence": float(probabilities[second_index]),
        "margin": float(probabilities[best_index] - probabilities[second_index]),
        "probabilities": {
            class_label: float(probabilities[index])
            for index, class_label in enumerate(meta["labels"])
        },
    }


def route_label(result, meta):
    label = result["label"]
    routed_label = label
    if label in COMMAND_LABELS:
        threshold = float(meta.get(f"{label}_threshold", meta.get("threshold", 0.75)))
        margin_threshold = float(
            meta.get(
                f"{label}_margin_threshold",
                meta.get("margin_threshold", 0.12),
            )
        )
        unknown_probability = float(result["probabilities"].get("unknown", 0.0))
        silence_probability = float(result["probabilities"].get("silence", 0.0))
        max_noise_probability = max(unknown_probability, silence_probability)
        gate_margin = float(meta.get("command_gate_margin", 0.0))
        gate_noise_limit = float(meta.get("command_gate_max_noise_probability", 1.0))
        if (
            result["confidence"] < threshold
            or result["margin"] < margin_threshold
            or (result["confidence"] - max_noise_probability) < gate_margin
            or max_noise_probability > gate_noise_limit
        ):
            routed_label = "unknown"
    return routed_label


def safe_divide(numerator: float, denominator: float):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(records, labels):
    label_to_index = {label: index for index, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int32)
    for record in records:
        expected_index = label_to_index[record["expected_label"]]
        predicted_index = label_to_index[record["predicted_label"]]
        confusion[expected_index, predicted_index] += 1

    per_label = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    total_correct = int(np.trace(confusion))
    total_examples = int(confusion.sum())
    for label in labels:
        index = label_to_index[label]
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum()) - tp
        fn = int(confusion[index, :].sum()) - tp
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)
        support = int(confusion[index, :].sum())
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    label_count = max(1, len(labels))
    command_total = sum(1 for record in records if record["expected_label"] in COMMAND_LABELS)
    non_command_total = sum(1 for record in records if record["expected_label"] in NON_COMMAND_LABELS)
    false_triggers = sum(
        1
        for record in records
        if record["expected_label"] in NON_COMMAND_LABELS and record["predicted_label"] in COMMAND_LABELS
    )
    missed_commands = sum(
        1
        for record in records
        if record["expected_label"] in COMMAND_LABELS and record["predicted_label"] != record["expected_label"]
    )

    return {
        "accuracy": safe_divide(total_correct, total_examples),
        "precision_macro": macro_precision / label_count,
        "recall_macro": macro_recall / label_count,
        "f1_macro": macro_f1 / label_count,
        "false_trigger_rate": safe_divide(false_triggers, non_command_total),
        "missed_command_rate": safe_divide(missed_commands, command_total),
        "counts": {
            "total_examples": total_examples,
            "command_examples": command_total,
            "non_command_examples": non_command_total,
            "false_triggers": false_triggers,
            "missed_commands": missed_commands,
        },
        "per_label": per_label,
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion.tolist(),
        },
    }


def render_confusion_matrix(metrics, output_path=None, show=False):
    if not importlib.util.find_spec("matplotlib"):
        raise SystemExit(
            "matplotlib is required for confusion matrix visualization. "
            "Install matplotlib or run without confusion matrix visualization flags."
        )

    import matplotlib.pyplot as plt

    labels = metrics["confusion_matrix"]["labels"]
    confusion = np.array(metrics["confusion_matrix"]["matrix"], dtype=np.float32)
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion), where=row_sums > 0)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, matrix, title, cmap in (
        (axes[0], confusion, "Confusion Matrix (Count)", "Blues"),
        (axes[1], normalized, "Confusion Matrix (Row-Normalized)", "Greens"),
    ):
        image = axis.imshow(matrix, cmap=cmap, aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Expected")
        axis.set_xticks(range(len(labels)), labels=labels, rotation=20)
        axis.set_yticks(range(len(labels)), labels=labels)
        for row_index in range(len(labels)):
            for column_index in range(len(labels)):
                value = matrix[row_index, column_index]
                text = f"{int(value)}" if title.endswith("(Count)") else f"{value:.2f}"
                axis.text(column_index, row_index, text, ha="center", va="center", color="black")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle("Model Evaluation Confusion Matrix")
    figure.tight_layout()

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(figure)


def format_example_name(example):
    if example["kind"] == "wav":
        return str(example["path"])
    return f"synthetic_silence:{example['seed']}"


def evaluate_dataset(model_tflite: Path, model_meta: Path, examples, max_examples_to_show: int):
    meta = json.loads(model_meta.read_text(encoding="utf-8"))
    interpreter = tf.lite.Interpreter(model_path=str(model_tflite))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    records = []
    false_trigger_examples = []
    missed_command_examples = []
    for example in examples:
        inputs = prepare_input(example, meta)
        result = run_inference(interpreter, input_details, output_details, inputs, meta)
        routed_label = route_label(result, meta)
        record = {
            "path": format_example_name(example),
            "expected_label": example["expected_label"],
            "predicted_label": routed_label,
            "raw_label": result["label"],
            "confidence": result["confidence"],
            "second_label": result["second_label"],
            "second_confidence": result["second_confidence"],
            "margin": result["margin"],
            "probabilities": result["probabilities"],
        }
        records.append(record)
        if example["expected_label"] in NON_COMMAND_LABELS and routed_label in COMMAND_LABELS:
            false_trigger_examples.append(record)
        if example["expected_label"] in COMMAND_LABELS and routed_label != example["expected_label"]:
            missed_command_examples.append(record)

    labels = list(meta["labels"])
    metrics = compute_metrics(records, labels)
    metrics["model"] = {
        "model_tflite": str(model_tflite),
        "model_meta": str(model_meta),
        "labels": labels,
    }
    metrics["evaluation_config"] = {
        "max_examples_to_show": max_examples_to_show,
        "note": (
            "Metrics use the current offline routed output "
            "(threshold + margin + command gate). "
            "Streaming stability/cooldown are not simulated here."
        ),
    }
    metrics["examples"] = {
        "false_triggers": false_trigger_examples[:max_examples_to_show],
        "missed_commands": missed_command_examples[:max_examples_to_show],
    }
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tflite")
    parser.add_argument("--model-meta")
    parser.add_argument("--on-dir", default="model/data/data_on")
    parser.add_argument("--off-dir", default="model/data/data_off")
    parser.add_argument("--unknown-dir", action="append")
    parser.add_argument("--silence-count", type=int, default=DEFAULT_SILENCE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-json")
    parser.add_argument("--save-confusion-matrix")
    parser.add_argument("--show-confusion-matrix", action="store_true")
    parser.add_argument("--max-examples-to-show", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_tflite and not args.model_meta:
        raise SystemExit("--model-meta is required when --model-tflite is provided")
    if args.model_meta and not args.model_tflite:
        raise SystemExit("--model-tflite is required when --model-meta is provided")

    if args.model_tflite and args.model_meta:
        model_tflite = Path(args.model_tflite)
        model_meta = Path(args.model_meta)
    else:
        model_tflite, model_meta = find_latest_model_pair(Path(".omx/tmp"))

    unknown_dirs = [Path(path) for path in (args.unknown_dir or ["model/data/unknown"])]
    examples = build_example_list(
        Path(args.on_dir),
        Path(args.off_dir),
        unknown_dirs,
        args.silence_count,
        args.seed,
    )
    metrics = evaluate_dataset(
        model_tflite,
        model_meta,
        examples,
        args.max_examples_to_show,
    )
    payload = json.dumps(metrics, indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")

    if args.save_confusion_matrix or args.show_confusion_matrix:
        render_confusion_matrix(
            metrics,
            output_path=args.save_confusion_matrix,
            show=args.show_confusion_matrix,
        )

    print(payload)


if __name__ == "__main__":
    main()
