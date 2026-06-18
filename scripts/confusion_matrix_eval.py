import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lungcancer3d.dataset import LungCancerDataset
from lungcancer3d.model import LungCancer3DCNN


@dataclass
class EvalResult:
    threshold: float
    cm: np.ndarray
    accuracy: float
    precision: float
    recall: float
    f1: float


def _confusion_matrix_2x2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return 2x2 confusion matrix as [[TN, FP], [FN, TP]]."""
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def _metrics_from_cm(cm: np.ndarray) -> tuple[float, float, float, float]:
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return float(accuracy), float(precision), float(recall), float(f1)


def _normalize_cm(cm: np.ndarray, mode: str | None) -> np.ndarray:
    if mode is None or mode == "none":
        return cm.astype(np.int64)

    cm = cm.astype(np.float64)
    if mode == "true":
        denom = cm.sum(axis=1, keepdims=True)
    elif mode == "pred":
        denom = cm.sum(axis=0, keepdims=True)
    elif mode == "all":
        denom = cm.sum()
        return cm / denom if denom > 0 else cm
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")

    denom = np.where(denom == 0, 1.0, denom)
    return cm / denom


def _load_checkpoint_model(checkpoint_path: str, device: torch.device) -> LungCancer3DCNN:
    model = LungCancer3DCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Fallback: allow a raw state_dict checkpoint.
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def _predict_probs(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    seen = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs).float().view(-1)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy().astype(np.int32))

        seen += int(labels.numel())
        if max_samples and seen >= max_samples:
            break

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return y_true, y_prob


def _eval_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> EvalResult:
    y_pred = (y_prob >= threshold).astype(np.int32)

    cm = _confusion_matrix_2x2(y_true, y_pred)
    acc, prec, rec, f1 = _metrics_from_cm(cm)

    return EvalResult(
        threshold=threshold,
        cm=cm,
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
    )


def _find_best_threshold_for_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    step: float = 0.01,
) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0

    thresholds = np.arange(0.0, 1.0 + 1e-9, step)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int32)
        cm = _confusion_matrix_2x2(y_true, y_pred)
        _, _, _, f1 = _metrics_from_cm(cm)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)

    return best_t, best_f1


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained checkpoint on the validation split and generate a confusion matrix. "
            "This uses the same patch-level random split logic as train.py."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=str(project_root / "data"),
        help="Directory containing extracted .npy patches",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(project_root / "outputs" / "checkpoints" / "best_model.pth"),
        help="Path to checkpoint (.pth)",
    )
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction (same as train.py)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (same as train.py)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (use 0 on Windows)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold applied to sigmoid probabilities",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on number of validation samples to evaluate (0 = all). Useful on CPU.",
    )
    parser.add_argument(
        "--find-best-threshold",
        action="store_true",
        help="Find threshold that maximizes F1 on the validation split",
    )
    parser.add_argument(
        "--save-fig",
        default=os.path.join("outputs", "figures", "confusion_matrix.png"),
        help="Output image path for confusion matrix (set empty to disable)",
    )
    parser.add_argument(
        "--normalize",
        choices=["none", "true", "pred", "all"],
        default="true",
        help="Normalization mode for the plotted confusion matrix",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(
            f"Data directory '{args.data_dir}' not found. "
            "Point --data-dir to your extracted .npy patch folder (same as train.py DATA_DIR)."
        )

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint '{args.checkpoint}' not found. "
            "Point --checkpoint to best_model.pth."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = LungCancerDataset(args.data_dir)
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size

    _, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = _load_checkpoint_model(args.checkpoint, device)
    y_true, y_prob = _predict_probs(model, val_loader, device, max_samples=args.max_samples)

    if args.find_best_threshold:
        best_t, best_f1 = _find_best_threshold_for_f1(y_true, y_prob)
        print(f"Best threshold by F1 (val): {best_t:.2f} (F1={best_f1:.4f})")
        threshold = best_t
    else:
        threshold = float(args.threshold)

    result = _eval_at_threshold(y_true, y_prob, threshold)

    tn, fp, fn, tp = result.cm.ravel()
    print("\nConfusion matrix (labels: 0=neg, 1=pos):")
    print(result.cm)
    print("\nCounts:")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")

    print("\nMetrics at threshold {:.2f}:".format(result.threshold))
    print(f"  Accuracy : {result.accuracy:.4f}")
    print(f"  Precision: {result.precision:.4f}")
    print(f"  Recall   : {result.recall:.4f}")
    print(f"  F1-score : {result.f1:.4f}")

    if args.save_fig:
        out_dir = os.path.dirname(os.path.abspath(args.save_fig))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        import matplotlib.pyplot as plt

        y_pred = (y_prob >= result.threshold).astype(np.int32)
        cm = _confusion_matrix_2x2(y_true, y_pred)
        normalize = None if args.normalize == "none" else args.normalize
        cm_plot = _normalize_cm(cm, normalize)

        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        im = ax.imshow(cm_plot, cmap="Blues")
        ax.set_xticks([0, 1], labels=["Negative", "Positive"])
        ax.set_yticks([0, 1], labels=["Negative", "Positive"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        value_format = ".2f" if normalize else "d"
        for (i, j), val in np.ndenumerate(cm_plot):
            ax.text(j, i, format(val, value_format), ha="center", va="center", color="black")

        title_norm = args.normalize
        ax.set_title(f"Confusion Matrix (threshold={result.threshold:.2f}, normalize={title_norm})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(args.save_fig, dpi=200)
        print(f"\nSaved confusion matrix figure to: {args.save_fig}")


if __name__ == "__main__":
    main()
