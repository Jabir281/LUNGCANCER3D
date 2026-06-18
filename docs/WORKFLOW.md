# Workflow (Clean + Repeatable)

This repo trains and evaluates a patch-based 3D CNN using pre-extracted `.npy` volumes of shape `64×64×64`.

## 1) Environment

```bash
pip install -r requirements.txt
```

## 2) Data

You should have:
- `luna16_processed/` containing `subset0_processed.zip` … `subset9_processed.zip`

Extract patches into `./data`:

```bash
python -m scripts.setup_data
```

After this, `./data` contains many `.npy` patch files named like `*_pos_*.npy` and `*_neg_*.npy`.

## 3) Train

```bash
python -m scripts.train
```

Outputs:
- `outputs/checkpoints/best_model.pth` (checkpoint with best validation accuracy)
- Console logs (metrics per epoch)

## 4) Confusion Matrix (Evaluation)

```bash
python -m scripts.confusion_matrix_eval --data-dir ./data --checkpoint outputs/checkpoints/best_model.pth
```

Outputs:
- `outputs/figures/confusion_matrix.png`
- Printed TN/FP/FN/TP + Accuracy/Precision/Recall/F1 at the chosen threshold

Optional: choose the threshold that maximizes F1 on the validation split:

```bash
python -m scripts.confusion_matrix_eval --data-dir ./data --checkpoint outputs/checkpoints/best_model.pth --find-best-threshold
```

## 5) Visualize Predictions (Notebook)

Open `notebooks/visualize_prediction.ipynb` and run cells in order.
- Uses `./data` if present; otherwise can sample from `data_raw/luna16_processed.zip`.
