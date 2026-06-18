import os
import sys
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from contextlib import nullcontext

try:
    from torch.amp import GradScaler
    import monai
    from monai.networks.nets import DenseNet121, resnet18
except ImportError:
    print("Please install monai first: pip install monai")
    sys.exit(1)

import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix


# ============ DATASET ============

class LungCancerDatasetWithUID(Dataset):
    def __init__(self, data_dir, metadata_df=None, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.file_list = list(self.data_dir.rglob("*.npy"))
        
        self.labels = []
        self.uids = []
        for file_path in self.file_list:
            filename = file_path.stem
            if "_pos_" in filename:
                self.labels.append(1)
                uid = filename.split("_pos_")[0]
            elif "_neg_" in filename:
                self.labels.append(0)
                uid = filename.split("_neg_")[0]
            else:
                raise ValueError(f"Invalid filename: {file_path.name}")
            self.uids.append(uid)
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        patch = np.load(file_path).astype(np.float32)
        patch = np.expand_dims(patch, axis=0)
        patch = torch.from_numpy(patch)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            patch = self.transform(patch)
        return patch, label, self.uids[idx]


# ============ MODEL DEFINITIONS ============

class ResNet18WithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(spatial_dims=3, n_input_channels=1, num_classes=512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        return self.fc(x)


class SEBlock3D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        import torch.nn.functional as F
        b, c, _, _, _ = x.size()
        y = F.adaptive_avg_pool3d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1, 1)
        return x * y


class HybridAttention3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        import torch.nn.functional as F
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(2)
        self.se1 = SEBlock3D(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)
        self.se2 = SEBlock3D(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(4)
        self.se3 = SEBlock3D(128)

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(8192, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        import torch.nn.functional as F
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.se1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.se2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.se3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        return x


# ============ PATCH-LEVEL EVALUATION METRICS ============

def evaluate_metrics_patch_level(probs, labels, uids):
    """
    Patch-level evaluation following original LUNA16 challenge protocol.
    Each candidate patch is treated as an independent detection.
    """
    probs = np.array(probs)
    labels = np.array(labels).astype(int)
    preds = (probs >= 0.5).astype(int)
    uids = np.array(uids)
    
    # Standard classification metrics
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # ===== PATCH-LEVEL FROC / CPM =====
    # We calculate the exact number of unique scans in the test set
    num_scans = len(np.unique(uids)) if len(uids) > 0 else 1
    num_patches = len(labels)
    total_pos = sum(labels == 1)
    
    # Sort by descending probability
    sorted_indices = np.argsort(probs)[::-1]
    sorted_labels = labels[sorted_indices]
    
    fps = 0
    tps = 0
    
    froc_fps_scan = []
    froc_sens = []
    
    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tps += 1
        else:
            fps += 1
            
        # Convert to FP per scan using exact scan count
        current_fp_rate = fps / num_scans
        current_sens = tps / total_pos if total_pos > 0 else 0
        
        froc_fps_scan.append(current_fp_rate)
        froc_sens.append(current_sens)
        
    # CPM at standard 7 rates
    cpm_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    max_fp_rate = (labels == 0).sum() / num_scans if num_scans > 0 else 0.0
    achievable_rates = [r for r in cpm_rates if r <= max_fp_rate]
    
    cpm_sensitivities = []
    for target_rate in achievable_rates:
        idx = np.searchsorted(froc_fps_scan, target_rate)
        if idx >= len(froc_sens):
            cpm_sensitivities.append(froc_sens[-1] if len(froc_sens) > 0 else 0.0)
        else:
            cpm_sensitivities.append(froc_sens[idx])
            
    cpm_value = np.mean(cpm_sensitivities) if cpm_sensitivities else 0.0
    
    return {
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "ROC-AUC": auc,
        "F1-Score": f1,
        "CPM": cpm_value,
        "FROC_rates": achievable_rates,
        "FROC_sensitivities": cpm_sensitivities
    }


def get_autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda")
    from torch.cuda.amp import autocast
    return autocast()


def evaluate_model(model, test_loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_uids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, labels, uids = batch[0], batch[1], batch[2]
            inputs = inputs.to(device)
            
            with get_autocast_context(device):
                outputs = model(inputs).view(-1)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_uids.extend(uids)
            
    return evaluate_metrics_patch_level(all_probs, all_labels, all_uids)


# ============ MAIN ============

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_dir = Path("data/test")
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        sys.exit(1)
    
    test_dataset = LungCancerDatasetWithUID(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True)
    
    print(f"Test patches: {len(test_dataset)}")
    
    # Define models and checkpoint paths
    models_to_eval = {
        "3D DenseNet-121": (DenseNet121(spatial_dims=3, in_channels=1, out_channels=1, dropout_prob=0.5), "3D_DenseNet-121_best.pth"),
        "3D ResNet-18": (ResNet18WithDropout(), "3D_ResNet-18_best.pth"),
        "Hybrid Attention 3D-CNN": (HybridAttention3DCNN(), "Hybrid_Attention_3D-CNN_best.pth"),
    }
    
    results = {}
    
    for name, (model, ckpt_path) in models_to_eval.items():
        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            print(f"\n⚠️  Checkpoint not found: {ckpt_path} — skipping {name}")
            continue
            
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print(f"Loading: {ckpt_path}")
        
        model = model.to(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        
        metrics = evaluate_model(model, test_loader, device)
        results[name] = metrics
        
        print(f"\n--- {name} PATCH-LEVEL TEST RESULTS ---")
        print(f"  Sensitivity (Recall): {metrics['Sensitivity (Recall)']:.4f}")
        print(f"  Specificity:          {metrics['Specificity']:.4f}")
        print(f"  ROC-AUC:              {metrics['ROC-AUC']:.4f}")
        print(f"  F1-Score:             {metrics['F1-Score']:.4f}")
        print(f"  CPM (7-point):        {metrics['CPM']:.4f}")
        for r, s in zip(metrics['FROC_rates'], metrics['FROC_sensitivities']):
            print(f"    FP/scan: {r} -> Sensitivity: {s:.4f}")
    
    # Save to markdown
    md_path = Path("re_evaluate_patch_level_results.md")
    with open(md_path, "w") as f:
        f.write("# PATCH-LEVEL CPM RE-EVALUATION\n\n")
        f.write("| Model | Sensitivity | Specificity | ROC-AUC | F1-Score | CPM |\n")
        f.write("|-------|-------------|-------------|---------|----------|-----|\n")
        for name, metrics in results.items():
            f.write(f"| {name} | {metrics['Sensitivity (Recall)']:.4f} | ")
            f.write(f"{metrics['Specificity']:.4f} | {metrics['ROC-AUC']:.4f} | ")
            f.write(f"{metrics['F1-Score']:.4f} | {metrics['CPM']:.4f} |\n")
        
        f.write("\n## FROC Curve Data (7 Standard Rates)\n\n")
        for name, metrics in results.items():
            f.write(f"### {name}\n")
            for r, s in zip(metrics['FROC_rates'], metrics['FROC_sensitivities']):
                f.write(f"- FP/scan: {r} -> Sensitivity: {s:.4f}\n")
            f.write("\n")
                
    print(f"\n{'='*50}")
    print(f"All results saved to: {md_path.absolute()}")


if __name__ == "__main__":
    main()