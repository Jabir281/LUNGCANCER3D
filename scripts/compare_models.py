import os
import argparse
import sys
from datetime import datetime
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from contextlib import nullcontext

try:
    from torch.amp import GradScaler
    import monai
    from monai.networks.nets import DenseNet121, resnet18
    from monai.transforms import Compose, RandRotate90, RandFlip, RandGaussianNoise, ToTensor
except ImportError:
    print("Please install monai first: pip install monai")
    sys.exit(1)

import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

class LungCancerDatasetWithUID(Dataset):
    """
    Custom Dataset for 3D Lung Cancer patches that also returns the seriesuid.
    """
    def __init__(self, data_dir, metadata_df=None, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.file_list = list(self.data_dir.rglob("*.npy"))
        
        self.labels = []
        self.uids = []
        self.coords_x = []
        self.coords_y = []
        self.coords_z = []
        self.diameters = []

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
            
            # Extract metadata
            cx, cy, cz, d = 0.0, 0.0, 0.0, 0.0
            if metadata_df is not None:
                match = metadata_df[metadata_df['filepath'].str.endswith(file_path.name)]
                if not match.empty:
                    row = match.iloc[0]
                    cx = float(row.get('coord_world_X', 0.0))
                    cy = float(row.get('coord_world_Y', 0.0))
                    cz = float(row.get('coord_world_Z', 0.0))
                    d_val = row.get('diameter_mm', 0.0)
                    if pd.notna(d_val): d = float(d_val)
                    
            self.coords_x.append(cx)
            self.coords_y.append(cy)
            self.coords_z.append(cz)
            self.diameters.append(d)
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        patch = np.load(file_path).astype(np.float32)
        patch = np.expand_dims(patch, axis=0) # (1, 64, 64, 64)
        patch = torch.from_numpy(patch)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform:
            patch = self.transform(patch)
            
        uid = self.uids[idx]
        cx = self.coords_x[idx]
        cy = self.coords_y[idx]
        cz = self.coords_z[idx]
        d = self.diameters[idx]
        
        return patch, label, uid, cx, cy, cz, d

def get_autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda")
    from torch.cuda.amp import autocast
    return autocast()

def evaluate_metrics(probs, labels, uids):
    # Group by seriesuid (take max probability for each scan)
    scan_probs = {}
    scan_labels = {}
    for p, l, u in zip(probs, labels, uids):
        if u not in scan_labels:
            scan_labels[u] = l
            scan_probs[u] = p
        else:
            scan_labels[u] = max(scan_labels[u], l)
            scan_probs[u] = max(scan_probs[u], p)
            
    g_probs = np.array(list(scan_probs.values()))
    g_labels = np.array(list(scan_labels.values())).astype(int)
    g_preds = (g_probs >= 0.5).astype(int)
    
    auc = roc_auc_score(g_labels, g_probs) if len(np.unique(g_labels)) > 1 else 0.0
    f1 = f1_score(g_labels, g_preds)
    cm = confusion_matrix(g_labels, g_preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # FROC Calculation
    num_scans = len(scan_labels)
    cpm_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    
    sorted_indices = np.argsort(g_probs)[::-1]
    sorted_probs = g_probs[sorted_indices]
    sorted_labels = g_labels[sorted_indices]
    
    fps = 0
    tps = 0
    total_pos = sum(g_labels == 1)
    
    froc_fps_scan = []
    froc_sens = []
    
    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tps += 1
        else:
            fps += 1
            
        current_fp_rate = fps / num_scans if num_scans > 0 else 0
        current_sens = tps / total_pos if total_pos > 0 else 0
        
        froc_fps_scan.append(current_fp_rate)
        froc_sens.append(current_sens)
        
    max_fp_rate = (g_labels == 0).sum() / num_scans if num_scans > 0 else 0.0
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

def _evaluate_loader(model, loader, desc, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_uids = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            if len(batch) == 7: # Dataset returns 7 items
                inputs, labels, uids = batch[0], batch[1], batch[2]
            else:
                inputs, labels, uids = batch[0], batch[1], batch[2]
            
            inputs = inputs.to(device)
            
            with get_autocast_context(device):
                outputs = model(inputs).view(-1)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_uids.extend(uids)
            
    return evaluate_metrics(all_probs, all_labels, all_uids)

def train_and_eval(model_name, model, train_loader, val_loader, test_loader, device, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) # Added weight decay
    criterion = nn.BCEWithLogitsLoss() # Added pos_weight
    
    scaler = GradScaler() if device.type == "cuda" else None
    best_val_auc = 0.0
    patience = 15
    patience_counter = 0
    best_model_state = None
    
    print(f"\n--- Training {model_name} ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            if len(batch) == 7:
                inputs, labels, _ = batch[0], batch[1], batch[2]
            else:
                inputs, labels, _ = batch[0], batch[1], batch[2]
                
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            with get_autocast_context(device):
                outputs = model(inputs).view(-1)
                loss = criterion(outputs, labels)
                
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        # Early stopping check
        val_metrics = _evaluate_loader(model, val_loader, "Val", device)
        current_auc = val_metrics["ROC-AUC"]
        print(f"Val AUC: {current_auc:.4f}")
        
        if current_auc > best_val_auc:
            best_val_auc = current_auc
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        torch.save(best_model_state, f"{model_name.replace(' ', '_')}_best.pth")
        print(f"Saved best model to {model_name.replace(' ', '_')}_best.pth")
    
        
    print(f"\n--- Evaluating {model_name} on Val Set ---")
    val_metrics = _evaluate_loader(model, val_loader, "Val", device)
    
    print(f"\n--- Evaluating {model_name} on Test Set ---")
    test_metrics = _evaluate_loader(model, test_loader, "Test", device)
    
    return {"val": val_metrics, "test": test_metrics}

import torch.nn.functional as F

class SEBlock3D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = F.adaptive_avg_pool3d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1, 1)
        return x * y

class HybridAttention3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
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
        # Input patch is 64x64x64.
        # pool1 (MaxPool3d 2) -> 32x32x32
        # pool2 (MaxPool3d 2) -> 16x16x16
        # pool3 (MaxPool3d 4) -> 4x4x4
        # 128 channels * 4 * 4 * 4 = 128 * 64 = 8192
        self.fc1 = nn.Linear(8192, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
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

class ResNet18WithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        # Use num_classes for the feature vector size before fully connected layer
        self.resnet = resnet18(spatial_dims=3, n_input_channels=1, num_classes=512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        return self.fc(x)

def main():
    parser = argparse.ArgumentParser("Compare 3D DenseNet-121 and 3D ResNet-18")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with .npy patches")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    metadata_df = None
    if Path("data/metadata_all.csv").exists():
        metadata_df = pd.read_csv("data/metadata_all.csv")
    print(f"Using device: {device}")
    
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"
    test_dir = Path(args.data_dir) / "test"
    
    from monai.transforms import Compose, RandRotate90, RandFlip, RandGaussianNoise
    
    print(f"Loading train data from {train_dir}")
    train_transforms = Compose([
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        RandRotate90(prob=0.5, spatial_axes=(0, 2)),
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
        RandGaussianNoise(prob=0.5, std=0.01)
    ])
    train_dataset = LungCancerDatasetWithUID(train_dir, metadata_df=metadata_df, transform=train_transforms)

    # Test transform on one sample
    sample = train_dataset[0][0]  # (1, 64, 64, 64)
    aug_sample = train_transforms(sample)
    print("Before:", sample.shape, "After:", aug_sample.shape)
    print("Before min/max:", sample.min(), sample.max())
    print("After min/max:", aug_sample.min(), aug_sample.max())
    # Visualize middle slice of both
    
    print(f"Loading val data from {val_dir}")
    val_dataset = LungCancerDatasetWithUID(val_dir, metadata_df=metadata_df)
    
    print(f"Loading test data from {test_dir}")
    test_dataset = LungCancerDatasetWithUID(test_dir, metadata_df=metadata_df)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("Not enough data to train, validate and test. Exiting.")
        sys.exit(1)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    models = {
        "Hybrid Attention 3D-CNN": HybridAttention3DCNN().to(device)
    }
    
    results = {}
    
    for name, model in models.items():
        metrics = train_and_eval(name, model, train_loader, val_loader, test_loader, device, epochs=args.epochs)
        results[name] = metrics
        
    log_path = Path("log_result_new.txt")
    with open(log_path, "a") as f:
        f.write(f"\n===== Date: {datetime.now().isoformat()} =====\n")
        for name, metrics_dict in results.items():
            f.write(f"\nModel: {name}\n")
            for split_name, metrics in metrics_dict.items():
                f.write(f"\n  [{split_name.upper()} SET]\n")
                f.write(f"  Sensitivity (Recall): {metrics['Sensitivity (Recall)']:.4f}\n")
                f.write(f"  Specificity: {metrics['Specificity']:.4f}\n")
                f.write(f"  ROC-AUC Curve: {metrics['ROC-AUC']:.4f}\n")
                f.write(f"  F1-Score: {metrics['F1-Score']:.4f}\n")
                f.write(f"  CPM Value: {metrics['CPM']:.4f}\n")
                f.write("  FROC Curve data (Sensitivity at 0.125, 0.25, 0.5, 1, 2, 4, 8 FPs/scan):\n")
                rates = metrics['FROC_rates']
                sensitivities = metrics['FROC_sensitivities']
                for r, s in zip(rates, sensitivities):
                    f.write(f"    - FP/scan: {r} -> Sensitivity: {s:.4f}\n")
                
    print(f"\nResults have been successfully saved to {log_path.absolute()}")

if __name__ == "__main__":
    main()