import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

try:
    from torch.amp import GradScaler
    from monai.transforms import Compose, RandRotate90, RandFlip, RandGaussianNoise
except ImportError:
    print("Please install monai first: pip install monai")
    sys.exit(1)

from scripts.compare_models import LungCancerDatasetWithUID, get_autocast_context
from scripts.models_hybrid import TrueHybrid3D

# From instruction 5.2 (evaluate_metrics_scan_level EXACT copy)
def evaluate_metrics_scan_level(probs, labels, uids):
    # Group by seriesuid (max probability per scan)
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
    
    # Standard metrics
    auc = roc_auc_score(g_labels, g_probs)
    preds = (g_probs >= 0.5).astype(int)
    f1 = f1_score(g_labels, preds)
    cm = confusion_matrix(g_labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # FROC (achievable rates only)
    num_scans = len(scan_labels)
    num_neg_scans = sum(g_labels == 0)
    max_fp_rate = num_neg_scans / num_scans if num_scans > 0 else 0.0
    
    # Sort descending
    sorted_indices = np.argsort(g_probs)[::-1]
    sorted_labels = g_labels[sorted_indices]
    
    fps = 0
    tps = 0
    total_pos = sum(g_labels == 1)
    froc_fps = []
    froc_sens = []
    
    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tps += 1
        else:
            fps += 1
        froc_fps.append(fps / num_scans if num_scans > 0 else 0)
        froc_sens.append(tps / total_pos if total_pos > 0 else 0)
    
    # Only achievable standard rates
    standard_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    achievable_rates = [r for r in standard_rates if r <= max_fp_rate]
    
    froc_points = []
    for rate in achievable_rates:
        idx = np.searchsorted(froc_fps, rate)
        sens = froc_sens[idx] if idx < len(froc_sens) else (froc_sens[-1] if froc_sens else 0.0)
        froc_points.append((rate, sens))
    
    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "ROC-AUC": auc,
        "F1-Score": f1,
        "FROC_points": froc_points,
        "Max_achievable_FP_rate": max_fp_rate
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Augmentation (Training Only)
    train_transforms = Compose([
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        RandRotate90(prob=0.5, spatial_axes=(0, 2)),
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
        RandGaussianNoise(prob=0.5, std=0.01)
    ])

    train_dir = Path("data/train")
    val_dir = Path("data/val")
    
    train_dataset = LungCancerDatasetWithUID(train_dir, transform=train_transforms)
    val_dataset = LungCancerDatasetWithUID(val_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True)
    
    resnet_path = "3D_ResNet-18_best.pth"
    if not Path(resnet_path).exists():
        print(f"Error: {resnet_path} not found!")
        sys.exit(1)
        
    model = TrueHybrid3D(resnet_path).to(device)
    
    # 10. First Verification Step
    print("Verifying model process one batch...")
    sample = next(iter(train_loader))[0].to(device)
    print("Input shape:", sample.shape)  # Should be (B, 1, 64, 64, 64)
    with get_autocast_context(device):
        output = model(sample)
    print("Output shape:", output.shape)  # Should be (B, 1)
    print("Output range:", output.min().item(), output.max().item())
    assert output.shape == (sample.size(0), 1), "Output dimension mismatch!"
    print("Verification passed! Starting training...")
    
    # 4.1 Optimizer Setup
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = 50
    patience = 15
    best_auc = 0.0
    patience_counter = 0
    scaler = GradScaler() if device.type == "cuda" else None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, labels, uids = batch[0].to(device), batch[1].to(device), batch[2]
            labels = labels.unsqueeze(1)
            
            optimizer.zero_grad()
            
            with get_autocast_context(device):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_probs, val_labels, val_uids = [], [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs, labels, uids = batch[0].to(device), batch[1].to(device), batch[2]
                
                with get_autocast_context(device):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.unsqueeze(1))
                    probs = torch.sigmoid(outputs)
                    
                val_loss += loss.item()
                val_probs.extend(probs.view(-1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_uids.extend(uids)
                
        val_loss /= len(val_loader)
        metrics = evaluate_metrics_scan_level(val_probs, val_labels, val_uids)
        current_auc = metrics['ROC-AUC']
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val AUC: {current_auc:.4f} | Val Sens: {metrics['Sensitivity']:.4f} | Val Spec: {metrics['Specificity']:.4f}")
        
        if current_auc > best_auc:
            best_auc = current_auc
            torch.save(model.state_dict(), "TrueHybrid3D_best.pth")
            print("=> Saved new best model")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"=> Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

if __name__ == "__main__":
    main()
