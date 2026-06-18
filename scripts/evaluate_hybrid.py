import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from scripts.compare_models import LungCancerDatasetWithUID, get_autocast_context, ResNet18WithDropout, HybridAttention3DCNN
from scripts.models_hybrid import TrueHybrid3D
from scripts.train_hybrid import evaluate_metrics_scan_level
from monai.networks.nets import DenseNet121

def evaluate_model(model, loader, device):
    model.eval()
    all_probs, all_labels, all_uids = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs, labels, uids = batch[0].to(device), batch[1].to(device), batch[2]
            with get_autocast_context(device):
                # Account for models that output (B,) instead of (B, 1) properly
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).view(-1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_uids.extend(uids)
    return evaluate_metrics_scan_level(all_probs, all_labels, all_uids)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dir = Path("data/test")
    test_dataset = LungCancerDatasetWithUID(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    models = {
        "3D ResNet-18 (Baseline)": (ResNet18WithDropout(), "3D_ResNet-18_best.pth"),
        "Hybrid Attention 3D-CNN (Standalone)": (HybridAttention3DCNN(), "Hybrid_Attention_3D-CNN_best.pth"),
        "TrueHybrid3D (Ours)": (TrueHybrid3D("3D_ResNet-18_best.pth"), "TrueHybrid3D_best.pth")
    }
    
    results = {}
    
    for name, (model, path) in models.items():
        if not Path(path).exists():
            print(f"Warning: {path} not found. Skipping {name}.")
            continue
            
        print(f"Evaluating: {name}")
        model.to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        
        metrics = evaluate_model(model, test_loader, device)
        results[name] = metrics
        
    # Generate Markdown Table
    md_table = "| Model | Test AUC | Test Sensitivity | Test Specificity | Test F1 | FROC @ 0.125 | FROC @ 0.25 | Max Achievable FP Rate |\n"
    md_table += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    
    for name, metrics in results.items():
        froc_points = {k: v for k, v in metrics["FROC_points"]}
        froc_0125 = froc_points.get(0.125, "N/A")
        if froc_0125 != "N/A": froc_0125 = f"{froc_0125:.4f}"
        
        froc_025 = froc_points.get(0.25, "N/A")
        if froc_025 != "N/A": froc_025 = f"{froc_025:.4f}"
        
        md_table += f"| {name} | {metrics['ROC-AUC']:.4f} | {metrics['Sensitivity']:.4f} | {metrics['Specificity']:.4f} | {metrics['F1-Score']:.4f} | {froc_0125} | {froc_025} | {metrics['Max_achievable_FP_rate']:.4f} |\n"
        
    with open("TrueHybrid3D_Evaluation.md", "w") as f:
        f.write("# TrueHybrid3D Final Evaluation\n\n")
        f.write(md_table)
        
    print("\n" + md_table)

if __name__ == "__main__":
    main()
