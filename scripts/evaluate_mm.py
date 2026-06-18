import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Import dataset and context (Updated with scripts. prefix)
from scripts.compare_models import LungCancerDatasetWithUID, get_autocast_context
# Import your new ResNet-50 model (UPDATED)
from scripts.mymodels_mm import ResNet50_MLP
# Import the evaluation logic from your training script (Updated with scripts. prefix)
from scripts.train_mm import evaluate_metrics_scan_level

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
    print(f"Using device: {device}")
    
    # Dataset Directories (Safely checks relative paths)
    test_dir = Path("../data/test") if not Path("data/test").exists() else Path("data/test")
    test_dataset = LungCancerDatasetWithUID(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Define your model and the weights file you just trained (UPDATED)
    models = {
        "ResNet50_MLP": (ResNet50_MLP(pretrained=False), "ResNet50_MLP_best.pth")
    }
    
    results = {}
    
    for name, (model, path) in models.items():
        # Check safely if weights exist locally or in the parent folder
        weight_path = Path(path)
        if not weight_path.exists():
            weight_path = Path("../" + path)
            
        if not weight_path.exists():
            print(f"Warning: {path} not found. Skipping {name}.")
            continue
            
        print(f"Evaluating: {name}")
        model.to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        
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
        
    # (UPDATED file name and header)
    with open("ResNet50_Evaluation.md", "w") as f:
        f.write("# ResNet-50 Final Evaluation\n\n")
        f.write(md_table)
        
    print("\n" + md_table)

if __name__ == "__main__":
    main()