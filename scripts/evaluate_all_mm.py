import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# Import dataset and context from your compare_models script
from scripts.compare_models import LungCancerDatasetWithUID, get_autocast_context

# Import your four updated architectures
from scripts.mymodels_mm import (
    EfficientNetB0_MLP,
    EfficientNetB1_MLP,
    ResNet34_MLP,
    ResNet50_MLP
)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    metadata_df = None
    if Path('data/metadata_all.csv').exists():
        metadata_df = pd.read_csv('data/metadata_all.csv')
    elif Path('../data/metadata_all.csv').exists():
        metadata_df = pd.read_csv('../data/metadata_all.csv')
    else:
        print('Metadata not found. Cannot compute size stratification.')
        
    # Dataset Directories (Safely checks relative paths)
    test_dir = Path('data/test') if Path('data/test').exists() else Path('../data/test')
    if not test_dir.exists():
        print('Test data folder not found.')
        sys.exit(1)
        
    test_dataset = LungCancerDatasetWithUID(test_dir, metadata_df=metadata_df)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Define models dictionary (UPDATED)
    models = {
        'EfficientNet-B0': EfficientNetB0_MLP(pretrained=False).to(device),
        'EfficientNet-B1': EfficientNetB1_MLP(pretrained=False).to(device),
        '3D ResNet-34': ResNet34_MLP(pretrained=False).to(device),
        '3D ResNet-50': ResNet50_MLP(pretrained=False).to(device)
    }
    
    # Define matching weights dictionary (UPDATED)
    weights = {
        'EfficientNet-B0': 'EfficientNetB0_MLP_best.pth',
        'EfficientNet-B1': 'EfficientNetB1_MLP_best.pth',
        '3D ResNet-34': 'ResNet34_MLP_best.pth',
        '3D ResNet-50': 'ResNet50_MLP_best.pth'
    }
    
    for k, v in weights.items():
        weight_path = Path(v)
        # Check parent directory fallback if not found locally
        if not weight_path.exists():
            weight_path = Path("../" + v)
            
        if weight_path.exists():
            print(f"Loading weights for {k} from {weight_path}")
            models[k].load_state_dict(torch.load(weight_path, map_location=device))
            models[k].eval()
        else:
            print(f'Warning: {v} not found. Running {k} with uninitialized weights.')
            models[k].eval()

    results_data = {}
    
    for name, model in models.items():
        print(f'Evaluating {name} on test set...')
        all_probs, all_labels, all_uids, all_diameters = [], [], [], []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Evaluating {name}'):
                if len(batch) == 7:
                    inputs, labels, uids, _, _, _, ds = batch
                else:
                    inputs, labels, uids = batch[0], batch[1], batch[2]
                    ds = [0] * len(labels)
                
                inputs = inputs.to(device)
                
                with get_autocast_context(device):
                    outputs = model(inputs).view(-1)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
                all_uids.extend(uids)
                all_diameters.extend([d.item() if torch.is_tensor(d) else d for d in ds])
                
        results_data[name] = {
            'probs': np.array(all_probs),
            'labels': np.array(all_labels),
            'uids': np.array(all_uids),
            'diameters': np.array(all_diameters)
        }

    cpm_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    table_rows = []
    
    plt.figure(figsize=(8, 6))
    
    for name, data in results_data.items():
        probs = data['probs']
        labels = data['labels'].astype(int)
        uids = data['uids']
        
        scan_probs, scan_labels = {}, {}
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
        
        sorted_indices = np.argsort(g_probs)[::-1]
        sorted_labels = g_labels[sorted_indices]
        
        num_scans = len(scan_labels)
        fps, tps = 0, 0
        total_pos = sum(g_labels == 1)
        froc_fps_scan, froc_sens = [], []
        
        for i in range(len(sorted_labels)):
            if sorted_labels[i] == 1: 
                tps += 1
            else: 
                fps += 1
            froc_fps_scan.append(fps / num_scans if num_scans > 0 else 0)
            froc_sens.append(tps / total_pos if total_pos > 0 else 0)
            
        cpm_sensitivities = []
        for target_rate in cpm_rates:
            idx = np.searchsorted(froc_fps_scan, target_rate)
            if idx >= len(froc_sens): 
                cpm_sensitivities.append(froc_sens[-1] if len(froc_sens) > 0 else 0.0)
            else: 
                cpm_sensitivities.append(froc_sens[idx])
        cpm_value = np.mean(cpm_sensitivities)
        
        table_rows.append({
            'Model': name, 'Test AUC': auc, 'Sensitivity': sensitivity, 
            'Specificity': specificity, 'F1': f1, 'CPM': cpm_value
        })
        
        plt.plot(froc_fps_scan, froc_sens, label=f'{name} (CPM={cpm_value:.3f})')

    df_table = pd.DataFrame(table_rows)
    print('\n--- 1. Main Comparison Table ---')
    print(df_table.to_markdown(index=False))
    
    with open('evaluation_metrics.md', 'w') as f:
        f.write('# 1. Main Comparison Table\n\n')
        f.write(df_table.to_markdown(index=False))
        f.write('\n\n')

    plt.xscale('log', base=2)
    plt.xticks(cpm_rates, labels=[str(r) for r in cpm_rates])
    plt.xlim(0.1, 8.5)
    plt.ylim(0, 1.05)
    plt.xlabel('False Positives per Scan')
    plt.ylabel('Sensitivity')
    plt.title('FROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('FROC_Curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('\n--- 2. FROC Curve Plot ---')
    print('Saved to FROC_Curve.png')

    # Size stratification focusing on 3D ResNet-50 as our focal deep network
    target_strat_model = '3D ResNet-50'
    print(f'\n--- 3. Size Stratification Table ({target_strat_model}) ---')
    try:
        if metadata_df is not None and target_strat_model in results_data:
            h_data = results_data[target_strat_model]
            is_pos = h_data['labels'] == 1
            pos_probs = h_data['probs'][is_pos]
            pos_diams = h_data['diameters'][is_pos]
            
            sizes = [
                ('<3mm', pos_diams < 3),
                ('3-10mm', (pos_diams >= 3) & (pos_diams <= 10)),
                ('>10mm', pos_diams > 10)
            ]
            
            size_rows = []
            for label, mask in sizes:
                group_probs = pos_probs[mask]
                total = len(group_probs)
                if total > 0:
                    detected = sum(group_probs >= 0.5)
                    sens = detected / total
                else:
                    sens = 0.0
                    detected = 0
                size_rows.append({
                    'Nodule Size': label, 
                    'Total Positive Patches': total, 
                    'Detected': detected, 
                    'Sensitivity': sens
                })
                
            df_size = pd.DataFrame(size_rows)
            print(df_size.to_markdown(index=False))
            
            with open('evaluation_metrics.md', 'a') as f:
                f.write(f'# 3. Size Stratification Table ({target_strat_model})\n\n')
                f.write(df_size.to_markdown(index=False))
                f.write('\n')
                
    except Exception as e:
        print(f'Could not compute size stratification: {e}')

if __name__ == '__main__':
    main()