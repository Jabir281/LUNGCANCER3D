import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from contextlib import nullcontext
try:
    from torch.amp import GradScaler
except ImportError:  # fallback for older PyTorch
    from torch.cuda.amp import GradScaler
from tqdm import tqdm
import numpy as np

from lungcancer3d.dataset import LungCancerDataset
from lungcancer3d.model import LungCancer3DCNN, count_parameters


def get_autocast_context(device: torch.device):
    """Return an autocast context manager appropriate for the current device."""
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda")
    from torch.cuda.amp import autocast
    return autocast()

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # For A100 GPU: Enable cudnn benchmark for better performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def calculate_metrics(outputs, labels):
    """Calculate classification metrics."""
    # Convert logits to probabilities
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()
    
    # Move to CPU and convert to numpy
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    probs_np = probs.cpu().detach().numpy()
    
    labels_np = labels_np.reshape(-1).astype(np.int64)
    preds_np = preds_np.reshape(-1).astype(np.int64)
    probs_np = probs_np.reshape(-1).astype(np.float64)

    # Accuracy
    accuracy = float((preds_np == labels_np).mean()) if labels_np.size else 0.0

    # Precision/Recall/F1
    tp = int(((preds_np == 1) & (labels_np == 1)).sum())
    fp = int(((preds_np == 1) & (labels_np == 0)).sum())
    fn = int(((preds_np == 0) & (labels_np == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # AUC (ROC AUC). If only one class present, AUC is undefined -> 0.0.
    pos = int((labels_np == 1).sum())
    neg = int((labels_np == 0).sum())
    if pos == 0 or neg == 0:
        auc = 0.0
    else:
        order = np.argsort(-probs_np)
        y = labels_np[order]
        tps = np.cumsum(y)
        fps = np.cumsum(1 - y)
        tpr = tps / pos
        fpr = fps / neg
        tpr = np.concatenate(([0.0], tpr))
        fpr = np.concatenate(([0.0], fpr))
        auc = float(np.trapz(tpr, fpr))
    
    return accuracy, precision, recall, f1, auc

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training (CUDA only)
        with get_autocast_context(device):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        all_outputs.append(outputs.detach())
        all_labels.append(labels.detach())
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Calculate epoch metrics
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    accuracy, precision, recall, f1, auc = calculate_metrics(all_outputs, all_labels)
    avg_loss = running_loss / len(train_loader)
    
    return avg_loss, accuracy, precision, recall, f1, auc

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    
    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Mixed precision inference
            with get_autocast_context(device):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            all_outputs.append(outputs)
            all_labels.append(labels)
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Calculate validation metrics
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    accuracy, precision, recall, f1, auc = calculate_metrics(all_outputs, all_labels)
    avg_loss = running_loss / len(val_loader)
    
    return avg_loss, accuracy, precision, recall, f1, auc

def main():
    project_root = Path(__file__).resolve().parents[1]

    # Configuration
    DATA_DIR = project_root / "data"  # Path to extracted .npy files
    CHECKPOINT_PATH = project_root / "outputs" / "checkpoints" / "best_model.pth"
    BATCH_SIZE = 128  # Optimized for A100 (80GB VRAM)
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    VALIDATION_SPLIT = 0.2
    NUM_WORKERS = 7  # Set to 7 as requested
    SEED = 42
    
    # Set random seed
    set_seed(SEED)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load dataset
    print("\nLoading dataset...")
    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"Warning: Data directory '{DATA_DIR}' not found. Run: python -m scripts.setup_data")
        # Create dummy dataset for demonstration if needed, or just exit
        # For now, we'll proceed assuming the user will fix the path
    
    try:
        full_dataset = LungCancerDataset(str(DATA_DIR))
        
        # Split into train and validation
        val_size = int(VALIDATION_SPLIT * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED)
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Create data loaders with RunPod optimizations
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True if NUM_WORKERS > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True if NUM_WORKERS > 0 else False
        )
        
        # Initialize model
        print("\nInitializing model...")
        model = LungCancer3DCNN().to(device)
        print(f"Model has {count_parameters(model):,} trainable parameters")
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Mixed precision scaler (enabled on CUDA)
        scaler = GradScaler(enabled=(device.type == "cuda"))
        
        # Training loop
        print("\nStarting training...")
        best_val_accuracy = 0.0
        train_history = []
        val_history = []
        
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device
            )
            
            # Validate
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate(
                model, val_loader, criterion, device
            )
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Print epoch results
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                  f"Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                  f"Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
            
            # Save history
            train_history.append({
                'epoch': epoch + 1,
                'loss': train_loss,
                'accuracy': train_acc,
                'precision': train_prec,
                'recall': train_rec,
                'f1': train_f1,
                'auc': train_auc
            })
            
            val_history.append({
                'epoch': epoch + 1,
                'loss': val_loss,
                'accuracy': val_acc,
                'precision': val_prec,
                'recall': val_rec,
                'f1': val_f1,
                'auc': val_auc
            })
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                }, str(CHECKPOINT_PATH))
                print(f"✓ Saved best model with validation accuracy: {val_acc:.4f}")
                print(f"  Path: {CHECKPOINT_PATH}")
        
        print("\n" + "="*60)
        print(f"Training completed!")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        print("="*60)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Please check your data directory and file formats.")

if __name__ == "__main__":
    main()
