# 3D Lung Cancer Detection System

A PyTorch-based 3D CNN with attention mechanisms for lung cancer detection from CT scan patches, optimized for NVIDIA A100 GPUs.

## 📋 Setup Instructions (RunPod)

### 1. **Clone/Upload Your Code**
Upload all Python files to your RunPod instance:
- `dataset.py`
- `model.py`
- `train.py`
- `setup_data.py`
- `requirements.txt`

### 2. **Download Your Data**
Transfer your OneDrive data to the RunPod instance. You can either:

**Option A: Direct download to RunPod (if link is publicly accessible)**
```bash
# Install rclone or use wget if you have a direct link
```

**Option B: Upload via RunPod's file manager**
- Upload the `luna16_processed` folder containing `subset0.zip` through `subset9.zip`

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Extract the Data**
```bash
python setup_data.py
```
This will:
- Unzip all subset ZIP files
- Extract `.npy` files to `./data/` directory
- Show you statistics about positive/negative samples

### 5. **Update Data Path (if needed)**
Open `train.py` and verify line 123:
```python
DATA_DIR = "./data"  # Should match where setup_data.py extracted files
```

### 6. **Start Training**
```bash
python train.py
```

## 🔧 Expected Data Format

Your `.npy` files should follow this naming convention:
- **Positive (cancer)**: `{uid}_pos_{i}.npy` → Label = 1
- **Negative (no cancer)**: `{uid}_neg_{i}.npy` → Label = 0

Each file should contain a 3D numpy array of shape `(64, 64, 64)`.

## 🚀 Model Architecture

- **Input**: `(Batch, 1, 64, 64, 64)`
- **Backbone**: 4-block 3D CNN
- **Attention**: 
  - Channel Attention (SE-Block style)
  - Spatial Attention (CBAM style)
- **Output**: Single logit for binary classification
- **Parameters**: ~1.5M trainable parameters

## ⚙️ Optimization Features

✅ **Mixed Precision Training** (torch.amp) for faster training  
✅ **Pin Memory** for faster GPU data transfer  
✅ **Persistent Workers** to avoid dataloader respawning  
✅ **CuDNN Benchmark** enabled for A100 performance  
✅ **Large Batch Size** (128) optimized for 80GB VRAM  
✅ **Multi-worker DataLoader** (16 workers on A100 instances)  

## 📊 Training Output

The script will show:
- Epoch-by-epoch metrics (Loss, Accuracy, Precision, Recall, F1, AUC)
- GPU information and VRAM usage
- Training progress bars
- Best model saved as `best_model.pth`

## 🛠️ Troubleshooting

**Issue**: `No .npy files found`
- Run `setup_data.py` first to extract ZIP files
- Verify DATA_DIR path in `train.py`

**Issue**: `CUDA out of memory`
- Reduce BATCH_SIZE in `train.py` (try 64 or 32)
- Reduce NUM_WORKERS if too high

**Issue**: `Invalid filename format`
- Ensure files are named with `_pos_` or `_neg_`
- Check extracted files with: `ls data/*.npy | head`

## 📈 Expected Performance

On an A100 GPU with batch size 128:
- **~10-15 seconds per epoch** (depends on dataset size)
- **Mixed precision**: ~30% faster than FP32
- **VRAM usage**: ~15-20GB (out of 80GB available)

## 📝 Files Overview

- `dataset.py` - Custom PyTorch Dataset for loading .npy patches
- `model.py` - 3D CNN with Channel & Spatial Attention
- `train.py` - Training loop with validation
- `setup_data.py` - Data extraction utility
- `requirements.txt` - Python dependencies
- `best_model.pth` - Saved best model (created during training)
