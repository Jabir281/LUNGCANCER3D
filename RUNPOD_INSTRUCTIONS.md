# RunPod A100 Setup and Training Instructions

This guide will help you set up your environment on a RunPod A100 instance, download your data, and run the training script.

## 1. Connect to your RunPod Instance

You can use the Web Terminal provided by RunPod or SSH into your instance.

## 2. Clone the Repository

First, clone your GitHub repository to the instance:

```bash
git clone https://github.com/Jabir281/LUNGCANCER3D.git
cd LUNGCANCER3D
```

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 4. Download and Prepare Data

You need to download the dataset from the OneDrive link you provided. We will use `wget` to download it.

### Option A: Direct Download (if link is direct)

```bash
# Create a directory for the data
mkdir data

# Download the zip file (using the link you provided)
# Note: If this link expires or is dynamic, you might need to generate a new one.
wget -O data.zip "https://southeastasia1-mediap.svc.ms/transform/zip?cs=fFNQTw"

# Unzip the data into the 'data' folder
unzip data.zip -d data
```

### Option B: If the link doesn't work with wget directly

If the link above fails or downloads an HTML file, you might need to:
1. Download the file to your local machine.
2. Upload it to RunPod using `scp` or the JupyterLab upload interface.
3. Or use a tool like `gdown` if it's a Google Drive link (but this is OneDrive).

**Verifying Data Structure:**
Ensure your `data` folder contains the `.npy` files. The script expects the structure to be something like:
```
LUNGCANCER3D/
  data/
    subset0/
      ...npy files...
    subset1/
      ...npy files...
```
Or just flat `.npy` files inside `data/`. The `dataset.py` script searches recursively for `.npy` files, so subfolders are fine.

## 5. Verify Configuration

The `train.py` script has been configured with:
- `BATCH_SIZE = 128` (Optimized for A100)
- `NUM_WORKERS = 7` (As requested)
- `DATA_DIR = "./data"`

You can verify the GPU availability:

```bash
nvidia-smi
```

## 6. Run Training

Start the training script:

```bash
python train.py
```

## 7. Monitoring

- The script uses `tqdm` for progress bars.
- It will save the best model to `best_model.pth`.
- You can monitor GPU usage in a separate terminal using `watch -n 1 nvidia-smi`.

## Troubleshooting

- **Out of Memory (OOM):** If you encounter OOM errors, reduce `BATCH_SIZE` in `train.py`.
- **Data Not Found:** Ensure `DATA_DIR` in `train.py` points to the correct location where you unzipped the data.
- **Permission Denied:** If you have trouble running scripts, try `chmod +x train.py`.
