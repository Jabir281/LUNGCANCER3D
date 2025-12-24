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

## 4. Transfer Data from Local PC (Fastest Method)

Since `wget` can struggle with OneDrive links, and you have the data locally, the fastest and most reliable method is to use **SCP (Secure Copy)**. This transfers the file directly from your computer to the RunPod instance over SSH.

### Prerequisites
1.  **Zipped Data:** Ensure your data is zipped on your local computer (e.g., `data.zip`).
2.  **SSH Connection Details:** Get your RunPod instance's **IP Address** and **Port** from the RunPod dashboard (click "Connect" -> "SSH").

### Command
Run this command **from your local computer's terminal** (PowerShell or Command Prompt):

```powershell
# Syntax: scp -P [PORT] [PATH_TO_LOCAL_FILE] root@[IP_ADDRESS]:[REMOTE_PATH]

# Example:
scp -P 12345 "C:\Users\Hp\OneDrive\Desktop\LungCancer3D\data.zip" root@192.168.1.1:/root/LUNGCANCER3D/
```

*   Replace `12345` with your Pod's Port.
*   Replace `192.168.1.1` with your Pod's IP.
*   Replace the local path with the actual location of your zip file.

### After Transfer (On RunPod)
Once the transfer finishes, go back to your RunPod terminal:

```bash
cd LUNGCANCER3D
unzip data.zip -d data
```
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
