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

## 4. Download Data from Google Drive

We will use `gdown` to download the dataset directly from Google Drive.

1.  **Ensure gdown is installed:**
    ```bash
    pip install gdown
    ```

2.  **Download the file:**
    ```bash
    # Try running gdown directly
    gdown 135ncVlpPbWPZZKWpWAtCivHEJVShrBo1 -O data.zip
    ```
    *If you get "command not found", try using python3:*
    ```bash
    python3 -m gdown 135ncVlpPbWPZZKWpWAtCivHEJVShrBo1 -O data.zip
    ```

3.  **Unzip the data:**
    If `unzip` is not installed, install it first:
    ```bash
    apt-get update && apt-get install -y unzip
    ```
    
    **Step A: Unzip Main Archive** (Skip if you already did this and deleted `data.zip`)
    ```bash
    unzip data.zip -d data
    ```

    **Step B: Unzip Nested Archives**
    Navigate to the folder:
    ```bash
    cd data/lung_cancer_processed_dataset
    ```
    
    **Option 1: If you have `.zip` files here:**
    Run this to extract them into organized folders:
    ```bash
    for f in *.zip; do
        unzip "$f" -d "${f%.zip}"
    done
    ```
    
    **Option 2: If you already extracted them (files are present but flat):**
    You don't need to do anything. The training script (`dataset.py`) searches recursively, so it will find the `.npy` files even if they are all in one folder.

    Return to the main directory:
    ```bash
    cd ../..
    ```

### Expected Structure
The data will be in `data/lung_cancer_processed_dataset`. 
*   **Organized:** You might see folders like `subset0_processed/`, `subset1_processed/`, etc.
*   **Flat:** Or you might see a long list of `.npy` files directly in this folder.

**Both are fine.** The `dataset.py` script uses recursive search, so it will automatically find all `.npy` files regardless of the folder structure.

4.  **Clean Up Zip Files (Save Space):**
    Once you have verified the folders are extracted, delete the zip files to free up disk space:
    ```bash
    # Remove the main zip file
    rm data.zip

    # Remove the nested zip files
    rm data/lung_cancer_processed_dataset/*.zip
    ```

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

## 8. After Training (IMPORTANT)

Once training reaches 50/50 epochs, follow these steps immediately to save your work and stop paying:

1.  **Save Training Logs:**
    *   The training script prints accuracy and loss to the terminal but does not save a file.
    *   **Action:** Select all text in your RunPod terminal, copy it, and paste it into a text file (e.g., `training_logs.txt`) on your local computer. You will need this to plot your accuracy curves later.

2.  **Download the Model:**
    You need to get `best_model.pth` off the RunPod instance.
    *   **Option A (RunPod Jupyter):** Right-click `best_model.pth` -> Download.
    *   **Option B (SCP):**
        ```powershell
        scp -P [PORT] root@[IP]:/root/LUNGCANCER3D/best_model.pth .
        ```

3.  **Download Sample Data (For Visualization):**
    To visualize predictions locally, you need a few sample files. You don't need the whole dataset.
    *   **Action:** Go to `data/lung_cancer_processed_dataset/subset0_processed/` (or similar) in the Jupyter file browser.
    *   Download **one positive sample** (filename contains `_pos_`) and **one negative sample** (filename contains `_neg_`).
    *   Save them to a `samples` folder on your local PC.

4.  **Terminate the Pod:**
    *   Go to the RunPod Dashboard.
    *   Find your pod.
    *   Click **Terminate** (to delete everything and stop billing).
    *   **Recommendation:** Do this immediately after downloading your files.
