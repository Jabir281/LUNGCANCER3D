# Project Debugging & Challenges Report

## 1. Environment & Setup Challenges (RunPod)

### A. Data Download Issues
*   **Problem:** The initial use of `wget` was unreliable for downloading large datasets from Google Drive/OneDrive. Use of `gdown` failed initially because the command was not in the system PATH.
*   **Fix:** 
    1.  Switched to using `gdown`.
    2.  Implemented the command as `python -m gdown <id>` to ensure execution through the active Python interpreter.
    3.  Added explicit `pip install gdown` step instructions.

### B. Missing System Tools
*   **Problem:** The minimal RunPod A100 instance lacked basic utilities like `unzip`.
*   **Fix:** Added the system command `apt-get update && apt-get install -y unzip` to the setup instructions.

## 2. Data Management & Structure

### A. Nested Zip Archives
*   **Problem:** The dataset (`luna16_processed.zip`) contained multiple inner zip files (`subset0.zip`, etc.) rather than loose files. Extracting just the main zip left the data compressed.
*   **Fix:** Wrote a shell loop in the instructions to iterate through and extract all nested zip files automatically:
    ```bash
    for f in *.zip; do unzip "$f" -d "${f%.zip}"; done
    ```

### B. Storage & Cost Optimization
*   **Problem:** Unzipping created duplicate data (zips + extracted files), risking disk space limits on the cloud instance.
*   **Fix:** Added a **"Cleanup"** step to immediately `rm` (remove) zip files after extraction.

## 3. Code Deprecations & Warnings

### A. PyTorch Mixed Precision Warning
*   **Problem:** The training script triggered a `FutureWarning`: `torch.cuda.amp.autocast(args...) is deprecated`.
*   **Fix:** Updated `train.py` to use the modern syntax `torch.amp.autocast('cuda', args...)` to ensure future compatibility.

## 4. Visualization & Local Inference Challenges

### A. Kernel "Hanging" / Freezing
*   **Problem:** The `visualize_prediction.ipynb` notebook would hang indefinitely when initializing.
*   **Debug:** Suspected conflict between PyTorch CUDA initialization and other imports, or `model.py` running code on import.
*   **Fix:** Split the import cells. Imported `torch` first to verify device inputs, then wrapped the `model` import in a `try-except` block to catch extraction errors.

### B. Missing Dependencies
*   **Problem:** Local notebook showed "yellow lines" (warnings) and missing packages like `matplotlib`.
*   **Fix:** 
    1.  Updated `requirements.txt`.
    2.  Added a magic command cell `!pip install -r requirements.txt` directly inside the notebook to ensure packages were installed in the correct Jupyter kernel.

### C. File Path Complexity
*   **Problem:** The visualization script originally required manual entry of file paths (`sample_files = [...]`), which was error-prone and tedious.
*   **Fix:** Updated the script to interact directly with the `luna16_processed.zip` file. It now automatically searches the zip structure (including nested zips) to find Positive and Negative samples without requiring manual extraction.

## 5. Overfitting & Model Performance

### The Overfitting Issue
**Observation:**
Deep learning models often memorize training data after many epochs. For example, at Epoch 45, Training Accuracy might be 95% while Validation Accuracy drops to 80%. This is **Overfitting**.

**Our Solution (Implemented in `train.py`):**
We addressed this proactively using **Model Checkpointing**:
*   Instead of just saving the model at the very end (Epoch 50), the script monitors **Validation Accuracy** after every epoch.
*   It only saves to `best_model.pth` if the current epoch's validation score is higher than the previous best.
*   **Result:** Even if the model overfits at Epoch 50, your `best_model.pth` file retains the "peak performance" version (e.g., from Epoch 10 or 20), ensuring the best possible generalization on new patients.
