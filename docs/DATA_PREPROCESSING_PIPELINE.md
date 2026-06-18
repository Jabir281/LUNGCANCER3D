# Data Preprocessing Pipeline (from `data_preprocess_pipline.py`)

This document explains, in execution order, everything performed by `data_preprocess_pipline.py` to convert raw LUNA16 CT volumes into training-ready 3D patch files.

## 1) High-level goal

The script builds a patch-level dataset for binary classification:

- **Positive patches** are extracted around annotated nodule coordinates from `annotations.csv`.
- **Negative patches** are extracted from candidate locations labeled as non-nodule in `candidates_V2.csv`.

Each patch is:

- A **3D cube of shape `(64, 64, 64)`**
- **HU-window normalized** into **`[0, 1]`**
- Saved as a NumPy file: **`.npy`**
- Named using the convention:
  - `{uid}_pos_{i}.npy` for positives
  - `{uid}_neg_{i}.npy` for negatives

## 2) Execution environment assumptions

This script is written specifically for a **Kaggle** environment:

- Input CSVs are expected at Kaggle dataset mount points under `/kaggle/input/...`
- It downloads the LUNA16 subsets directly from Zenodo using `wget`
- Temporary processing happens in `/kaggle/tmp/...`
- Final output zips are saved to `/kaggle/working/zips/` for download

If you run it outside Kaggle, the absolute paths must be changed.

## 3) Configuration and paths

### Subsets processed

```python
SUBSETS_TO_PROCESS = range(2, 10)
```

- Processes **subset2 → subset9**.
- This is a “resume” configuration and **does not process subset0–subset1** in its current state.

### Input CSVs

- `ANNOTATIONS_PATH = /kaggle/input/luna16/annotations.csv`
- `CANDIDATES_V2_PATH = /kaggle/input/luna16/candidates_V2/candidates_V2.csv`

### Directories

- `TEMP_DIR = /kaggle/tmp/luna16/`
  - Stores downloaded `subsetX.zip` files and extracted `.mhd/.raw` data temporarily
- `PROCESSED_DIR = /kaggle/tmp/processed_patches/`
  - Temporary directory where `.npy` patches are written per subset
- `FINAL_ZIP_DIR = /kaggle/working/zips/`
  - Final subset archives saved as `subsetX_processed.zip`

## 4) Step-by-step pipeline

### Step 4.1 — Clean slate (deletes temp output)

Before processing, the script deletes all contents of:

- `/kaggle/working/zips/`
- `/kaggle/tmp/`

Then recreates:

- `TEMP_DIR`
- `PROCESSED_DIR`
- `FINAL_ZIP_DIR`

**Implication:** This is destructive. It removes anything previously in those folders.

### Step 4.2 — Determine download URL for each subset

The script chooses a Zenodo URL based on subset number:

- subsets **0–6**: Zenodo record **3723295**
- subsets **7–9**: Zenodo record **4121926**

This is implemented in `get_download_url(subset_num)`.

### Step 4.3 — Download each subset zip with retries and verification

`download_with_retry(subset_num, retries=5)` does:

- Downloads `subsetX.zip` into `TEMP_DIR` using:
  - `wget -c` (supports resume)
- Verifies the file is a valid zip (`zipfile.is_zipfile`)
- Retries up to 5 times
- If a file is extremely small (`< 1000` bytes), it is deleted before retry

If verification fails after all retries, it raises an exception.

### Step 4.4 — Unzip the subset and delete the downloaded zip

After a successful download:

- Extracts `subsetX.zip` into `TEMP_DIR`
- Removes `subsetX.zip` immediately (saves disk space)

### Step 4.5 — Load and resample each CT scan to isotropic 1mm spacing

For every `.mhd` file in the extracted subset folder:

1. Read the image via SimpleITK:
   - `itk_img = sitk.ReadImage(mhd_path)`
2. Resample to isotropic spacing:
   - `itk_img = resample_image(itk_img, out_spacing=(1,1,1))`
3. Convert to numpy array:
   - `img_array = sitk.GetArrayFromImage(itk_img)`

Resampling details:

- Output spacing: **(1.0, 1.0, 1.0)**
- Interpolation: **linear** (`sitk.sitkLinear`)
- Output size computed from original size and spacing

### Step 4.6 — Convert world coordinates → voxel coordinates

For both annotations and candidates, the script converts each point `(coordX, coordY, coordZ)` from world coordinates into voxel indices using:

```python
v_coord = np.absolute(world_coord - origin) / spacing
v_z, v_y, v_x = int(v_coord[2]), int(v_coord[1]), int(v_coord[0])
```

Notes:

- `origin` and `spacing` are taken from the **resampled** image.
- The mapping uses the array order returned by `GetArrayFromImage`, which is typically **(z, y, x)**.
- The use of `np.absolute(...)` is unusual (it forces non-negative offsets); in many pipelines you would typically use `(world - origin) / spacing` without absolute value. This may affect correctness if coordinates are “below” origin.

### Step 4.7 — Pad the volume and extract a 64×64×64 patch

To safely crop around borders:

- The full 3D volume array is padded by 32 voxels in each dimension:
  - `np.pad(img_array, 32, constant_values=-1000)`
- The center voxel is shifted by +32 in each dimension
- Patch extraction:

```python
patch = padded[v_z-32:v_z+32, v_y-32:v_y+32, v_x-32:v_x+32]
```

This yields a patch of size:

- Depth: 64
- Height: 64
- Width: 64

Padding constant is **-1000 HU**, representing air.

### Step 4.8 — Normalize HU to [0, 1]

Each patch is normalized using:

- HU clip window: **[-1000, 400]**
- Then scaled to [0,1]
- Values outside [0,1] clipped
- Final dtype: `float32`

```python
patch = (patch - (-1000)) / (400 - (-1000))
patch = clip(patch, 0, 1)
patch = patch.astype(np.float32)
```

### Step 4.9 — Save positive patches

Positive patches are defined by entries in `annotations.csv`.

For a scan `uid`:

- Select rows: `df_ann[df_ann['seriesuid'] == uid]`
- For each matching row, extract a patch and save:

- Output file name:
  - `{uid}_pos_{i}.npy`

Where `{i}` is the **dataframe row index** from `df_ann.iterrows()`.

### Step 4.10 — Save negative patches

Negative patches are taken from `candidates_V2.csv` where:

- `seriesuid == uid`
- `class == 0` (non-nodule)

Then the script selects only the first 3 rows:

```python
scan_cands = df_cand[(df_cand['seriesuid'] == uid) & (df_cand['class'] == 0)].head(3)
```

For each of those candidates it extracts and saves:

- Output file name:
  - `{uid}_neg_{i}.npy`

Where `{i}` is the **dataframe row index** from `df_cand.iterrows()`.

**Important:** This creates at most **3 negatives per scan**, which may or may not match the desired class balance.

### Step 4.11 — Error handling behavior

The entire scan processing block is wrapped in:

```python
try:
    ...
except Exception:
    pass
```

That means:

- Any error during reading, resampling, coordinate conversion, patch extraction, or saving
  is silently ignored.
- No message is printed for failures.

This is storage/throughput-friendly, but it makes debugging harder and can reduce dataset completeness without warning.

### Step 4.12 — Zip results per subset and delete intermediate folders

After all `.mhd` scans in the subset are processed:

- Creates a zip archive:
  - `subsetX_processed.zip`
- Saved to:
  - `/kaggle/working/zips/`

Then deletes:

- The extracted raw subset directory
- The processed patch directory for that subset

This keeps disk usage low and supports the intended “process subset → export zip → clean up” workflow.

## 5) Final outputs (what you get)

For each subset processed, the script produces:

- `/kaggle/working/zips/subsetX_processed.zip`

Inside each zip:

- `.npy` patch files, each with shape `(64, 64, 64)` and dtype `float32`
- Filenames encode class:
  - `_pos_` for positive (label 1)
  - `_neg_` for negative (label 0)

## 6) Practical caveats and improvement opportunities

- **Patch-level split risk:** if multiple patches per scan exist, you should split by patient/scan to avoid leakage during training.
- **Silent failures:** the broad `except: pass` can hide issues; logging errors (at least per UID) is recommended.
- **Negative sampling:** `head(3)` restricts negatives per scan; consider sampling more negatives or controlling class ratio explicitly.
- **World→voxel conversion:** `abs(world - origin)` is unusual; verify coordinate conversion correctness against a reference pipeline.

---

If you want, I can also update Chapter 4 to describe this preprocessing exactly (including resampling to 1mm, HU clipping [-1000, 400], padding strategy, and the “3 negatives per scan” detail), so your thesis stays consistent with the script.
