import os
import shutil
import zipfile
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

# --- CONFIGURATION ---
# RESUMING: Processing from Subset 2 to 9
SUBSETS_TO_PROCESS = range(2, 10)

CANDIDATES_V2_PATH = "/kaggle/input/luna16/candidates_V2/candidates_V2.csv"
ANNOTATIONS_PATH = "/kaggle/input/luna16/annotations.csv"

# Directories
FINAL_ZIP_DIR = "/kaggle/working/zips/"
TEMP_DIR = "/kaggle/tmp/luna16/"
PROCESSED_DIR = "/kaggle/tmp/processed_patches/" 

# --- STEP 1: CLEAN SLATE ---
print("🧹 PREPARING ENVIRONMENT: Clearing temp folders...")
dirs_to_clean = ["/kaggle/working/zips/", "/kaggle/tmp/"]
for directory in dirs_to_clean:
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception:
                pass
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FINAL_ZIP_DIR, exist_ok=True)
print("✨ READY. Starting from Subset 2.")

# --- SMART URL SELECTOR ---
def get_download_url(subset_num):
    # Subsets 0-6 are in Zenodo Part 1
    if subset_num <= 6:
        return f"https://zenodo.org/record/3723295/files/subset{subset_num}.zip?download=1"
    # Subsets 7-9 are in Zenodo Part 2
    else:
        return f"https://zenodo.org/record/4121926/files/subset{subset_num}.zip?download=1"

# --- HELPER FUNCTIONS ---
def download_with_retry(subset_num, retries=5):
    filename = f"subset{subset_num}.zip"
    url = get_download_url(subset_num)
    file_path = f"{TEMP_DIR}{filename}"
    
    for attempt in range(retries):
        print(f"⬇️ Downloading {filename} (Attempt {attempt+1}/{retries})...")
        
        # Visible progress bar (-c resumes if connection drops)
        exit_code = os.system(f'wget -c --progress=bar:force:noscroll -O {file_path} "{url}"')
        
        if os.path.exists(file_path) and zipfile.is_zipfile(file_path):
            print(f"✅ {filename} downloaded and verified.")
            return True
        else:
            print(f"⚠️ Download failed or stalled. Retrying in 5s...")
            time.sleep(5)
            # Only delete if file is tiny/corrupted; keep if it's a partial download to resume
            if os.path.exists(file_path) and os.path.getsize(file_path) < 1000: 
                os.remove(file_path)
                
    raise Exception(f"❌ Failed to download {filename} after {retries} attempts.")

def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0)):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    out_size = [
        int(round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(itk_image)

def normalize(patch):
    MIN_BOUND, MAX_BOUND = -1000.0, 400.0
    patch = (patch - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    patch[patch > 1] = 1.
    patch[patch < 0] = 0.
    return patch.astype(np.float32)

# --- MAIN LOOP ---

print(f"✅ Loading CSVs...")
df_ann = pd.read_csv(ANNOTATIONS_PATH)
df_cand = pd.read_csv(CANDIDATES_V2_PATH)

for subset_num in SUBSETS_TO_PROCESS:
    subset_name = f"subset{subset_num}"
    zip_name = f"{subset_name}.zip"
    
    print(f"\n==========================================")
    print(f"🚀 STARTING {subset_name}")
    print(f"==========================================")
    
    # 1. Download
    download_with_retry(subset_num)
    
    print(f"Unzipping {subset_name}...")
    try:
        with zipfile.ZipFile(f"{TEMP_DIR}{zip_name}", 'r') as z:
            z.extractall(TEMP_DIR)
        os.remove(f"{TEMP_DIR}{zip_name}") 
    except zipfile.BadZipFile:
        print("❌ CRITICAL ERROR: Bad Zip. Skipping.")
        continue 

    # 2. Process
    subset_path = os.path.join(TEMP_DIR, subset_name)
    mhd_files = [f for f in os.listdir(subset_path) if f.endswith('.mhd')]
    
    current_output_dir = os.path.join(PROCESSED_DIR, subset_name)
    os.makedirs(current_output_dir, exist_ok=True)

    for file_name in tqdm(mhd_files, desc=f"Processing {subset_name}"):
        uid = file_name[:-4]
        mhd_path = os.path.join(subset_path, file_name)
        
        try:
            itk_img = sitk.ReadImage(mhd_path)
            itk_img = resample_image(itk_img)
            img_array = sitk.GetArrayFromImage(itk_img) 
            origin = np.array(itk_img.GetOrigin())
            spacing = np.array(itk_img.GetSpacing())
            
            # Positives
            scan_nodules = df_ann[df_ann['seriesuid'] == uid]
            for i, row in scan_nodules.iterrows():
                world_coord = np.array([row['coordX'], row['coordY'], row['coordZ']])
                v_coord = np.absolute(world_coord - origin) / spacing
                v_z, v_y, v_x = int(v_coord[2]), int(v_coord[1]), int(v_coord[0])
                
                padded = np.pad(img_array, 32, mode='constant', constant_values=-1000)
                v_z, v_y, v_x = v_z + 32, v_y + 32, v_x + 32
                patch = padded[v_z-32:v_z+32, v_y-32:v_y+32, v_x-32:v_x+32]
                np.save(f"{current_output_dir}/{uid}_pos_{i}.npy", normalize(patch))

            # Negatives
            scan_cands = df_cand[(df_cand['seriesuid'] == uid) & (df_cand['class'] == 0)].head(3)
            for i, row in scan_cands.iterrows():
                world_coord = np.array([row['coordX'], row['coordY'], row['coordZ']])
                v_coord = np.absolute(world_coord - origin) / spacing
                v_z, v_y, v_x = int(v_coord[2]), int(v_coord[1]), int(v_coord[0])
                
                padded = np.pad(img_array, 32, mode='constant', constant_values=-1000)
                v_z, v_y, v_x = v_z + 32, v_y + 32, v_x + 32
                patch = padded[v_z-32:v_z+32, v_y-32:v_y+32, v_x-32:v_x+32]
                np.save(f"{current_output_dir}/{uid}_neg_{i}.npy", normalize(patch))
                
        except Exception:
            pass

    # 3. ZIP AND CLEAN
    print(f"📦 Zipping {subset_name} results...")
    shutil.make_archive(f"{FINAL_ZIP_DIR}{subset_name}_processed", 'zip', current_output_dir)
    
    print(f"✅ SAVED: {FINAL_ZIP_DIR}{subset_name}_processed.zip")
    print(f"👉 ACTION REQUIRED: Download {subset_name}_processed.zip now!")

    shutil.rmtree(subset_path) 
    shutil.rmtree(current_output_dir)

print("\n🎉 ALL DONE! You should have 10 zip files total (0-9).")