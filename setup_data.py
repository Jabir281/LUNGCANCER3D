"""
Script to extract and organize LUNA16 data on RunPod.
Run this FIRST before training to unzip all subset files.
"""

import os
import zipfile
from pathlib import Path
from tqdm import tqdm

def extract_luna16_data(data_root="./luna16_processed", output_dir="./data"):
    """
    Extract all subset ZIP files from luna16_processed folder.
    
    Args:
        data_root: Path to luna16_processed folder containing subset0.zip - subset9.zip
        output_dir: Where to extract all .npy files
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find all zip files
    zip_files = list(data_root.glob("subset*.zip"))
    
    if len(zip_files) == 0:
        print(f"No subset*.zip files found in {data_root}")
        print("Please ensure you have downloaded and placed the data correctly.")
        return
    
    print(f"Found {len(zip_files)} subset ZIP files")
    
    # Extract each ZIP file
    for zip_path in tqdm(sorted(zip_files), desc="Extracting subsets"):
        print(f"\nExtracting {zip_path.name}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract all files
                zip_ref.extractall(output_dir)
                
            print(f"✓ Extracted {zip_path.name}")
            
        except Exception as e:
            print(f"✗ Error extracting {zip_path.name}: {e}")
    
    # Count total .npy files
    npy_files = list(output_dir.rglob("*.npy"))
    pos_files = [f for f in npy_files if "_pos_" in f.stem]
    neg_files = [f for f in npy_files if "_neg_" in f.stem]
    
    print("\n" + "="*60)
    print("Extraction complete!")
    print(f"Total .npy files: {len(npy_files)}")
    print(f"Positive samples (_pos_): {len(pos_files)}")
    print(f"Negative samples (_neg_): {len(neg_files)}")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*60)
    
    # Update train.py suggestion
    print(f"\n📝 Update train.py: Set DATA_DIR = '{output_dir}'")

if __name__ == "__main__":
    # Default paths - modify these if your structure is different
    extract_luna16_data(
        data_root="./luna16_processed",  # Folder with subset*.zip files
        output_dir="./data"              # Where to extract .npy files
    )
