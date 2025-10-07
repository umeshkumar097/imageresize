# --- Aiclex Media Processor ---
# Author: Aiclex Technologies (aiclex.in)
# Version: 1.1.0
# ----------------------------

import os
import sys
import zipfile
import shutil
import argparse
import csv
import tempfile
from pathlib import Path
from PIL import Image, ImageOps

# Third-party libraries (install from requirements.txt)
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# --- CONFIGURATION ---

TARGET_SIZES_KB = {
    "id_proof": (10, 24),
    "photo": (10, 24),
    "signature": (10, 24),
    "qualification_proof": (50, 200),
    "unknown": (10, 500) # Fallback for uncategorized files
}

CATEGORY_KEYWORDS = {
    "id_proof": ["id", "aadhaar", "pan", "passport", "dl", "driving"],
    "photo": ["photo", "pic", "profile", "image"],
    "signature": ["sign", "signature"],
    "qualification_proof": ["degree", "certificate", "marksheet", "qualification"]
}

SUPPORTED_MEDIA_EXT = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf']
MIN_QUALITY = 10
DPI = 200

# --- CORE FUNCTIONS ---

def extract_all_zips(start_dir):
    """Recursively finds and extracts all nested ZIP files."""
    while True:
        zip_found_in_scan = False
        for root, _, files in os.walk(start_dir):
            for filename in files:
                if filename.lower().endswith('.zip'):
                    zip_path = Path(root) / filename
                    # Extract to a folder with the same name as the zip
                    extract_folder = zip_path.with_suffix('')
                    print(f"  -> Found nested ZIP, extracting: {zip_path.relative_to(start_dir)}")
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_folder)
                        # Delete the inner zip file after successful extraction
                        zip_path.unlink()
                        zip_found_in_scan = True
                    except zipfile.BadZipFile:
                        print(f"ERROR: Bad ZIP file, cannot extract: {zip_path}")
                    # Break to restart the walk from the top
                    break
            if zip_found_in_scan:
                break
        # If no zip was found in a full scan, we are done
        if not zip_found_in_scan:
            break

def guess_category_by_filename(filename):
    """Guesses document category based on keywords in the filename."""
    fn_lower = Path(filename).stem.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in fn_lower for keyword in keywords):
            return category
    return "unknown"

def convert_any_to_jpg(input_path, output_jpg_path):
    """Converts the first page of a PDF or any image to a JPG file."""
    try:
        ext = input_path.suffix.lower()
        output_jpg_path.parent.mkdir(parents=True, exist_ok=True)

        if ext == '.pdf':
            if not PDF_SUPPORT:
                print(f"WARN: 'pdf2image' not found. Skipping PDF: {input_path.name}")
                return False
            try:
                pages = convert_from_path(input_path, dpi=DPI, first_page=1, last_page=1)
                if pages:
                    pages[0].save(output_jpg_path, 'JPEG')
                    return True
            except Exception as e:
                print(f"ERROR: Could not convert PDF '{input_path.name}': {e}")
                return False
        elif ext in SUPPORTED_MEDIA_EXT:
            with Image.open(input_path) as img:
                img = ImageOps.exif_transpose(img) # Fix orientation
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output_jpg_path, 'JPEG')
            return True
        else:
            return False
    except Exception as e:
        print(f"ERROR: Failed to process '{input_path.name}': {e}")
        return False

def compress_jpg_to_target(input_path, output_path, min_kb, max_kb):
    """Iteratively compresses a JPG to fit within a target KB range."""
    try:
        img = Image.open(input_path)
        img_copy = img.copy()
        
        for quality in range(95, MIN_QUALITY - 1, -5):
            img_copy.save(output_path, 'JPEG', quality=quality, optimize=True)
            size_kb = output_path.stat().st_size / 1024
            if min_kb <= size_kb <= max_kb:
                return "ok", size_kb

        best_attempt_path = Path(str(output_path) + ".best")
        shutil.copy(output_path, best_attempt_path)

        for scale in [0.9, 0.75, 0.5]:
            new_size = (int(img_copy.width * scale), int(img_copy.height * scale))
            img_resized = img_copy.resize(new_size, Image.Resampling.LANCZOS)
            for quality in range(90, MIN_QUALITY - 1, -10):
                img_resized.save(output_path, 'JPEG', quality=quality, optimize=True)
                size_kb = output_path.stat().st_size / 1024
                if min_kb <= size_kb <= max_kb:
                    best_attempt_path.unlink()
                    return "ok", size_kb
                if size_kb < best_attempt_path.stat().st_size / 1024:
                     shutil.copy(output_path, best_attempt_path)

        shutil.move(best_attempt_path, output_path)
        final_size_kb = output_path.stat().st_size / 1024
        return "partial", final_size_kb

    except Exception as e:
        print(f"ERROR: Compression failed for '{input_path.name}': {e}")
        return "compress_failed", 0

def process_tree(source_dir, processed_dir):
    """Processes all files in a directory tree, returning a report."""
    report_data = []
    print("\n--- Starting Processing ---")
    
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith('.zip'): continue # Skip any zips that failed extraction

            input_path = Path(root) / filename
            relative_path = input_path.relative_to(source_dir)
            print(f"Processing: {relative_path}")
            
            original_kb = input_path.stat().st_size / 1024
            ext = input_path.suffix.lower()

            # If it's a media file, process it. Otherwise, copy as-is.
            if ext in SUPPORTED_MEDIA_EXT:
                temp_jpg_path = Path(tempfile.gettempdir()) / f"temp_{relative_path.name}.jpg"

                if not convert_any_to_jpg(input_path, temp_jpg_path):
                    report_data.append([relative_path, 'unknown', f"{original_kb:.2f}", 0, "convert_failed", "name"])
                    continue
                
                category = guess_category_by_filename(filename)
                min_kb, max_kb = TARGET_SIZES_KB.get(category, TARGET_SIZES_KB["unknown"])
                
                output_path = (processed_dir / relative_path).with_suffix(".jpg")
                output_path.parent.mkdir(parents=True, exist_ok=True)

                status, final_kb = compress_jpg_to_target(temp_jpg_path, output_path, min_kb, max_kb)
                report_data.append([relative_path, category, f"{original_kb:.2f}", f"{final_kb:.2f}", status, "name"])
                
                if temp_jpg_path.exists(): temp_jpg_path.unlink()
            else:
                # For non-media files (CSV, Excel, etc.), copy them directly
                output_path = processed_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(input_path, output_path)
                report_data.append([relative_path, 'document', f"{original_kb:.2f}", f"{original_kb:.2f}", 'copied_as_is', 'name'])

    print("--- Processing Complete ---\n")
    return report_data

def main():
    """Main function to handle CLI arguments and orchestrate the workflow."""
    print("\n--- Aiclex Media Processor (aiclex.in) ---")
    
    parser = argparse.ArgumentParser(
        description="Aiclex ZIP to JPG Processor - Compresses and standardizes media files from a ZIP archive.",
        epilog="© 2025 Aiclex Technologies | aiclex.in"
    )
    parser.add_argument("input_path", type=str, help="Path to the input ZIP file.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.is_file() or input_path.suffix.lower() != '.zip':
        print(f"Error: Input path must be a .zip file. You provided: {input_path}")
        sys.exit(1)

    base_name = input_path.stem
    output_zip_path = Path.cwd() / f"processed_{base_name}.zip"
    output_report_path = Path.cwd() / f"processing_report_{base_name}.csv"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        source_files_dir = temp_path / "source"
        processed_files_dir = temp_path / "processed"
        source_files_dir.mkdir()
        processed_files_dir.mkdir()

        # --- 1. Extract main ZIP and all nested ZIPs ---
        print(f"Extracting main ZIP: {input_path.name}")
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(source_files_dir)
        
        extract_all_zips(source_files_dir)
            
        # --- 2. Process all extracted files ---
        report_rows = process_tree(source_files_dir, processed_files_dir)

        # --- 3. Generate CSV Report ---
        report_headers = ["path", "category", "original_kb", "final_kb", "status", "decided_via"]
        with open(output_report_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(report_headers)
            writer.writerows(report_rows)
        print(f"Generated report: {output_report_path}")

        # --- 4. Create final output ZIP ---
        shutil.make_archive(output_zip_path.with_suffix(''), 'zip', processed_files_dir)
        print(f"Created processed ZIP: {output_zip_path}")
        
    print("\n✅ All tasks completed successfully.")
    if not PDF_SUPPORT:
        print("\nNOTE: PDF processing was skipped. To enable it, install 'poppler-utils' on your system and the 'pdf2image' Python library.")

if __name__ == "__main__":
    main()
