#!/usr/bin/env python3
"""
compress_media.py

Simple tool:
- Input: path to a ZIP file OR path to a folder OR single file
- Output: processed ZIP where every file is converted to JPG and compressed to target KB ranges
- CSV report of results created alongside output ZIP

Usage:
    python compress_media.py /path/to/input.zip
    python compress_media.py /path/to/folder
    python compress_media.py /path/to/file.pdf

Notes:
- PDF -> JPG conversion uses pdf2image (requires poppler/pdftoppm installed on system).
  If poppler is not present, PDFs will be skipped (copied into result as-is).
- This script uses only filename-based category detection (fast, stable).
"""

import sys
import os
import zipfile
import shutil
import tempfile
import pathlib
import io
from PIL import Image, ImageOps
import csv
from pdf2image import convert_from_path, exceptions as pdf2img_exceptions

# ---------- Configuration ----------
TARGET_SIZES_KB = {
    'id_proof': (10, 24),
    'photo': (10, 24),
    'signature': (10, 24),
    'qualification': (50, 200),
}
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.tiff'}
PDF_EXT = {'.pdf'}
ZIP_EXT = {'.zip'}
MAX_DIMENSION = 3500  # px
MIN_QUALITY = 10

# ---------- Helpers ----------
def sanitize_name(name: str) -> str:
    return name.replace('..', '').replace('/', '_')

def guess_category_by_filename(filename: str):
    name = filename.lower()
    if 'id' in name and ('proof' in name or 'id' in name):
        return 'id_proof'
    if any(k in name for k in ['photo','passport','profile']):
        return 'photo'
    if any(k in name for k in ['sign','signature','sig']):
        return 'signature'
    if any(k in name for k in ['qual','qualification','degree','certificate','marksheet']):
        return 'qualification'
    return None

def is_zip_file(path: str):
    return pathlib.Path(path).suffix.lower() == '.zip'

def recursive_extract_zip(zip_path: str, dest_dir: str):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)
    # extract nested zips one level at a time
    for root, _, files in os.walk(dest_dir):
        for f in files:
            full = os.path.join(root, f)
            if is_zip_file(full):
                nested_dir = os.path.splitext(full)[0]
                try:
                    with zipfile.ZipFile(full, 'r') as nz:
                        nz.extractall(nested_dir)
                    os.remove(full)
                except Exception:
                    # skip corrupted nested zips
                    continue

# Convert any file to a single-page JPG
def convert_any_to_jpg(input_path: str, output_path: str, dpi=200):
    ext = pathlib.Path(input_path).suffix.lower()
    try:
        if ext in PDF_EXT:
            try:
                pages = convert_from_path(input_path, dpi=dpi, first_page=1, last_page=1)
                if pages:
                    img = pages[0]
                    rgb = img.convert('RGB')
                    rgb.save(output_path, 'JPEG')
                    return True
                return False
            except pdf2img_exceptions.PDFInfoNotInstalledError:
                # poppler not installed
                return False
            except Exception:
                return False
        else:
            img = Image.open(input_path)
            img = ImageOps.exif_transpose(img)
            rgb = img.convert('RGB')
            rgb.save(output_path, 'JPEG')
            return True
    except Exception:
        return False

# compress iterative
def compress_jpg_to_target(tmp_input_path: str, tmp_output_path: str, min_kb: int, max_kb: int):
    try:
        img = Image.open(tmp_input_path)
        img = ImageOps.exif_transpose(img)
        w,h = img.size
        if max(w,h) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(w,h)
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

        best_data = None
        # try reducing quality first
        for q in range(95, MIN_QUALITY-1, -5):
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=q, optimize=True)
            data = buf.getvalue()
            size_kb = len(data) // 1024
            if min_kb <= size_kb <= max_kb:
                with open(tmp_output_path, 'wb') as f:
                    f.write(data)
                return True
            if size_kb <= max_kb:
                best_data = data
        # fallback: downscale gradually with reasonable quality
        if best_data:
            with open(tmp_output_path, 'wb') as f:
                f.write(best_data)
            return True

        scale = 0.9
        attempts = 0
        while attempts < 8:
            attempts += 1
            nw = int(img.width * scale)
            nh = int(img.height * scale)
            if nw < 60 or nh < 60:
                break
            img_small = img.resize((nw, nh), Image.LANCZOS)
            for q in range(85, MIN_QUALITY-1, -5):
                buf = io.BytesIO()
                img_small.save(buf, format='JPEG', quality=q, optimize=True)
                data = buf.getvalue()
                size_kb = len(data) // 1024
                if min_kb <= size_kb <= max_kb:
                    with open(tmp_output_path, 'wb') as f:
                        f.write(data)
                    return True
                if size_kb <= max_kb:
                    best_data = data
            scale -= 0.1
        # final fallback: write best_data if exists
        if best_data:
            with open(tmp_output_path,'wb') as f:
                f.write(best_data)
            return True
        # last resort: very low quality save
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=MIN_QUALITY, optimize=True)
        with open(tmp_output_path,'wb') as f:
            f.write(buf.getvalue())
        return False
    except Exception:
        try:
            shutil.copy2(tmp_input_path, tmp_output_path)
        except Exception:
            pass
        return False

# Process extracted directory tree in-place; returns list of report rows
def process_extracted_tree(root_dir: str):
    report = []
    processed = 0
    changed = 0
    for cur_root, _, files in os.walk(root_dir):
        for fname in files:
            full = os.path.join(cur_root, fname)
            rel = os.path.relpath(full, root_dir)
            ext = pathlib.Path(full).suffix.lower()
            orig_kb = os.path.getsize(full) // 1024
            category = guess_category_by_filename(fname)
            decided_via = 'name' if category else 'name-fallback'

            if category is None:
                # unknown -> still convert to JPG (but no forced size)
                out_rel = pathlib.Path(rel).with_suffix('.jpg')
                out_full = os.path.join(root_dir, out_rel)
                os.makedirs(os.path.dirname(out_full), exist_ok=True)
                converted = convert_any_to_jpg(full, out_full)
                final_kb = os.path.getsize(out_full) // 1024 if converted else orig_kb
                status = 'converted' if converted else 'skipped'
                report.append([rel, 'unknown', orig_kb, final_kb, status, decided_via])
                processed += 1
                continue

            # For known category -> enforce size range
            min_kb, max_kb = TARGET_SIZES_KB.get(category, (10,200))
            tmp_in = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False); tmp_in.close()
            tmp_out = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False); tmp_out.close()
            converted = convert_any_to_jpg(full, tmp_in.name)
            if not converted:
                # cannot convert (eg PDF and poppler missing)
                report.append([rel, category, orig_kb, orig_kb, 'convert_failed', decided_via])
                processed += 1
                try:
                    os.remove(tmp_in.name); os.remove(tmp_out.name)
                except Exception:
                    pass
                continue
            success = compress_jpg_to_target(tmp_in.name, tmp_out.name, min_kb, max_kb)
            # move final into place (replace extension with .jpg)
            out_rel = pathlib.Path(rel).with_suffix('.jpg')
            out_full = os.path.join(root_dir, out_rel)
            os.makedirs(os.path.dirname(out_full), exist_ok=True)
            try:
                shutil.move(tmp_out.name, out_full)
            except Exception:
                try: shutil.copy2(tmp_out.name, out_full)
                except Exception: pass
            # remove original if different
            try:
                if os.path.abspath(full) != os.path.abspath(out_full):
                    os.remove(full)
            except Exception:
                pass
            final_kb = os.path.getsize(out_full) // 1024 if os.path.exists(out_full) else orig_kb
            status = 'ok' if success and min_kb <= final_kb <= max_kb else 'partial'
            report.append([rel, category, orig_kb, final_kb, status, decided_via])
            processed += 1
            if status in ('ok','partial'):
                changed += 1
            try:
                if os.path.exists(tmp_in.name):
                    os.remove(tmp_in.name)
            except Exception:
                pass
    return report, processed, changed

# ---------- Main CLI ----------
def main(input_path):
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        print("Input not found:", input_path)
        return 1

    with tempfile.TemporaryDirectory() as workdir:
        root_extract = os.path.join(workdir, 'extracted')
        os.makedirs(root_extract, exist_ok=True)

        # If ZIP -> extract recursively
        if is_zip_file(input_path):
            print("Extracting ZIP...")
            recursive_extract_zip(input_path, root_extract)
        elif os.path.isdir(input_path):
            # copy files into extracted dir keeping structure
            print("Copying folder into workspace...")
            base = os.path.basename(os.path.abspath(input_path))
            shutil.copytree(input_path, os.path.join(root_extract, base))
        else:
            # single file
            shutil.copy2(input_path, os.path.join(root_extract, os.path.basename(input_path)))

        print("Processing files (converting + compressing)...")
        report_rows, processed, changed = process_extracted_tree(root_extract)
        print(f"Processed: {processed} files. Changed: {changed}")

        # write CSV report
        report_csv = os.path.join(workdir, 'processing_report.csv')
        with open(report_csv, 'w', newline='', encoding='utf-8') as rf:
            writer = csv.writer(rf)
            writer.writerow(['path','category','original_kb','final_kb','status','decided_via'])
            for r in report_rows:
                writer.writerow(r)

        # create zip of processed folder
        out_zip = os.path.join(os.getcwd(), f'processed_{os.path.basename(input_path)}.zip')
        # if out_zip exists, delete first
        if os.path.exists(out_zip):
            os.remove(out_zip)
        shutil.make_archive(out_zip.replace('.zip',''), 'zip', root_extract)
        # copy report next to zip
        shutil.copy2(report_csv, os.path.join(os.getcwd(), 'processing_report.csv'))

        print("Done.")
        print("Output ZIP:", out_zip)
        print("Report CSV:", os.path.join(os.getcwd(), 'processing_report.csv'))
        return 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compress_media.py <input.zip | input_folder | input_file>")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
