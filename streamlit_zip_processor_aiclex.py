"""
Streamlit App: Aiclex Technologies - ZIP to JPG Processor
Fix: Removed recursion in guess_category_by_name to avoid RecursionError.
"""

import streamlit as st
import zipfile
import os
import shutil
import tempfile
import io
from PIL import Image, ImageOps
import pathlib
import pandas as pd
from pdf2image import convert_from_path, exceptions as pdf2img_exceptions
import pytesseract
import time

# ----------------------- Configuration -----------------------
TARGET_SIZES_KB = {
    'id_proof': (10, 24),
    'photo': (10, 24),
    'signature': (10, 24),
    'qualification': (50, 200),
}

IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.tiff'}
PDF_EXT = {'.pdf'}
ZIP_EXT = {'.zip'}

MAX_DIMENSION = 3500
MIN_QUALITY = 10

# ----------------------- Helpers -----------------------
def sanitize_name(name: str) -> str:
    return name.replace('..', '').replace('/', '_')

def guess_category_by_name(filename: str):
    name = filename.lower()
    if 'id' in name and ('proof' in name or 'id' in name):
        return 'id_proof'
    if 'photo' in name or 'passport' in name or 'profile' in name:
        return 'photo'
    if 'sign' in name or 'signature' in name or 'sig' in name:
        return 'signature'
    if 'qual' in name or 'degree' in name or 'certificate' in name or 'marksheet' in name:
        return 'qualification'
    # check last token only once (no recursion)
    tokens = [t for t in pathlib.Path(name).stem.replace('_',' ').split() if t]
    if tokens:
        last = tokens[-1]
        if last in ['id','idproof','identity']:
            return 'id_proof'
        if last in ['photo','passport','profile']:
            return 'photo'
        if last in ['sign','signature','sig']:
            return 'signature'
        if last in ['qual','qualification','degree','certificate','marksheet']:
            return 'qualification'
    return None

def ocr_guess_category(image_path: str):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='eng').lower()
        if any(k in text for k in ['id','identity','aadhar','pan','passport']):
            return 'id_proof'
        if any(k in text for k in ['photo','passport','profile']):
            return 'photo'
        if any(k in text for k in ['signature','signed']):
            return 'signature'
        if any(k in text for k in ['degree','certificate','marksheet','qualification']):
            return 'qualification'
    except Exception:
        return None
    return None

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def is_zip_file(path: str):
    return pathlib.Path(path).suffix.lower() == '.zip'

def recursive_extract_zip(zip_path: str, dest_dir: str):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)
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
                    continue

# ----------------------- Conversion & Compression -----------------------
def convert_any_to_jpg(input_path: str, output_path: str, dpi=200):
    ext = pathlib.Path(input_path).suffix.lower()
    try:
        if ext in PDF_EXT:
            pages = convert_from_path(input_path, dpi=dpi, first_page=1, last_page=1)
            if pages:
                img = pages[0]
                rgb = img.convert('RGB')
                rgb.save(output_path, 'JPEG')
                return True
            return False
        else:
            img = Image.open(input_path)
            img = ImageOps.exif_transpose(img)
            rgb = img.convert('RGB')
            rgb.save(output_path, 'JPEG')
            return True
    except Exception:
        return False

def compress_jpg_to_target(tmp_input_path: str, tmp_output_path: str, min_kb: int, max_kb: int):
    try:
        img = Image.open(tmp_input_path)
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        if max(w, h) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(w, h)
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

        best_data, best_size = None, None
        for q in range(95, MIN_QUALITY - 1, -5):
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=q, optimize=True)
            data = buf.getvalue()
            size_kb = len(data)//1024
            if min_kb <= size_kb <= max_kb:
                with open(tmp_output_path, 'wb') as f: f.write(data)
                return True
            if size_kb <= max_kb:
                best_data, best_size = data, size_kb
        if best_data:
            with open(tmp_output_path, 'wb') as f: f.write(best_data)
            return True
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=MIN_QUALITY, optimize=True)
        with open(tmp_output_path, 'wb') as f: f.write(buf.getvalue())
        return False
    except Exception:
        shutil.copy2(tmp_input_path, tmp_output_path)
        return False

# ----------------------- Processing Tree -----------------------
def process_extracted_tree(root_dir: str):
    report_rows, processed_count, changed_count = [], 0, 0
    for current_root, _, files in os.walk(root_dir):
        for f in files:
            full = os.path.join(current_root, f)
            rel_path = os.path.relpath(full, root_dir)
            ext = pathlib.Path(full).suffix.lower()
            orig_size_kb = os.path.getsize(full)//1024
            category = guess_category_by_name(f)
            decided_via = 'name'

            if category is None and (ext in IMAGE_EXT or ext in PDF_EXT):
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tjpg:
                    tmpj = tjpg.name
                if convert_any_to_jpg(full, tmpj):
                    ocr_cat = ocr_guess_category(tmpj)
                    if ocr_cat:
                        category, decided_via = ocr_cat, 'ocr'
                os.remove(tmpj)

            if category is None:
                out_path = os.path.join(root_dir, pathlib.Path(rel_path).with_suffix('.jpg'))
                ensure_dir(os.path.dirname(out_path))
                converted = convert_any_to_jpg(full, out_path)
                final_size = os.path.getsize(out_path)//1024 if converted else orig_size_kb
                report_rows.append([rel_path, 'unknown', orig_size_kb, final_size, 'converted' if converted else 'skipped', decided_via])
                processed_count += 1
                continue

            min_kb, max_kb = TARGET_SIZES_KB.get(category, (10,200))
            tmp_in = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            tmp_out = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            tmp_in.close(); tmp_out.close()

            if not convert_any_to_jpg(full, tmp_in.name):
                report_rows.append([rel_path, category, orig_size_kb, orig_size_kb, 'convert_failed', decided_via])
                processed_count += 1
                continue

            success = compress_jpg_to_target(tmp_in.name, tmp_out.name, min_kb, max_kb)
            output_rel = pathlib.Path(rel_path).with_suffix('.jpg')
            out_full = os.path.join(root_dir, output_rel)
            ensure_dir(os.path.dirname(out_full))
            shutil.move(tmp_out.name, out_full)

            if os.path.abspath(full) != os.path.abspath(out_full):
                try: os.remove(full)
                except: pass

            final_size = os.path.getsize(out_full)//1024
            status = 'ok' if success and min_kb <= final_size <= max_kb else 'partial'
            report_rows.append([rel_path, category, orig_size_kb, final_size, status, decided_via])
            processed_count += 1; changed_count += 1
            os.remove(tmp_in.name)

    df = pd.DataFrame(report_rows, columns=['path','category','original_kb','final_kb','status','decided_via'])
    return df, processed_count, changed_count

# ----------------------- Streamlit UI -----------------------
st.set_page_config(page_title='Aiclex - ZIP to JPG Processor', page_icon='🧾', layout='wide')
st.title("Aiclex Technologies — ZIP to JPG Processor")
st.info("Upload ZIP/PDF/Images → Convert to JPG → Compress to required KB range → Download ZIP + Report")

uploaded = st.file_uploader("Upload file", type=['zip','pdf','png','jpg','jpeg','webp','tiff'])
if uploaded:
    with tempfile.TemporaryDirectory() as workdir:
        input_path = os.path.join(workdir, sanitize_name(uploaded.name))
        with open(input_path,'wb') as f: f.write(uploaded.getbuffer())

        extraction_root = os.path.join(workdir,'extracted'); ensure_dir(extraction_root)
        if pathlib.Path(input_path).suffix.lower() == '.zip':
            recursive_extract_zip(input_path, extraction_root)
        else:
            shutil.copy2(input_path, os.path.join(extraction_root, os.path.basename(input_path)))

        st.write("Processing...")
        start = time.time()
        report_df, processed_count, changed_count = process_extracted_tree(extraction_root)
        st.success(f"Done in {time.time()-start:.1f}s → {processed_count} files, {changed_count} changed")

        st.dataframe(report_df)

        csv_buf = io.StringIO(); report_df.to_csv(csv_buf, index=False)
        st.download_button("Download Report CSV", csv_buf.getvalue().encode('utf-8'), "report.csv")

        out_zip = os.path.join(workdir,"processed.zip")
        shutil.make_archive(out_zip.replace('.zip',''),'zip', extraction_root)
        with open(out_zip,'rb') as f: st.download_button("Download Processed ZIP", f, "processed_"+uploaded.name)
