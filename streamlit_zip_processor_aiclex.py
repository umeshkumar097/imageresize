"""
Streamlit app: Aiclex Technologies - Recursive ZIP Processor (Complete)

Features:
- Accepts uploaded file (zip or individual files PDF/PNG/JPG/etc.)
- Recursively extracts nested zips
- Converts all documents/images to JPG
- If filename indicates category (id/photo/signature/qualification) it uses that; otherwise runs OCR to guess
- Compresses JPGs into strict target size ranges (by iterative quality and resizing)
- Preserves folder structure and returns a processed ZIP
- Generates a CSV/Streamlit table report with original vs final sizes + status

Dependencies (pip):
- streamlit
- pillow
- pdf2image
- pytesseract
- PyMuPDF (optional)

System packages required:
- poppler-utils (for pdf2image) — install via apt / brew
- tesseract-ocr (for pytesseract) — install via apt / brew

Run:
- pip install -r requirements.txt
- streamlit run streamlit_zip_processor_aiclex.py

Notes:
- PDF -> JPG conversion uses pdf2image (needs poppler). For environments without poppler, PDF pages will be skipped.
- OCR is used only when filename doesn't reveal category.
"""

import streamlit as st
import zipfile
import os
import shutil
import tempfile
import io
from PIL import Image, ImageOps
import pathlib
import csv
import pandas as pd
from pdf2image import convert_from_path, exceptions as pdf2img_exceptions
import pytesseract
import time

# ----------------------- Configuration -----------------------
TARGET_SIZES_KB = {
    'id_proof': (10, 24),         # KB
    'photo': (10, 24),
    'signature': (10, 24),
    'qualification': (50, 200),
}

IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.tiff'}
PDF_EXT = {'.pdf'}
ZIP_EXT = {'.zip'}

MAX_DIMENSION = 3500  # px - reduce very large images
MIN_QUALITY = 10

# ----------------------- Helpers -----------------------

def sanitize_name(name: str) -> str:
    # simple sanitize to avoid weird chars in file names
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
    # try suffix/last token
    tokens = [t for t in pathlib.Path(name).stem.split() if t]
    if tokens:
        last = tokens[-1]
        return guess_category_by_name(last)
    return None


def ocr_guess_category(image_path: str):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='eng')
        text = text.lower()
        if any(k in text for k in ['id', 'identity', 'aadhar', 'pan', 'passport']):
            return 'id_proof'
        if any(k in text for k in ['photo', 'passport', 'profile']):
            return 'photo'
        if any(k in text for k in ['signature', 'signed']):
            return 'signature'
        if any(k in text for k in ['degree', 'certificate', 'marksheet', 'qualification']):
            return 'qualification'
    except Exception:
        return None
    return None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def is_zip_file(path: str):
    return pathlib.Path(path).suffix.lower() == '.zip'


def recursive_extract_zip(zip_path: str, dest_dir: str):
    """Extract zip and recursively extract nested zips preserving structure."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)

    # walk and extract nested zips
    for root, _, files in os.walk(dest_dir):
        for f in files:
            full = os.path.join(root, f)
            if is_zip_file(full):
                nested_dir = os.path.splitext(full)[0]
                try:
                    with zipfile.ZipFile(full, 'r') as nz:
                        nz.extractall(nested_dir)
                    # optionally remove nested zip after extraction
                    os.remove(full)
                except Exception:
                    # corrupted nested zip? skip
                    continue


# ----------------------- Conversion & Compression -----------------------

def convert_any_to_jpg(input_path: str, output_path: str, dpi=200):
    """Convert images or PDFs to a JPG (for PDF, uses first page)."""
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


def compress_jpg_to_target(tmp_input_path: str, tmp_output_path: str, min_kb: int, max_kb: int):
    """Iteratively reduce quality and resize until within target KB range.
    Returns True if success, False otherwise (but still writes an output).
    """
    try:
        img = Image.open(tmp_input_path)
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        # initial resize if massive
        max_dim = max(w, h)
        if max_dim > MAX_DIMENSION:
            scale = MAX_DIMENSION / max_dim
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # try descending quality
        best_data = None
        best_size = None
        for q in range(95, MIN_QUALITY - 1, -5):
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=q, optimize=True)
            data = buf.getvalue()
            size_kb = len(data) // 1024
            # exact match
            if min_kb <= size_kb <= max_kb:
                with open(tmp_output_path, 'wb') as f:
                    f.write(data)
                return True
            # keep best under max_kb
            if size_kb <= max_kb:
                best_data = data
                best_size = size_kb
            # if size is still too big, continue loop
        # If not found, try resizing gradually
        if best_data is not None:
            with open(tmp_output_path, 'wb') as f:
                f.write(best_data)
            return True

        # try iterative downscale + quality
        scale = 0.9
        attempts = 0
        while attempts < 8:
            attempts += 1
            nw = int(img.width * scale)
            nh = int(img.height * scale)
            if nw < 50 or nh < 50:
                break
            img_small = img.resize((nw, nh), Image.LANCZOS)
            for q in range(85, MIN_QUALITY - 1, -5):
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
                    best_size = size_kb
            scale -= 0.1
        # fallback: write best_data if any
        if best_data:
            with open(tmp_output_path, 'wb') as f:
                f.write(best_data)
            return True
        # last resort: save very low quality
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=MIN_QUALITY, optimize=True)
        with open(tmp_output_path, 'wb') as f:
            f.write(buf.getvalue())
        return False
    except Exception as e:
        # on error, just copy input to output
        try:
            shutil.copy2(tmp_input_path, tmp_output_path)
        except Exception:
            pass
        return False


# ----------------------- Processing Tree -----------------------

def process_extracted_tree(root_dir: str):
    report_rows = []
    processed_count = 0
    changed_count = 0

    for current_root, _, files in os.walk(root_dir):
        for f in files:
            full = os.path.join(current_root, f)
            rel_path = os.path.relpath(full, root_dir)
            name = sanitize_name(f)
            ext = pathlib.Path(full).suffix.lower()
            orig_size_kb = os.path.getsize(full) // 1024
            category = guess_category_by_name(name)
            decided_via = 'name'

            # If not identified and image/pdf -> try quick OCR on a converted jpg
            if category is None and (ext in IMAGE_EXT or ext in PDF_EXT):
                # create temp jpg for OCR
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tjpg:
                    tmpj = tjpg.name
                ok = convert_any_to_jpg(full, tmpj)
                if ok:
                    ocr_cat = ocr_guess_category(tmpj)
                    if ocr_cat:
                        category = ocr_cat
                        decided_via = 'ocr'
                    os.remove(tmpj)

            if category is None:
                # unknown - leave file unchanged (but if not jpg, convert to jpg by default)
                target_ext = '.jpg'
                output_name = pathlib.Path(rel_path).with_suffix(target_ext)
                out_path = os.path.join(root_dir, output_name)
                ensure_dir(os.path.dirname(out_path))
                converted = convert_any_to_jpg(full, out_path)
                final_size_kb = os.path.getsize(out_path) // 1024 if converted else orig_size_kb
                status = 'converted' if converted else 'skipped'
                report_rows.append([rel_path, category or 'unknown', orig_size_kb, final_size_kb, status, decided_via])
                processed_count += 1
                continue

            # We have a category -> enforce target size and JPG
            min_kb, max_kb = TARGET_SIZES_KB.get(category, (10, 200))
            # convert to jpg first
            tmp_in = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            tmp_in.close()
            tmp_out = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            tmp_out.close()
            converted_ok = convert_any_to_jpg(full, tmp_in.name)
            if not converted_ok:
                # cannot convert - copy as-is
                status = 'convert_failed'
                final_size_kb = orig_size_kb
                report_rows.append([rel_path, category, orig_size_kb, final_size_kb, status, decided_via])
                processed_count += 1
                # cleanup tmp
                try:
                    os.remove(tmp_in.name)
                    os.remove(tmp_out.name)
                except Exception:
                    pass
                continue

            # compress to target
            success = compress_jpg_to_target(tmp_in.name, tmp_out.name, min_kb, max_kb)

            # place output replacing original but keep folder structure and .jpg extension
            output_rel = pathlib.Path(rel_path).with_suffix('.jpg')
            out_full = os.path.join(root_dir, output_rel)
            ensure_dir(os.path.dirname(out_full))
            try:
                shutil.move(tmp_out.name, out_full)
            except Exception:
                # fallback copy
                try:
                    shutil.copy2(tmp_out.name, out_full)
                except Exception:
                    pass
            # remove original file if output placed at different path
            try:
                if os.path.abspath(full) != os.path.abspath(out_full):
                    os.remove(full)
            except Exception:
                pass

            final_size_kb = os.path.getsize(out_full) // 1024 if os.path.exists(out_full) else orig_size_kb
            status = 'ok' if success and (min_kb <= final_size_kb <= max_kb) else 'partial' if os.path.exists(out_full) else 'failed'
            report_rows.append([rel_path, category, orig_size_kb, final_size_kb, status, decided_via])
            processed_count += 1
            if status in ('ok','partial'):
                changed_count += 1

            # cleanup tmp input
            try:
                if os.path.exists(tmp_in.name):
                    os.remove(tmp_in.name)
            except Exception:
                pass

    report_df = pd.DataFrame(report_rows, columns=['path', 'category', 'original_kb', 'final_kb', 'status', 'decided_via'])
    return report_df, processed_count, changed_count


# ----------------------- Streamlit UI -----------------------

st.set_page_config(page_title='Aiclex - ZIP to JPG Processor', page_icon='🧾', layout='wide')

# Header
cols = st.columns([1, 8, 1])
with cols[1]:
    st.markdown("""
    <div style='display:flex; align-items:center;'>
      <div style='border-radius:8px; padding:10px;'>
        <h2 style='margin:0; color:#0b5cff;'>Aiclex Technologies</h2>
        <div style='font-size:14px; color:#6b7280;'>ZIP -> JPG Processor — strict size enforcement</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.info('Upload a ZIP file (or a PDF/PNG/JPG). The app will recursively extract nested zips, convert files to JPG, enforce size ranges per category, and return a processed ZIP plus a CSV report.')

uploaded = st.file_uploader('Upload ZIP or single file (pdf/jpg/png/tiff/webp) — client file', type=['zip','pdf','png','jpg','jpeg','webp','tiff'], accept_multiple_files=False)

if uploaded is not None:
    with tempfile.TemporaryDirectory() as workdir:
        input_path = os.path.join(workdir, sanitize_name(uploaded.name))
        with open(input_path, 'wb') as f:
            f.write(uploaded.getbuffer())

        extraction_root = os.path.join(workdir, 'extracted')
        ensure_dir(extraction_root)

        if pathlib.Path(input_path).suffix.lower() == '.zip':
            try:
                recursive_extract_zip(input_path, extraction_root)
            except Exception as e:
                st.error('Failed to extract ZIP: {}'.format(e))
                st.stop()
        else:
            # single file: place inside root preserving name
            shutil.copy2(input_path, os.path.join(extraction_root, os.path.basename(input_path)))

        st.write('Files extracted — starting processing...')
        start = time.time()
        report_df, processed_count, changed_count = process_extracted_tree(extraction_root)
        elapsed = time.time() - start

        st.success(f'Processing complete in {elapsed:.1f}s — processed {processed_count} files, changed {changed_count}.')

        # Show report
        st.subheader('Report')
        st.dataframe(report_df)

        # Offer CSV download
        csv_buf = io.StringIO()
        report_df.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode('utf-8')
        st.download_button('Download report CSV', csv_bytes, file_name='processing_report.csv')

        # Create processed ZIP
        output_zip_path = os.path.join(workdir, 'processed_output.zip')
        base_name = os.path.splitext(output_zip_path)[0]
        shutil.make_archive(base_name, 'zip', extraction_root)
        with open(output_zip_path, 'rb') as f:
            data = f.read()
        st.download_button('Download processed ZIP', data, file_name='processed_' + sanitize_name(uploaded.name) )

        st.markdown('**Notes**')
        st.markdown('- Filenames are used to guess category first. If not found OCR is attempted on image/PDF. For best results, name files with keywords `id/photo/signature/qualification`.')
        st.markdown('- PDF -> JPG requires `poppler` installed on the system for `pdf2image` to work. If not present, PDFs may fail conversion.')
        st.markdown('- OCR requires `tesseract-ocr` installed. If not present, OCR guessing will be skipped.')
        st.markdown('- All outputs are JPG. Original files that were not convertible are left as-is in the ZIP.')

        st.balloons()

else:
    st.write('Waiting for client upload...')

# Footer
st.markdown("""
<div style='position:fixed; bottom:10px; right:10px; font-size:12px; color:#9ca3af;'>
  © Aiclex Technologies — Built for secure file processing
</div>
""", unsafe_allow_html=True)
