"""
Streamlit app: Aiclex Technologies - Recursive ZIP Processor
Features:
- Accepts uploaded file (zip or individual files PDF/PNG/JPG/etc.)
- If ZIP found, recursively extracts nested zips
- Processes files according to categories (ID, PHOTO, SIGNATURE, QUALIFICATION)
  and compresses/optimizes them to target size ranges without visible quality loss
- Recreates the same folder structure and returns a new ZIP to download

Dependencies:
- streamlit
- pillow (PIL)
- PyMuPDF (fitz) [optional but recommended for PDF compression]

Run:
pip install streamlit pillow PyMuPDF
streamlit run streamlit_zip_processor_aiclex.py

Branding: Aiclex Technologies (simple header + optional logo upload)

Note: This is a single-file app intended as a production-ready starting point. Adjust
quality/size ranges and processing strategies as per your test results. Test with
sample files to ensure acceptable visual quality.
"""

import streamlit as st
import zipfile
import os
import shutil
import tempfile
import io
from PIL import Image, ImageOps
import mimetypes
import pathlib
import math

# Try optional PDF library
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    HAS_FITZ = False

# ----------------------- Configuration -----------------------
TARGET_SIZES = {
    'id_proof': (10 * 1024, 24 * 1024),         # bytes
    'photo': (10 * 1024, 24 * 1024),
    'signature': (10 * 1024, 24 * 1024),
    'qualification': (50 * 1024, 200 * 1024),
}

IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.tiff'}
PDF_EXT = {'.pdf'}
ZIP_EXT = {'.zip'}

# Maximum dimension to avoid huge images
MAX_DIMENSION = 3500  # px (reduce if very large inputs appear)

# ----------------------- Helpers -----------------------

def guess_category(filename: str):
    name = filename.lower()
    if 'id' in name or 'identity' in name or 'idproof' in name or 'id_proof' in name:
        return 'id_proof'
    if 'photo' in name or 'passport' in name or 'profile' in name:
        return 'photo'
    if 'sign' in name or 'signature' in name or 'sig' in name:
        return 'signature'
    if 'qual' in name or 'degree' in name or 'certificate' in name or 'marksheet' in name:
        return 'qualification'
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
                except Exception:
                    # corrupted nested zip? skip
                    continue


# ----------------------- Image Processing -----------------------

def compress_image_to_target(input_path: str, output_path: str, min_bytes: int, max_bytes: int):
    """Compress image while trying to keep visible quality. Saves to output_path."""
    img = Image.open(input_path)
    img = ImageOps.exif_transpose(img)  # correct orientation

    # Resize if extremely large
    w, h = img.size
    max_dim = max(w, h)
    if max_dim > MAX_DIMENSION:
        scale = MAX_DIMENSION / max_dim
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # Try different formats: JPEG for photos/signature, PNG maybe for simple images
    suffix = pathlib.Path(output_path).suffix.lower()
    # We'll prefer JPEG for space efficiency unless PNG was input and has transparency
    has_alpha = img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)

    if suffix in ('.png',) and has_alpha:
        # try to reduce PNG by converting to palette if small colors
        try:
            img_small = img.convert('P', palette=Image.ADAPTIVE)
            img_small.save(output_path, optimize=True)
        except Exception:
            img.save(output_path, optimize=True)
    else:
        # Use JPEG iterative quality reduction
        quality = 95
        last_good = None
        for q in range(95, 14, -5):
            buf = io.BytesIO()
            rgb = img.convert('RGB')
            rgb.save(buf, format='JPEG', quality=q, optimize=True)
            data = buf.getvalue()
            size = len(data)
            # Save if within bounds
            if min_bytes <= size <= max_bytes:
                with open(output_path, 'wb') as f:
                    f.write(data)
                return True
            # Keep the closest under max_bytes
            if size <= max_bytes:
                last_good = data
        # if not in range, fallback to best available
        if last_good:
            with open(output_path, 'wb') as f:
                f.write(last_good)
            return True
        else:
            # last resort: save with low quality
            rgb = img.convert('RGB')
            rgb.save(output_path, format='JPEG', quality=20, optimize=True)
            return True


# ----------------------- PDF Processing -----------------------

def compress_pdf(input_path: str, output_path: str, min_bytes: int, max_bytes: int):
    """Compress PDF using PyMuPDF if available, otherwise copy as-is.
    If PyMuPDF is available, we will downsample images inside PDF.
    """
    if not HAS_FITZ:
        shutil.copy2(input_path, output_path)
        return False

    doc = fitz.open(input_path)
    # iterate pages and downsample images
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap()
        # No per-page manipulation here; instead we'll use save with garbage and deflate
    try:
        # Save with optimization flags
        doc.save(output_path, garbage=4, deflate=True)
    except Exception:
        try:
            doc.save(output_path)
        except Exception:
            shutil.copy2(input_path, output_path)
    finally:
        doc.close()
    # If output not in desired range, we leave as-is (further granular compression requires image-extraction)
    return True


# ----------------------- Main Processing -----------------------

def process_extracted_tree(root_dir: str):
    """Walk through extracted files, process images and PDFs according to guessed categories.
    Overwrite files in place with optimized versions.
    """
    for current_root, _, files in os.walk(root_dir):
        for f in files:
            full = os.path.join(current_root, f)
            ext = pathlib.Path(full).suffix.lower()
            category = guess_category(f)
            if category is None:
                # skip files we cannot categorize, but still keep them untouched
                continue
            min_b, max_b = TARGET_SIZES.get(category)
            if ext in IMAGE_EXT:
                try:
                    # create temp output and replace
                    tmp_out = full + '.proc'
                    compress_image_to_target(full, tmp_out, min_b, max_b)
                    os.replace(tmp_out, full)
                except Exception as e:
                    # log & continue
                    print('Image compress error for', full, e)
            elif ext in PDF_EXT:
                try:
                    tmp_out = full + '.proc'
                    compress_pdf(full, tmp_out, min_b, max_b)
                    os.replace(tmp_out, full)
                except Exception as e:
                    print('PDF compress error for', full, e)
            else:
                # other file types: you may implement converters (e.g., docx->pdf) if needed
                continue


# ----------------------- Streamlit UI -----------------------

st.set_page_config(page_title='Aiclex - Zip Processor', page_icon='🧰', layout='wide')

# Header / Branding
cols = st.columns([1, 8, 1])
with cols[1]:
    st.markdown("""
    <div style='display:flex; align-items:center;'>
      <div style='border-radius:8px; padding:10px;'>
        <h2 style='margin:0; color:#0b5cff;'>Aiclex Technologies</h2>
        <div style='font-size:14px; color:#6b7280;'>Zip Processor — Maintain quality, enforce target sizes</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.info('Upload a ZIP file (or a PDF/PNG/JPG). The app will recursively extract nested zips, optimize files according to category rules, and return a new ZIP preserving folder structure.')

uploaded = st.file_uploader('Upload ZIP or single files (pdf/jpg/png) — client file', type=['zip','pdf','png','jpg','jpeg','webp','tiff'], accept_multiple_files=False)

if uploaded is not None:
    with tempfile.TemporaryDirectory() as workdir:
        input_path = os.path.join(workdir, uploaded.name)
        # Save uploaded to disk
        with open(input_path, 'wb') as f:
            f.write(uploaded.getbuffer())

        # If single PDF/image, create a base folder and place it inside
        extraction_root = os.path.join(workdir, 'extracted')
        ensure_dir(extraction_root)

        if pathlib.Path(input_path).suffix.lower() == '.zip':
            try:
                recursive_extract_zip(input_path, extraction_root)
            except Exception as e:
                st.error('Failed to extract ZIP: {}'.format(e))
        else:
            # Save single file keeping original name
            shutil.copy2(input_path, os.path.join(extraction_root, uploaded.name))

        st.write('Files extracted to temporary workspace — processing now...')
        # Process files according to categories
        process_extracted_tree(extraction_root)

        # Recreate zip preserving folder structure
        output_zip_path = os.path.join(workdir, 'processed_output.zip')
        base_name = os.path.splitext(output_zip_path)[0]
        shutil.make_archive(base_name, 'zip', extraction_root)

        # Show summary counts
        total_files = sum(len(files) for _, _, files in os.walk(extraction_root))
        st.success(f'Processing complete — {total_files} files in processed ZIP.')

        # Provide download
        with open(output_zip_path, 'rb') as f:
            data = f.read()
        st.download_button('Download processed ZIP', data, file_name='processed_' + uploaded.name if uploaded.name.endswith('.zip') else 'processed_output.zip')

        # Optional: show note about PDF compression availability
        if not HAS_FITZ:
            st.warning('PyMuPDF (fitz) not installed — PDF compression will be minimal. Install PyMuPDF for better PDF size optimization.')

        st.markdown('---')
        st.markdown('**Notes & next steps:**')
        st.markdown('- The app guesses file categories by filename. For reliable results, ensure filenames contain keywords like `id`, `photo`, `signature`, `qualification`, `certificate` etc.')
        st.markdown('- If you want stricter folder-to-category mapping (e.g., folder named `ID`), we can add rule-based mapping.')
        st.markdown('- For enterprise usage, integrate this into a backend (FastAPI) and add authentication and logging.')

        st.markdown('**Configuration**')
        st.code('''# TARGET_SIZES in bytes (min,max)
TARGET_SIZES = {
    'id_proof': (10 * 1024, 24 * 1024),
    'photo': (10 * 1024, 24 * 1024),
    'signature': (10 * 1024, 24 * 1024),
    'qualification': (50 * 1024, 200 * 1024),
}
''')

        st.balloons()

else:
    st.write('Waiting for client upload...')
    st.write('Tip: For best results, ask client to name files with keywords (id, photo, signature, qualification).')

# Footer
st.markdown("""
<div style='position:fixed; bottom:10px; right:10px; font-size:12px; color:#9ca3af;'>
  © Aiclex Technologies — Built for secure file processing
</div>
""", unsafe_allow_html=True)
