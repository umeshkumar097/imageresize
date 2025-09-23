"""
Aiclex Technologies — ZIP to JPG Processor (Improved detection)
- Improvements:
  - Face detection (OpenCV) to mark PHOTOS
  - OCR + keyword matching for documents/qualification/signature
  - Visual heuristics (edge density / ink pixel %) to identify SIGNATUREs
  - More robust fallback logic
- Requirements (add to requirements.txt):
  streamlit
  pillow
  pdf2image
  pytesseract
  PyMuPDF
  pandas
  opencv-python
  numpy
"""

import streamlit as st
import zipfile, os, shutil, tempfile, io, time, pathlib
from PIL import Image, ImageOps, ImageFilter
import pandas as pd
from pdf2image import convert_from_path, exceptions as pdf2img_exceptions
import pytesseract
import numpy as np
import cv2

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

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def is_zip_file(path): return pathlib.Path(path).suffix.lower() == '.zip'

# ----------------------- Detection Helpers -----------------------
# Keywords for OCR-based detection
QUAL_KEYWORDS = ['degree','certificate','marksheet','qualification','transcript','result','diploma']
ID_KEYWORDS = ['id','identity','aadhar','pan','passport','driving license','driving-license','voter']
SIGN_KEYWORDS = ['signature','signed','sign']

# Face detection using OpenCV Haarcascade (uses cv2.data.haarcascades path)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face_in_image(pil_img):
    try:
        # convert to grayscale numpy
        img = pil_img.convert('RGB')
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
        return len(faces) > 0
    except Exception:
        return False

def ocr_text_from_image(pil_img):
    try:
        # small resize for speed if huge
        w,h = pil_img.size
        if max(w,h) > 2000:
            pil_img = pil_img.resize((int(w*0.5), int(h*0.5)), Image.LANCZOS)
        text = pytesseract.image_to_string(pil_img, lang='eng')
        return text.lower()
    except Exception:
        return ""

def signature_heuristic(pil_img):
    """
    Heuristic to detect signature-like images:
    - Low OCR text content
    - High edge density but low overall ink coverage (signature strokes)
    - Aspect (width vs height) often long and narrow, but not always
    Returns score and boolean
    """
    try:
        gray = pil_img.convert('L')
        # Resize for performance
        w,h = gray.size
        base = 800
        if max(w,h) > base:
            scale = base / max(w,h)
            gray = gray.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

        arr = np.array(gray)
        # normalized ink % (dark pixels)
        ink_pixels = np.sum(arr < 200)
        tot = arr.size
        ink_pct = ink_pixels / tot

        # edge density
        edges = cv2.Canny(arr, 50, 150)
        edge_count = np.sum(edges > 0)
        edge_density = edge_count / tot

        # small handwriting/signatures: low ink_pct but moderate-high edge density
        # thresholds tuned empirically (may be adjusted)
        is_signature = (ink_pct < 0.06 and edge_density > 0.005) or (edge_density > 0.02 and ink_pct < 0.15)
        score = edge_density * 100 - ink_pct * 50
        return score, bool(is_signature)
    except Exception:
        return 0.0, False

def guess_category_visual_and_ocr(pil_img, filename):
    """
    Combined logic:
    1. filename (quick)
    2. face detection -> photo
    3. OCR keywords -> id/qualification/signature
    4. signature heuristic -> signature
    5. OCR text length small & many non-text strokes -> signature
    6. fallback: unknown
    """
    name = filename.lower()
    # 1. filename hints
    # direct checks similar to earlier helper
    if ('id' in name and ('proof' in name or 'id' in name)) or any(k in name for k in ID_KEYWORDS):
        return 'id_proof', 'name'
    if any(k in name for k in ['photo','passport','profile']):
        return 'photo', 'name'
    if any(k in name for k in ['sign','signature','sig']):
        return 'signature', 'name'
    if any(k in name for k in QUAL_KEYWORDS):
        return 'qualification', 'name'

    # 2. face detection
    try:
        if detect_face_in_image(pil_img):
            return 'photo', 'face'
    except Exception:
        pass

    # 3. OCR
    text = ocr_text_from_image(pil_img)
    # direct keyword detection
    for k in QUAL_KEYWORDS:
        if k in text: return 'qualification', 'ocr'
    for k in ID_KEYWORDS:
        if k in text: return 'id_proof', 'ocr'
    for k in SIGN_KEYWORDS:
        if k in text: return 'signature', 'ocr'
    # 4. signature heuristic
    score, sig_flag = signature_heuristic(pil_img)
    if sig_flag:
        return 'signature', 'visual'
    # 5. if OCR has very little text but image has strokes -> signature
    words = [w for w in text.split() if w.isalpha()]
    if len(words) < 6 and score > 0.5:
        return 'signature', 'heuristic'
    # 6. if OCR moderate-large text -> document / id
    if len(words) >= 6 and len(words) < 80:
        # decide id vs qualification using keywords earlier checked, default id_proof
        return 'id_proof', 'ocr_len'
    if len(words) >= 80:
        return 'qualification', 'ocr_len'

    return None, 'none'

# ----------------------- Conversion & Compression (reused) -----------------------
from PIL import Image
import io
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
        if max(w,h) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(w,h)
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

        best_data = None
        for q in range(95, MIN_QUALITY-1, -5):
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=q, optimize=True)
            data = buf.getvalue()
            size_kb = len(data)//1024
            if min_kb <= size_kb <= max_kb:
                with open(tmp_output_path, 'wb') as f: f.write(data)
                return True
            if size_kb <= max_kb:
                best_data = data
        if best_data:
            with open(tmp_output_path,'wb') as f: f.write(best_data); return True
        buf = io.BytesIO(); img.save(buf, format='JPEG', quality=MIN_QUALITY, optimize=True)
        with open(tmp_output_path,'wb') as f: f.write(buf.getvalue())
        return False
    except Exception:
        shutil.copy2(tmp_input_path, tmp_output_path)
        return False

# ----------------------- ZIP extract (recursive) -----------------------
def recursive_extract_zip(zip_path: str, dest_dir: str):
    with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(dest_dir)
    for root, _, files in os.walk(dest_dir):
        for f in files:
            full = os.path.join(root, f)
            if is_zip_file(full):
                nested_dir = os.path.splitext(full)[0]
                try:
                    with zipfile.ZipFile(full, 'r') as nz: nz.extractall(nested_dir)
                    os.remove(full)
                except Exception:
                    continue

# ----------------------- Process Tree (with improved detection) -----------------------
def process_extracted_tree(root_dir: str):
    report_rows = []
    processed, changed = 0, 0
    for cur_root, _, files in os.walk(root_dir):
        for f in files:
            full = os.path.join(cur_root, f)
            rel = os.path.relpath(full, root_dir)
            ext = pathlib.Path(full).suffix.lower()
            orig_kb = os.path.getsize(full)//1024
            decided_by = 'filename'

            # If it's zip nested (rare here), skip (we extracted earlier)
            if ext == '.zip':
                continue

            # Try load PIL for detection (if convertible)
            pil_img = None
            can_open = False
            if ext in IMAGE_EXT:
                try:
                    pil_img = Image.open(full)
                    pil_img = ImageOps.exif_transpose(pil_img)
                    can_open = True
                except Exception:
                    can_open = False
            elif ext in PDF_EXT:
                # convert first page to image for detection
                try:
                    pages = convert_from_path(full, dpi=200, first_page=1, last_page=1)
                    if pages:
                        pil_img = pages[0].convert('RGB')
                        can_open = True
                    else:
                        can_open = False
                except Exception:
                    can_open = False
            else:
                # other file types: attempt to open as image (some tiffs/webp handled above)
                try:
                    pil_img = Image.open(full)
                    pil_img = ImageOps.exif_transpose(pil_img)
                    can_open = True
                except Exception:
                    can_open = False

            category = None
            dec_method = 'none'
            # First try filename-based quick guess
            lname = f.lower()
            if ('id' in lname and ('proof' in lname or 'id' in lname)) or any(k in lname for k in ID_KEYWORDS):
                category, dec_method = 'id_proof', 'name'
            elif any(k in lname for k in ['photo','passport','profile']):
                category, dec_method = 'photo', 'name'
            elif any(k in lname for k in ['sign','signature','sig']):
                category, dec_method = 'signature', 'name'
            elif any(k in lname for k in QUAL_KEYWORDS):
                category, dec_method = 'qualification', 'name'

            # If not decided and we have an image, do visual+OCR detection
            if category is None and can_open and pil_img is not None:
                cat, method = guess_category_visual_and_ocr(pil_img, f)
                category, dec_method = (cat, method) if cat else (None, method)

            # If still None, fallback to converting to jpg and leave as unknown
            if category is None:
                # convert to jpg (no category)
                out_rel = pathlib.Path(rel).with_suffix('.jpg')
                out_full = os.path.join(root_dir, out_rel)
                ensure_dir(os.path.dirname(out_full))
                converted = convert_any_to_jpg(full, out_full)
                final_kb = os.path.getsize(out_full)//1024 if converted else orig_kb
                status = 'converted' if converted else 'skipped'
                report_rows.append([rel, 'unknown', orig_kb, final_kb, status, dec_method])
                processed += 1
                continue

            # Now we have category - enforce JPG + size range
            min_kb, max_kb = TARGET_SIZES_KB.get(category, (10,200))
            t_in = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False); t_out = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            t_in.close(); t_out.close()
            if not convert_any_to_jpg(full, t_in.name):
                report_rows.append([rel, category, orig_kb, orig_kb, 'convert_failed', dec_method])
                processed += 1
                # cleanup temp
                try: os.remove(t_in.name); os.remove(t_out.name)
                except: pass
                continue
            success = compress_jpg_to_target(t_in.name, t_out.name, min_kb, max_kb)
            # move to final place (replace extension with .jpg)
            out_rel = pathlib.Path(rel).with_suffix('.jpg')
            out_full = os.path.join(root_dir, out_rel)
            ensure_dir(os.path.dirname(out_full))
            try:
                shutil.move(t_out.name, out_full)
            except Exception:
                try: shutil.copy2(t_out.name, out_full)
                except Exception: pass
            # remove original if different
            try:
                if os.path.abspath(full) != os.path.abspath(out_full):
                    os.remove(full)
            except Exception:
                pass
            final_kb = os.path.getsize(out_full)//1024 if os.path.exists(out_full) else orig_kb
            status = 'ok' if success and min_kb <= final_kb <= max_kb else 'partial'
            report_rows.append([rel, category, orig_kb, final_kb, status, dec_method])
            processed += 1
            if status in ('ok','partial'): changed += 1
            # cleanup tmp input
            try: os.remove(t_in.name)
            except: pass

    df = pd.DataFrame(report_rows, columns=['path','category','original_kb','final_kb','status','decided_via'])
    return df, processed, changed

# ----------------------- Streamlit UI -----------------------
st.set_page_config(page_title='Aiclex - ZIP to JPG Processor', page_icon='🧾', layout='wide')
st.title("Aiclex — ZIP to JPG Processor (Improved Detection)")
st.write("Upload ZIP / PDF / Images. Detection improved: face, OCR keywords, and visual heuristics for signatures.")

uploaded = st.file_uploader("Upload file (zip/pdf/jpg/png/tiff/webp)", type=['zip','pdf','png','jpg','jpeg','webp','tiff'])
if uploaded:
    with tempfile.TemporaryDirectory() as workdir:
        input_path = os.path.join(workdir, sanitize_name(uploaded.name))
        with open(input_path,'wb') as f: f.write(uploaded.getbuffer())
        extraction_root = os.path.join(workdir,'extracted'); ensure_dir(extraction_root)
        if pathlib.Path(input_path).suffix.lower() == '.zip':
            try:
                recursive_extract_zip(input_path, extraction_root)
            except Exception as e:
                st.error(f"ZIP extraction failed: {e}"); st.stop()
        else:
            shutil.copy2(input_path, os.path.join(extraction_root, os.path.basename(input_path)))

        st.info("Processing started...")
        start = time.time()
        report_df, processed_count, changed_count = process_extracted_tree(extraction_root)
        elapsed = time.time() - start
        st.success(f"Done in {elapsed:.1f}s — processed {processed_count}, changed {changed_count}")

        st.subheader("Report")
        st.dataframe(report_df)

        csv_buf = io.StringIO(); report_df.to_csv(csv_buf, index=False)
        st.download_button("Download report CSV", csv_buf.getvalue().encode('utf-8'), "processing_report.csv")

        out_zip = os.path.join(workdir, "processed_output.zip")
        shutil.make_archive(out_zip.replace('.zip',''), 'zip', extraction_root)
        with open(out_zip, 'rb') as f:
            st.download_button("Download processed ZIP", f, "processed_" + sanitize_name(uploaded.name))
else:
    st.info("Waiting for upload...")

# Footer
st.markdown("<small>© Aiclex Technologies — improved detection (face, OCR, signature heuristics)</small>", unsafe_allow_html=True)
