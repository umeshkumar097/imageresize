# --- Aiclex Media Processor (Streamlit UI) ---
# Author: Aiclex Technologies (aiclex.in)
# Version: 1.4.1 (Signature Auto-Crop Enhanced + Quality Safe) + Aadhaar OCR-crop
# ---------------------------------------------------

import streamlit as st
import pandas as pd
import os
import zipfile
import shutil
import tempfile
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np
import pytesseract
import uuid

# Make sure your Tesseract path is set 


try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# --- SESSION STATE INIT ---
if "processed" not in st.session_state:
    st.session_state.processed = False
if "output_zip_path" not in st.session_state:
    st.session_state.output_zip_path = None
if "output_csv_path" not in st.session_state:
    st.session_state.output_csv_path = None
if "df_report" not in st.session_state:
    st.session_state.df_report = pd.DataFrame()
if "base_name" not in st.session_state:
    st.session_state.base_name = ""

# --- CONFIG ---
TARGET_SIZES_KB = {
    "id_proof": (10, 24),
    "photo": (10, 24),
    "signature": (10, 24),
    "qualification_proof": (50, 200),
    "unknown": (10, 500)
}

CATEGORY_KEYWORDS = {
    "id_proof": ["id", "aadhaar", "aadhar", "adhar", "pan", "passport", "dl", "driving",
                 "identity", "idcard", "id_proof", "idproof", "pan_card",
                 "passport_scan", "drivinglicence", "dlcard", "id-card","Aadhar","AADHAR","Adhaar"],
    "photo": ["photo", "pic", "profile", "image", "photograph", "passportphoto",
              "profilepic", "profile_pic", "dp", "phto", "phoo", "img", "pic1","phtto"],
    "signature": ["sign", "signature", "sig", "sgn", "sin", "si", "autograph",
                  "signutre", "signatr", "signeture", "signa", "signatur",
                  "signtr", "sigture", "sgture", "sgnature", "signture"],
    "qualification_proof": ["degree", "certificate", "marksheet", "qualification",
                            "qual", "degree_certificate", "deg_cert", "cert",
                            "marksheets", "qual_proof", "qualificationproof",
                            "degreeproof", "deg_proof", "QUAL", "Qual"]
}

SUPPORTED_MEDIA_EXT = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf']
MIN_QUALITY = 10
DPI = 200

# --- CORE FUNCTIONS ---

def extract_all_zips(start_dir, status_placeholder):
    """Recursively extract nested ZIPs."""
    while True:
        zip_found = False
        for root, _, files in os.walk(start_dir):
            for f in files:
                if f.lower().endswith('.zip'):
                    zip_path = Path(root) / f
                    extract_folder = zip_path.with_suffix('')
                    status_placeholder.text(f"→ Extracting nested ZIP: {f}")
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_folder)
                        zip_path.unlink()
                        zip_found = True
                    except zipfile.BadZipFile:
                        st.warning(f"Bad ZIP file skipped: {f}")
                    break
            if zip_found:
                break
        if not zip_found:
            break


def guess_category_by_filename(filename):
    fn_lower = Path(filename).stem.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in fn_lower for keyword in keywords):
            return category
    return "unknown"


def convert_any_to_jpg(input_path, output_jpg_path):
    """Convert PDF or image to JPG."""
    try:
        ext = input_path.suffix.lower()
        output_jpg_path.parent.mkdir(parents=True, exist_ok=True)

        if ext == '.pdf':
            import fitz
            doc = fitz.open(str(input_path))
            if doc.page_count == 0:
                return False
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            pix.save(str(output_jpg_path))
            return True

        elif ext in SUPPORTED_MEDIA_EXT:
            with Image.open(input_path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output_jpg_path, 'JPEG')
            return True

        return False
    except Exception as e:
        print(f"Conversion failed for {input_path}: {e}")
        return False


# ✅ IMPROVED SIGNATURE AUTO-CROP LOGIC
def autocrop_signature_image(image_path, save_path):
    """
    Robust auto-cropping for signature images, ignoring watermark text at bottom.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"⚠️ Failed to load image for cropping: {image_path}")
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        inverted = cv2.bitwise_not(gray)
        _, thresh = cv2.threshold(inverted, 180, 255, cv2.THRESH_BINARY)

        # ⚙️ Ignore bottom 20% area (where "Scanned with ..." text usually appears)
        h, w = thresh.shape
        ignore_height = int(h * 0.2)
        thresh[h - ignore_height:, :] = 0  # black out bottom region

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("⚠️ No signature contours found.")
            return False

        # Merge contours
        x, y, ww, hh = cv2.boundingRect(np.vstack(contours))

        # Padding
        pad = 50
        x, y = max(0, x - pad), max(0, y - pad)
        ww, hh = min(img.shape[1] - x, ww + 2 * pad), min(img.shape[0] - y, hh + 2 * pad)

        cropped = img[y:y+hh, x:x+ww]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        Image.fromarray(cropped).save(save_path, "JPEG", quality=95)

        print(f"✅ Cropped and saved to {save_path}")
        return True

    except Exception as e:
        print(f"❌ Auto-crop failed: {e}")
        return False


# --- Aadhaar helpers to be added carefully (do not disturb other logic) ---

import cv2
import pytesseract
import numpy as np
from PIL import Image
import os

# --- Aadhaar Front Detector ---
def detect_and_crop_aadhaar_front(image_path, save_path):
    """
    Detect and crop Aadhaar front side intelligently.
    Uses both filename keywords and OCR-based detection.
    """

    # ✅ Step 0: Quick filename-based Aadhaar check
    filename = os.path.basename(image_path).lower()
    aadhaar_file_keywords = ["aadhaar", "aadhar", "adhar"]

    aadhaar_in_name = any(k in filename for k in aadhaar_file_keywords)

    # --- Step 1: Load and preprocess image ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Failed to read image: {image_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Run OCR only if Aadhaar not detected in filename
    if not aadhaar_in_name:
        text = pytesseract.image_to_string(gray)
        front_keywords = [
            "government of india",
            "unique identification",
            "aadhaar",
            "aadhar",
            "adhar",
            "dob",
            "vid:"
        ]
        if not any(k in text.lower() for k in front_keywords):
            print("❌ Aadhaar front not detected by OCR or filename.")
            return False
    else:
        print("📄 Aadhaar detected from filename — proceeding with cropping.")

    # --- Step 2: Face detection to locate photo region ---
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        # Expand around face region
        x1 = max(x - 80, 0)
        y1 = max(y - 120, 0)
        x2 = min(x + w + 400, img.shape[1])
        y2 = min(y + h + 200, img.shape[0])
        crop = img[y1:y2, x1:x2]
    else:
        # Fallback: crop center region if no face detected
        h, w = img.shape[:2]
        crop = img[int(h * 0.2):int(h * 0.8), int(w * 0.1):int(w * 0.9)]

    # --- Step 3: Save cropped Aadhaar front ---
    im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    im.save(save_path, "JPEG", quality=90)

    print(f"✅ Cropped Aadhaar front saved: {save_path}")
    return True



# --- Compression ---
def compress_jpg_to_target(input_path, output_path, min_kb, max_kb):
    """Compress JPEG to target size range."""
    try:
        img = Image.open(input_path)
        img_copy = img.copy()

        for quality in range(95, MIN_QUALITY - 1, -5):
            img_copy.save(output_path, 'JPEG', quality=quality, optimize=True)
            size_kb = output_path.stat().st_size / 1024
            if min_kb <= size_kb <= max_kb:
                return "ok", size_kb

        for scale in [0.9, 0.75, 0.5, 0.25]:
            new_size = (int(img_copy.width * scale), int(img_copy.height * scale))
            img_resized = img_copy.resize(new_size, Image.Resampling.LANCZOS)
            for quality in range(90, MIN_QUALITY - 1, -10):
                img_resized.save(output_path, 'JPEG', quality=quality, optimize=True)
                size_kb = output_path.stat().st_size / 1024
                if min_kb <= size_kb <= max_kb:
                    return "ok", size_kb

        img_copy.save(output_path, 'JPEG', quality=MIN_QUALITY, optimize=True)
        return "ok", output_path.stat().st_size / 1024
    except Exception:
        return "compress_failed", 0
# --- Helper: Aadhaar filename detection ---
def is_aadhaar_filename(filename: str) -> bool:
    """
    Detect if a file is likely Aadhaar based on its name.
    """
    keywords = ["aadhaar", "aadhar", "adhar"]
    return any(k in filename.lower() for k in keywords)



def process_tree(source_dir, processed_dir, status_placeholder):
    """Walk through files, process and crop signatures and Aadhaar when applicable."""
    report_data = []
    status_placeholder.text("Starting file processing...")
    files_to_process = [p for p in source_dir.rglob('*') if p.is_file() and not p.name.lower().endswith('.zip')]
    progress_bar = st.progress(0)

    for i, input_path in enumerate(files_to_process):
        relative_path = input_path.relative_to(source_dir)
        status_placeholder.text(f"Processing: {relative_path}")
        original_kb = input_path.stat().st_size / 1024
        ext = input_path.suffix.lower()

        if ext in SUPPORTED_MEDIA_EXT:
            temp_jpg_path = Path(tempfile.gettempdir()) / f"temp_{uuid.uuid4().hex}_{relative_path.name}.jpg"
            if not convert_any_to_jpg(input_path, temp_jpg_path):
                report_data.append([relative_path, 'unknown', f"{original_kb:.2f}", 0, "convert_failed", "name"])
                continue

            category = guess_category_by_filename(input_path.name)
            min_kb, max_kb = TARGET_SIZES_KB.get(category, TARGET_SIZES_KB["unknown"])

            # ✅ Aadhaar OCR crop if filename indicates Aadhaar
            try:
                if is_aadhaar_filename(input_path.name):
                    aadhaar_cropped_path = Path(tempfile.gettempdir()) / f"aadhaar_{uuid.uuid4().hex}_{relative_path.name}.jpg"
                    success = detect_and_crop_aadhaar_front(temp_jpg_path, aadhaar_cropped_path)
                    if success:
                        temp_jpg_path = aadhaar_cropped_path
                    else:
                        print(f"⚠️ Aadhaar crop failed or not detected for {input_path.name} — will compress full image.")
            except Exception as e:
                print(f"❌ Aadhaar detection runtime error for {input_path.name}: {e}")

            # ✅ Signature Auto-Crop Applied Here (unchanged)
            if category == "signature":
                cropped_path = Path(tempfile.gettempdir()) / f"cropped_{uuid.uuid4().hex}_{relative_path.name}.jpg"
                if autocrop_signature_image(temp_jpg_path, cropped_path):
                    temp_jpg_path = cropped_path

            output_path = (processed_dir / relative_path).with_suffix(".jpg")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            status, final_kb = compress_jpg_to_target(temp_jpg_path, output_path, min_kb, max_kb)
            report_data.append([relative_path, category, f"{original_kb:.2f}", f"{final_kb:.2f}", status, "name"])

            try:
                if temp_jpg_path.exists():
                    temp_jpg_path.unlink()
            except Exception:
                pass

        else:
            output_path = processed_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(input_path, output_path)
            report_data.append([relative_path, 'document', f"{original_kb:.2f}", f"{original_kb:.2f}", 'copied_as_is', 'name'])

        progress_bar.progress((i + 1) / len(files_to_process))

    status_placeholder.text("Processing complete!")
    return report_data


# --- STREAMLIT UI (same as your version, untouched below) ---
st.set_page_config(page_title="Aiclex Media Processor", layout="wide")
st.title("🗂️ Aiclex Media Processor v1.4.1")
st.caption("Developed by Aiclex Technologies | aiclex.in")

if not PDF_SUPPORT:
    st.error("PDF processing is disabled. Install 'poppler-utils' for PDF support.")

uploaded_file = st.file_uploader("📦 Upload your .zip file", type=["zip"], help="Upload a ZIP containing all your documents")

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.get("base_name", ""):
        st.session_state.processed = False
        st.session_state.start_processing = False
        st.session_state.output_zip_path = None
        st.session_state.output_csv_path = None
        st.session_state.df_report = pd.DataFrame()
        st.session_state.base_name = uploaded_file.name

if uploaded_file is not None and not st.session_state.get("processed", False):
    if st.button("▶️ Process File"):
        st.session_state.start_processing = True

if uploaded_file is not None and st.session_state.get("start_processing", False):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        src_dir = temp_path / "source"
        out_dir = temp_path / "processed"
        src_dir.mkdir(); out_dir.mkdir()

        zip_path = temp_path / uploaded_file.name
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info(f"Processing '{uploaded_file.name}'...")
        status_box = st.empty()

        with st.spinner('Extracting ZIP...'):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(src_dir)
            extract_all_zips(src_dir, status_box)

        report_rows = process_tree(src_dir, out_dir, status_box)
        df = pd.DataFrame(report_rows, columns=["path", "category", "original_kb", "final_kb", "status", "decided_via"])

        out_dir_final = Path("processed_output")
        out_dir_final.mkdir(exist_ok=True)
        base_name = uploaded_file.name.replace('.zip', '')

        zip_out = out_dir_final / f"processed_{base_name}.zip"
        shutil.make_archive(str(zip_out.with_suffix('')), 'zip', out_dir)

        csv_out = out_dir_final / f"processing_report_{base_name}.csv"
        df.to_csv(csv_out, index=False)

        st.session_state.output_zip_path = zip_out
        st.session_state.output_csv_path = csv_out
        st.session_state.df_report = df
        st.session_state.processed = True
        st.session_state.start_processing = False

if st.session_state.get("processed", False):
    st.success("✅ All files processed successfully!")
    st.header("📊 Processing Report")
    st.dataframe(st.session_state.df_report)

    c1, c2 = st.columns(2)
    with open(st.session_state.output_zip_path, "rb") as fp:
        c1.download_button(
            label="📂 Download Processed ZIP",
            data=fp,
            file_name=f"processed_{st.session_state.base_name}.zip",
            mime="application/zip"
        )

    with open(st.session_state.output_csv_path, "rb") as fp:
        c2.download_button(
            label="📄 Download Report (CSV)",
            data=fp,
            file_name=f"processing_report_{st.session_state.base_name}.csv",
            mime='text/csv'
        )
