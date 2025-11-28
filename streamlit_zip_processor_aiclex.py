# --- Aiclex Media Processor (Streamlit UI) ---
# Author: Aiclex Technologies (aiclex.in)
# Version: 1.4.1 (Signature Auto-Crop Enhanced + Quality Safe) + Aadhaar OCR-crop
# ---------------------------------------------------
#
# IMPORTANT: Tesseract OCR Installation Required
# - For Aadhaar card detection, Tesseract OCR must be installed
# - See TESSERACT_SETUP.md for installation instructions
# - For cloud deployment, see DEPLOYMENT.md
# - If Tesseract is not installed, update TESSERACT_PATH below (line 21)
# ---------------------------------------------------

import streamlit as st
import pandas as pd
import os
import zipfile
import shutil
import tempfile
import subprocess
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np
import pytesseract
import uuid

# --- Tesseract Path Configuration ---
# Set Tesseract path explicitly (Windows default)
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize Tesseract path and verify it works
TESSERACT_AVAILABLE = False

# Platform-specific detection
if os.name == 'nt':  # Windows
    # Windows paths
    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        try:
            version = pytesseract.get_tesseract_version()
            TESSERACT_AVAILABLE = True
            print(f"✅ Tesseract {version} configured at: {TESSERACT_PATH}")
        except Exception as e:
            print(f"⚠️ Tesseract path set but not working: {e}")
            TESSERACT_AVAILABLE = False
    else:
        # Try to auto-detect Tesseract path on Windows
        try:
            version = pytesseract.get_tesseract_version()
            TESSERACT_AVAILABLE = True
            print(f"✅ Tesseract {version} found in PATH")
        except Exception:
            common_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
            ]
            found = False
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    try:
                        version = pytesseract.get_tesseract_version()
                        TESSERACT_AVAILABLE = True
                        print(f"✅ Auto-detected Tesseract {version} at: {path}")
                        found = True
                        break
                    except Exception:
                        continue
            if not found:
                print("⚠️ Tesseract not found. OCR features will be disabled.")
else:  # Linux/macOS (for cloud deployment)
    # On Linux/Unix systems, Tesseract is usually in PATH
    # Common paths: /usr/bin/tesseract, /usr/local/bin/tesseract
    try:
        version = pytesseract.get_tesseract_version()
        TESSERACT_AVAILABLE = True
        print(f"✅ Tesseract {version} found in PATH")
    except Exception as e:
        print(f"⚠️ Tesseract not in PATH: {type(e).__name__}: {str(e)}")
        # Try common Linux/Unix paths
        unix_paths = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract',  # macOS Homebrew
        ]
        found = False
        for path in unix_paths:
            if os.path.exists(path):
                print(f"   Trying path: {path}")
                pytesseract.pytesseract.tesseract_cmd = path
                try:
                    version = pytesseract.get_tesseract_version()
                    TESSERACT_AVAILABLE = True
                    print(f"✅ Auto-detected Tesseract {version} at: {path}")
                    found = True
                    break
                except Exception as path_err:
                    print(f"   Path {path} exists but failed: {type(path_err).__name__}: {str(path_err)}")
                    continue
        
        # If still not found, try using 'which' or 'whereis' command
        if not found:
            print("   Attempting to find tesseract using system commands...")
            for cmd in ['which', 'whereis']:
                try:
                    result = subprocess.run([cmd, 'tesseract'], capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        tesseract_path = result.stdout.strip()
                        if cmd == 'whereis':
                            # whereis returns: tesseract: /usr/bin/tesseract
                            parts = tesseract_path.split()
                            if len(parts) > 1:
                                tesseract_path = parts[1]
                            else:
                                continue
                        
                        if os.path.exists(tesseract_path):
                            print(f"   Found tesseract at: {tesseract_path} (via {cmd})")
                            pytesseract.pytesseract.tesseract_cmd = tesseract_path
                            try:
                                version = pytesseract.get_tesseract_version()
                                TESSERACT_AVAILABLE = True
                                print(f"✅ Tesseract {version} found via '{cmd}' command")
                                found = True
                                break
                            except Exception as cmd_err:
                                print(f"   Command found path but failed: {type(cmd_err).__name__}: {str(cmd_err)}")
                                continue
                except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as cmd_err:
                    continue
        
        if not found:
            print("⚠️ Tesseract not found. OCR features will be disabled.")
            print("   On Linux, install with: sudo apt install tesseract-ocr")
            print("   On macOS, install with: brew install tesseract")
            print("   For Streamlit Cloud, ensure packages.txt contains: tesseract-ocr") 


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
    "id_proof": (10, 20),
    "photo": (10, 20),
    "signature": (10, 20),
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
PHOTO_MAX_DIM = 1600
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


def shrink_photo_if_needed(image_path, max_dim=PHOTO_MAX_DIM):
    """
    Downscale oversized photos before compression to avoid heavy quality loss.
    """
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size
            longer_side = max(width, height)
            if longer_side <= max_dim:
                return False

            scale = max_dim / float(longer_side)
            new_size = (
                max(1, int(width * scale)),
                max(1, int(height * scale))
            )
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            resized.save(image_path, "JPEG", quality=95, optimize=True)
            return True
    except Exception as exc:
        print(f"⚠️ Photo resize skipped ({image_path}): {exc}")
        return False


def clean_photo_border_artifacts(image_path, source_name=None):
    """
    Trim uniform scanner borders or text strips around photographs.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_h, img_w = img.shape[:2]
        min_trim = max(img_w, img_h) * 0.01

        trimmed = False
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            trim_left = x
            trim_top = y
            trim_right = img_w - (x + w)
            trim_bottom = img_h - (y + h)
            if (
                w >= img_w * 0.5
                and h >= img_h * 0.5
                and max(trim_left, trim_top, trim_right, trim_bottom) >= min_trim
            ):
                pad = 4
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(img_w, x + w + pad)
                y2 = min(img_h, y + h + pad)
                img = img[y1:y2, x1:x2]
                gray = gray[y1:y2, x1:x2]
                img_h, img_w = img.shape[:2]
                trimmed = True

        if trimmed:
            _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

        bottom_band = mask[int(mask.shape[0] * 0.7) :, :]
        band_ratio = np.mean(bottom_band > 0)
        if band_ratio > 0.02:
            rows = np.where(np.sum(mask > 0, axis=1) > mask.shape[1] * 0.02)[0]
            if rows.size > 0:
                cutoff = rows[-1]
                crop_limit = max(cutoff - 5, int(img_h * 0.5))
                if crop_limit < img_h - 1:
                    img = img[:crop_limit, :]
                    trimmed = True

        if trimmed:
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(
                image_path, "JPEG", quality=95, optimize=True
            )
            display_name = source_name or Path(image_path).name
            print(f"ℹ️ Photo border trimmed for original file: {display_name}")
            return True
        return False
    except Exception as exc:
        print(f"⚠️ Photo border clean skipped ({image_path}): {exc}")
        return False


# Improved signature auto-crop logic
def autocrop_signature_image(image_path, save_path):
    """
    Robust auto-cropping for signature images, ignoring watermark text at bottom.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ Signature crop failed (unable to read image): {image_path}")
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        inverted = cv2.bitwise_not(enhanced)

        def build_mask():
            adaptive = cv2.adaptiveThreshold(
                inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5
            )
            _, otsu = cv2.threshold(
                inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            edges = cv2.Canny(enhanced, 35, 150)
            mask_local = cv2.bitwise_or(adaptive, otsu)
            mask_local = cv2.bitwise_or(mask_local, edges)
            return mask_local

        def refine_mask(mask_local):
            h, w = mask_local.shape
            ignore_bottom = int(h * 0.18)
            ignore_top = int(h * 0.05)
            mask_local[h - ignore_bottom :, :] = 0
            mask_local[:ignore_top, :] = 0
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_local = cv2.morphologyEx(mask_local, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask_local = cv2.dilate(mask_local, kernel, iterations=1)
            return mask_local

        def best_bbox_from_mask(mask_local, img_shape):
            contours, _ = cv2.findContours(mask_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            h, w = img_shape[:2]
            total_area = w * h
            candidates = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < total_area * 0.0005:  # Lower threshold for better detection
                    continue
                x, y, ww, hh = cv2.boundingRect(cnt)
                aspect = ww / max(1, hh)
                # Accept wider aspect ratios (signatures are typically wide)
                if aspect < 1.5:
                    continue
                margin = min(x, y, w - (x + ww), h - (y + hh))
                # Score by area and aspect ratio (prefer wider signatures)
                score = area * (1 + min(aspect / 2, 8))
                candidates.append((score, (x, y, ww, hh), margin))

            if not candidates:
                # Merge all contours and get tight bounding box
                merged = np.vstack(contours)
                return cv2.boundingRect(merged)

            candidates.sort(key=lambda c: c[0], reverse=True)
            # Get the best candidate and refine it to be tighter
            best_bbox = candidates[0][1]
            x, y, ww, hh = best_bbox
            
            # If multiple candidates, try to merge nearby ones for a tighter box
            if len(candidates) > 1:
                merged_points = []
                for score, (cx, cy, cww, chh), _ in candidates[:3]:  # Top 3 candidates
                    merged_points.extend([(cx, cy), (cx + cww, cy), (cx, cy + chh), (cx + cww, cy + chh)])
                if merged_points:
                    merged_points = np.array(merged_points)
                    x = int(np.min(merged_points[:, 0]))
                    y = int(np.min(merged_points[:, 1]))
                    x2 = int(np.max(merged_points[:, 0]))
                    y2 = int(np.max(merged_points[:, 1]))
                    return (x, y, x2 - x, y2 - y)
            
            return best_bbox

        mask = refine_mask(build_mask())
        bbox = best_bbox_from_mask(mask, img.shape)

        if bbox is None:
            blur_mask = refine_mask(cv2.GaussianBlur(mask, (5, 5), 0))
            bbox = best_bbox_from_mask(blur_mask, img.shape)

        if bbox is None:
            projection = np.sum(mask > 0, axis=1)
            threshold = mask.shape[1] * 0.02
            rows = np.where(projection > threshold)[0]
            if rows.size > 0:
                y1, y2 = rows[0], rows[-1]
                x_projection = np.sum(mask > 0, axis=0)
                cols = np.where(x_projection > mask.shape[0] * 0.02)[0]
                if cols.size > 0:
                    x1, x2 = cols[0], cols[-1]
                    bbox = (x1, y1, x2 - x1, y2 - y1)

        if bbox is None:
            _, simple_thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
            coords = cv2.findNonZero(simple_thresh)
            if coords is not None:
                bbox = cv2.boundingRect(coords)
                print(f"ℹ️ Signature fallback bbox applied (no watermark cues): {image_path}")

        if bbox is None:
            print(f"❌ Signature crop failed (no contours detected): {image_path}")
            return False

        x, y, ww, hh = bbox
        h_img, w_img = img.shape[:2]
        
        # Refine bounding box by analyzing content density for tighter crop
        # Extract the detected region and find the tightest box within it
        try:
            gray_crop = gray[y:min(y+hh, h_img), x:min(x+ww, w_img)]
            if gray_crop.size > 0:
                # Use adaptive threshold to find signature pixels
                _, refine_mask = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                # Find the tightest box within the detected region
                rows = np.any(refine_mask > 0, axis=1)
                cols = np.any(refine_mask > 0, axis=0)
                if rows.any() and cols.any():
                    y_min, y_max = np.where(rows)[0][[0, -1]]
                    x_min, x_max = np.where(cols)[0][[0, -1]]
                    # Adjust original bbox to the tighter region
                    x = x + x_min
                    y = y + y_min
                    ww = x_max - x_min + 1
                    hh = y_max - y_min + 1
        except Exception:
            # If refinement fails, use the original bbox
            pass
        
        # Tight padding: use small fixed padding or small percentage of signature size (whichever is smaller)
        # This ensures the crop fits tightly around the signature box with minimal padding
        pad_x = min(8, max(3, int(ww * 0.05)))  # 5% of signature width, but max 8px, min 3px
        pad_y = min(8, max(3, int(hh * 0.05)))  # 5% of signature height, but max 8px, min 3px
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + ww + pad_x)
        y2 = min(h_img, y + hh + pad_y)

        # Validate minimum dimensions
        min_height = h_img * 0.03
        min_width = w_img * 0.10
        if (y2 - y1) < min_height or (x2 - x1) < min_width:
            print(f"❌ Signature crop failed (bounding box too small): {image_path}")
            return False

        cropped = img[y1:y2, x1:x2]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        Image.fromarray(cropped).save(save_path, "JPEG", quality=95)

        print(f"✅ Signature cropped precisely (box: {x1},{y1}→{x2},{y2}, padding: {pad_x}x{pad_y}px)")
        return True

    except Exception as e:
        print(f"❌ Signature auto-crop runtime error for {image_path}: {e}")
        return False


# --- Aadhaar helpers to be added carefully (do not disturb other logic) ---

import cv2
import pytesseract
import numpy as np
from PIL import Image
import os

# --- Aadhaar Front Detector ---
def detect_and_crop_aadhaar_front(image_path, save_path, source_name=None):
    """
    Detect and crop Aadhaar front side intelligently.
    Uses face detection to locate and crop the front side of Aadhaar card.
    """

    # ✅ Step 0: Quick filename-based Aadhaar check
    filename_path = Path(source_name) if source_name else Path(image_path)
    display_name = filename_path.name
    filename_lower = display_name.lower()
    aadhaar_file_keywords = ["aadhaar", "aadhar", "adhar"]
    aadhaar_in_name = any(k in filename_lower for k in aadhaar_file_keywords)

    # --- Step 1: Load and preprocess image ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Failed to read image: {image_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocess image to improve face detection (handle blur and enhance contrast)
    # Apply sharpening to reduce blur effect
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(sharpened)
    
    # Use both original and enhanced for detection
    gray_for_detection = enhanced

    # --- Step 2: Optional OCR validation (only if available) ---
    front_detected = False
    if TESSERACT_AVAILABLE:
        try:
            text = pytesseract.image_to_string(gray).lower()
            front_keywords = [
                "government of india",
                "unique identification",
                "aadhaar",
                "aadhar",
                "adhar",
                "dob",
                "date of birth",
                "male",
                "female"
            ]
            front_detected = any(k in text for k in front_keywords)
            if front_detected:
                print("✅ Aadhaar FRONT side detected by OCR.")
        except Exception as e:
            print(f"⚠️ OCR validation skipped: {e}")

    # If filename doesn't indicate Aadhaar and OCR doesn't detect front, skip
    if not aadhaar_in_name and not front_detected:
        print(f"⚠️ Aadhaar not detected in filename or OCR for '{display_name}'. Skipping Aadhaar crop.")
        return False

    # --- Step 3: Face detection to locate photo region (front side has face) ---
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Try multiple detection parameters for better results
    # First try with enhanced image and relaxed parameters
    faces1 = face_cascade.detectMultiScale(gray_for_detection, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50), maxSize=(300, 300))
    # Also try with original image
    faces2 = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(50, 50), maxSize=(300, 300))
    
    # Combine and deduplicate faces
    all_faces = []
    seen_faces = set()
    
    for face in list(faces1) + list(faces2):
        x, y, w, h = face
        # Create a unique key for this face region
        face_key = (x // 20, y // 20, w // 20, h // 20)  # Round to avoid duplicates
        if face_key not in seen_faces:
            seen_faces.add(face_key)
            all_faces.append((x, y, w, h))
    
    if len(all_faces) > 0:
        # Choose the best face: prefer larger faces, and faces in lower portion (actual Aadhaar card)
        img_height = img.shape[0]
        
        def face_score(face):
            x, y, w, h = face
            area = w * h
            # Prefer faces in lower 70% of image (where actual Aadhaar card usually is)
            position_score = 1.0 if y > img_height * 0.3 else 0.5
            # Prefer larger faces (actual photo, not small preview)
            size_score = area / (img.shape[0] * img.shape[1])
            return size_score * position_score
        
        # Sort by score and pick the best one
        all_faces.sort(key=face_score, reverse=True)
        best_face = all_faces[0]
        (x, y, w, h) = best_face
        
        if len(all_faces) > 1:
            print(f"ℹ️ Aadhaar file '{display_name}' has multiple faces ({len(all_faces)}). Selected largest/clearest face for cropping.")
        
        # Expand around face region with increased margins (front side has face on left)
        x1 = max(x - 150, 0)  # Increased left margin from 80 to 150
        y1 = max(y - 200, 0)  # Increased top margin from 120 to 200
        x2 = min(x + w + 600, img.shape[1])  # Increased right margin from 400 to 600
        y2 = min(y + h + 300, img.shape[0])  # Increased bottom margin from 200 to 300
        crop = img[y1:y2, x1:x2]
        print(f"✅ Aadhaar face detected for '{display_name}' (size: {w}x{h}) - cropping around face region with increased margins.")
    else:
        # Fallback: crop left portion with increased margins (front side typically has photo on left)
        h, w = img.shape[:2]
        # Front side: photo on left, crop with more margin (increased from 5%-65% to 3%-70% width, 5%-95% height)
        crop = img[int(h * 0.05):int(h * 0.95), int(w * 0.03):int(w * 0.70)]
        print(f"⚠️ No face detected for Aadhaar '{display_name}' - cropping left portion with fallback margins.")

    # --- Step 4: Save cropped Aadhaar front ---
    im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    im.save(save_path, "JPEG", quality=90)

    print(f"✅ Cropped Aadhaar front saved: {save_path}")
    return True



# --- Compression ---
def compress_jpg_to_target(input_path, output_path, min_kb, max_kb, category=None):
    """Compress JPEG to target size range."""
    try:
        img = Image.open(input_path)
        img_copy = img.copy()
        width, height = img_copy.size

        def save_and_check(pil_img, quality):
            pil_img.save(output_path, 'JPEG', quality=quality, optimize=True)
            return output_path.stat().st_size / 1024

        if category == "photo":
            photo_scales = [1.0, 0.9, 0.8, 0.7, 0.6]
            photo_qualities = [90, 85, 80, 75, 70]
            for scale in photo_scales:
                if scale == 1.0:
                    working_img = img_copy
                else:
                    new_size = (
                        max(1, int(width * scale)),
                        max(1, int(height * scale))
                    )
                    working_img = img_copy.resize(new_size, Image.Resampling.LANCZOS)
                for quality in photo_qualities:
                    size_kb = save_and_check(working_img, quality)
                    if min_kb <= size_kb <= max_kb:
                        return "ok", size_kb

        for quality in range(95, MIN_QUALITY - 1, -5):
            size_kb = save_and_check(img_copy, quality)
            if min_kb <= size_kb <= max_kb:
                return "ok", size_kb

        for scale in [0.9, 0.75, 0.5, 0.25]:
            new_size = (
                max(1, int(width * scale)),
                max(1, int(height * scale))
            )
            img_resized = img_copy.resize(new_size, Image.Resampling.LANCZOS)
            for quality in range(90, MIN_QUALITY - 1, -10):
                size_kb = save_and_check(img_resized, quality)
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
                report_data.append([str(relative_path), 'unknown', f"{original_kb:.2f}", 0, "convert_failed", "name"])
                continue

            category = guess_category_by_filename(input_path.name)
            min_kb, max_kb = TARGET_SIZES_KB.get(category, TARGET_SIZES_KB["unknown"])

            # ✅ Aadhaar OCR crop if filename indicates Aadhaar
            try:
                if is_aadhaar_filename(input_path.name):
                    aadhaar_cropped_path = Path(tempfile.gettempdir()) / f"aadhaar_{uuid.uuid4().hex}_{relative_path.name}.jpg"
                    success = detect_and_crop_aadhaar_front(
                        temp_jpg_path, aadhaar_cropped_path, source_name=str(relative_path)
                    )
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
                else:
                    print(f"⚠️ Signature auto-crop skipped for {input_path.name}; continuing with original image.")

            if category == "photo":
                clean_photo_border_artifacts(temp_jpg_path, source_name=str(relative_path))
                shrink_photo_if_needed(temp_jpg_path)

            output_path = (processed_dir / relative_path).with_suffix(".jpg")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file is already within target size range (after any cropping)
            current_size_kb = temp_jpg_path.stat().st_size / 1024
            
            if min_kb <= current_size_kb <= max_kb:
                # File is already within target size - copy as-is without compression
                shutil.copy(temp_jpg_path, output_path)
                final_kb = output_path.stat().st_size / 1024
                status = "copied_as_is"
                report_data.append([str(relative_path), category, f"{original_kb:.2f}", f"{final_kb:.2f}", status, "name"])
            else:
                # File is outside target size - compress to target
                status, final_kb = compress_jpg_to_target(temp_jpg_path, output_path, min_kb, max_kb, category=category)
                report_data.append([str(relative_path), category, f"{original_kb:.2f}", f"{final_kb:.2f}", status, "name"])

            try:
                if temp_jpg_path.exists():
                    temp_jpg_path.unlink()
            except Exception:
                pass

        else:
            output_path = processed_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(input_path, output_path)
            report_data.append([str(relative_path), 'document', f"{original_kb:.2f}", f"{original_kb:.2f}", 'copied_as_is', 'name'])

        progress_bar.progress((i + 1) / len(files_to_process))

    status_placeholder.text("Processing complete!")
    return report_data


# --- STREAMLIT UI (same as your version, untouched below) ---
st.set_page_config(page_title="Aiclex Media Processor", layout="wide")
st.title("🗂️ Aiclex Media Processor v1.4.1")
st.caption("Developed by Aiclex Technologies | aiclex.in")

if not PDF_SUPPORT:
    st.error("PDF processing is disabled. Install 'poppler-utils' for PDF support.")

# Show Tesseract OCR status
if TESSERACT_AVAILABLE:
    st.success("✅ Tesseract OCR is available and configured")
else:
    st.warning("⚠️ Tesseract OCR is not available. Aadhaar front/back validation will be limited. Install Tesseract for full OCR support.")

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
    banner = f"\n{'=' * 20} NEW FILE RUN: {uploaded_file.name} {'=' * 20}\n"
    print(banner)
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

    c1, c2, c3 = st.columns(3)
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
    
    if c3.button("🔄 Reprocess File", help="Reprocess the uploaded file"):
        st.session_state.processed = False
        st.session_state.start_processing = True
        st.rerun()
