"""
Aadhaar Front Side Cropper
Automatically detects and crops the front side of an Aadhaar card from an input image.
Uses OpenCV for image processing and OCR only for validation.
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
import os
import sys
from pathlib import Path

# Optional: EasyOCR support (uncomment if you want to use EasyOCR instead of Tesseract)
# import easyocr
# EASYOCR_READER = easyocr.Reader(['en'])

# Aadhaar card dimensions (approximate aspect ratio)
AADHAAR_WIDTH_MM = 85.6
AADHAAR_HEIGHT_MM = 53.98
AADHAAR_ASPECT_RATIO = AADHAAR_WIDTH_MM / AADHAAR_HEIGHT_MM  # ~1.586

# Front side validation keywords
FRONT_KEYWORDS = [
    "government of india",
    "dob",
    "aadhaar",
    "aadhar",
    "adhar",
    "xxxx xxxx xxxx",
    "male",
    "female"
]

# Back side rejection keywords
BACK_KEYWORDS = [
    "address",
    "vid",
    "helpline",
    "qr code"
]


def preprocess_image(image):
    """
    Preprocess image for edge detection.
    Resize, blur, and apply threshold.
    
    Args:
        image: Input image (numpy array)
    
    Returns:
        Processed binary image
    """
    # Resize if too large (for faster processing)
    height, width = image.shape[:2]
    max_dimension = 2000
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive threshold for better edge detection
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Also try Otsu's threshold as alternative
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return thresh, thresh_otsu, gray, image


def find_card_contour(thresh, thresh_otsu, original_shape):
    """
    Find the biggest 4-corner contour that matches Aadhaar card aspect ratio.
    
    Args:
        thresh: Binary threshold image
        thresh_otsu: Otsu threshold image
        original_shape: Original image shape for aspect ratio calculation
    
    Returns:
        Contour points (4 corners) or None
    """
    contours_list = []
    
    # Try both threshold methods
    for threshold_img in [thresh, thresh_otsu]:
        # Apply morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area (must be reasonably large)
        min_area = original_shape[0] * original_shape[1] * 0.1  # At least 10% of image
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        for contour in large_contours:
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Check if it has 4 corners
            if len(approx) == 4:
                # Calculate aspect ratio
                rect = cv2.minAreaRect(approx)
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    # Aadhaar card aspect ratio is ~1.586, allow some tolerance
                    if 1.3 <= aspect_ratio <= 1.9:
                        contours_list.append((approx, cv2.contourArea(contour)))
    
    # If no 4-corner contour found, try to find rectangular contours and extract corners
    if not contours_list:
        for threshold_img in [thresh, thresh_otsu]:
            # Use Canny edge detection as alternative
            edges = cv2.Canny(cv2.bitwise_not(threshold_img), 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > original_shape[0] * original_shape[1] * 0.1:
                    # Get bounding rectangle
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    # Calculate aspect ratio
                    width, height = rect[1]
                    if width > 0 and height > 0:
                        aspect_ratio = max(width, height) / min(width, height)
                        if 1.3 <= aspect_ratio <= 1.9:
                            contours_list.append((box, area))
    
    # Return the largest valid contour
    if contours_list:
        contours_list.sort(key=lambda x: x[1], reverse=True)
        return contours_list[0][0]
    
    return None


def order_points(pts):
    """
    Order points in the order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts: Array of 4 points
    
    Returns:
        Ordered points
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and difference to find corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]  # top-left (smallest sum)
    rect[2] = pts[np.argmax(s)]  # bottom-right (largest sum)
    rect[1] = pts[np.argmin(diff)]  # top-right (smallest difference)
    rect[3] = pts[np.argmax(diff)]  # bottom-left (largest difference)
    
    return rect


def warp_card(image, contour):
    """
    Apply perspective transform to get a top-down view of the card.
    
    Args:
        image: Original image
        contour: 4 corner points of the card
    
    Returns:
        Warped card image
    """
    # Order the points
    pts = order_points(contour.reshape(4, 2))
    
    # Calculate dimensions of the new image
    (tl, tr, br, bl) = pts
    
    # Compute width and height
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points for perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(pts, dst)
    
    # Warp the image
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def validate_front_side(image):
    """
    Validate that the detected region is the FRONT side using OCR.
    
    Args:
        image: Cropped card image
    
    Returns:
        True if front side, False if back side or uncertain
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Enhance image for better OCR
    # Apply slight denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Try OCR with Tesseract
    try:
        # Use pytesseract
        text = pytesseract.image_to_string(denoised, lang='eng').lower()
        
        # Alternative: Use EasyOCR (uncomment if you prefer EasyOCR)
        # results = EASYOCR_READER.readtext(denoised)
        # text = ' '.join([result[1] for result in results]).lower()
        
    except Exception as e:
        print(f"⚠️ OCR error: {e}")
        # If OCR fails, assume it might be front (conservative approach)
        # You can change this to return False if you want strict validation
        return None
    
    # Check for back side keywords first (reject if found)
    for keyword in BACK_KEYWORDS:
        if keyword in text:
            print(f"❌ Back side detected (found keyword: '{keyword}')")
            return False
    
    # Check for front side keywords
    front_matches = 0
    for keyword in FRONT_KEYWORDS:
        if keyword in text:
            front_matches += 1
            print(f"✅ Front side keyword found: '{keyword}'")
    
    # If at least one front keyword found and no back keywords, it's front
    if front_matches > 0:
        return True
    
    # If no keywords found, return None (uncertain)
    return None


def load_image(image_path):
    """
    Load image from file, supporting multiple formats including PDF.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image as numpy array
    """
    path = Path(image_path)
    ext = path.suffix.lower()
    
    if ext == '.pdf':
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(str(image_path), dpi=200)
            if images:
                # Convert PIL to numpy array
                img_array = np.array(images[0])
                # Convert RGB to BGR for OpenCV
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                return img_array
            else:
                raise ValueError("No pages found in PDF")
        except ImportError:
            raise ImportError("pdf2image not installed. Install with: pip install pdf2image")
        except Exception as e:
            raise Exception(f"Failed to load PDF: {e}")
    else:
        # Load image using OpenCV
        image = cv2.imread(str(image_path))
        if image is None:
            # Try with PIL as fallback
            try:
                pil_image = Image.open(image_path)
                # Convert to RGB if needed
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                img_array = np.array(pil_image)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                return img_array
            except Exception as e:
                raise Exception(f"Failed to load image: {e}")
        return image


def crop_front_aadhaar(input_path, output_path="aadhar_front.jpg"):
    """
    Main function to crop the front side of Aadhaar card.
    
    Args:
        input_path: Path to input image file
        output_path: Path to save cropped front side
    
    Returns:
        True if successful, False otherwise
    """
    print(f"📄 Processing: {input_path}")
    
    # Load image
    try:
        original_image = load_image(input_path)
        if original_image is None:
            print("❌ Failed to load image")
            return False
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return False
    
    print(f"✅ Image loaded: {original_image.shape}")
    
    # Preprocess image
    print("🔄 Preprocessing image...")
    thresh, thresh_otsu, gray, processed_image = preprocess_image(original_image)
    
    # Find card contour
    print("🔍 Finding Aadhaar card contour...")
    contour = find_card_contour(thresh, thresh_otsu, original_image.shape)
    
    if contour is None:
        print("❌ Could not detect Aadhaar card rectangle")
        return False
    
    print(f"✅ Card contour found with {len(contour)} corners")
    
    # Warp card to get top-down view
    print("🔄 Applying perspective correction...")
    warped_card = warp_card(original_image, contour)
    
    print(f"✅ Card warped: {warped_card.shape}")
    
    # Validate front side
    print("🔍 Validating front side with OCR...")
    validation_result = validate_front_side(warped_card)
    
    if validation_result is False:
        print("❌ Back side detected. Rejecting.")
        return False
    elif validation_result is None:
        print("⚠️ Could not determine side (no keywords found). Assuming front side.")
        # You can change this to return False if you want strict validation
    else:
        print("✅ Front side validated!")
    
    # Save cropped front card
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert BGR to RGB for saving with PIL (better quality)
    warped_rgb = cv2.cvtColor(warped_card, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(warped_rgb)
    pil_image.save(str(output_path), "JPEG", quality=95)
    
    print(f"✅ Front Aadhaar card saved: {output_path}")
    return True


def main():
    """Main entry point."""
    # Find input file with name containing "aadhar" (case insensitive)
    current_dir = Path(".")
    input_file = None
    
    # Look for files with "aadhar" in the name
    for ext in ['.jpg', '.jpeg', '.png', '.pdf']:
        for file in current_dir.glob(f"*aadhar*{ext}"):
            input_file = file
            break
        for file in current_dir.glob(f"*aadhar*{ext.upper()}"):
            input_file = file
            break
        if input_file:
            break
    
    # Also check command line argument
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    
    if input_file is None or not input_file.exists():
        print("❌ No input file found.")
        print("Usage: python crop_front_aadhar.py [path_to_aadhar_image]")
        print("Or place a file with 'aadhar' in the name in the current directory.")
        return
    
    # Process
    success = crop_front_aadhaar(input_file, "aadhar_front.jpg")
    
    if not success:
        print("❌ Front Aadhaar side not detected.")
        sys.exit(1)
    else:
        print("✅ Successfully cropped front Aadhaar card!")


if __name__ == "__main__":
    main()

