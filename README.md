# Aiclex Media Processor

A Streamlit-based media processing tool that automatically resizes, crops, and optimizes images including Aadhaar cards, signatures, photos, and qualification documents.

## Features

- 📦 Batch process ZIP files containing multiple images
- 🪪 Automatic Aadhaar card front-side detection and cropping
- ✍️ Signature auto-cropping with watermark removal
- 📸 Photo optimization
- 📄 Document processing (qualification proofs, etc.)
- 📊 Processing reports in CSV format

## Requirements

- Python 3.8+
- Tesseract OCR (for Aadhaar detection)
- Poppler (for PDF support, optional)

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

**⚠️ IMPORTANT**: Tesseract is required for Aadhaar card processing.

**Windows:**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install the executable
3. Add to PATH or update `TESSERACT_PATH` in the code (see `TESSERACT_SETUP.md`)

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt install tesseract-ocr
```

📖 **Detailed instructions**: See [TESSERACT_SETUP.md](TESSERACT_SETUP.md)

### 4. (Optional) Install Poppler for PDF Support

**Windows:** Download from http://blog.alivate.com.au/poppler-windows/

**macOS:**
```bash
brew install poppler
```

**Linux:**
```bash
sudo apt install poppler-utils
```

## Usage

1. **Run the Streamlit app:**
```bash
streamlit run streamlit_zip_processor_aiclex.py
```

2. **Upload a ZIP file** containing your documents
3. **Click "Process File"**
4. **Download** the processed ZIP and CSV report

## File Categories

The processor automatically categorizes files based on filename keywords:
- **ID Proof**: aadhaar, pan, passport, dl, etc.
- **Photo**: photo, pic, profile, etc.
- **Signature**: sign, signature, sig, etc.
- **Qualification Proof**: degree, certificate, marksheet, etc.

## Target Sizes

- ID Proof, Photo, Signature: 10-24 KB
- Qualification Proof: 50-200 KB
- Unknown: 10-500 KB

**Note**: Files already within the target size range are copied as-is without compression.

## Configuration

### Setting Tesseract Path (Windows)

If Tesseract is not in your system PATH, edit `streamlit_zip_processor_aiclex.py`:

Find line 21 and update:
```python
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

Replace with your actual Tesseract installation path.

## Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to https://share.streamlit.io/
3. Connect your repository
4. Deploy!

**Important**: Make sure `packages.txt` is in your repository root with:
```
tesseract-ocr
poppler-utils
```

Streamlit Cloud will automatically install these system packages.

📖 **Detailed deployment guide**: See [DEPLOYMENT.md](DEPLOYMENT.md)

## What happens if Tesseract is not installed?

The application will still work, but with limited functionality:
- ✅ File processing and compression will work
- ✅ Signature cropping will work
- ⚠️ Aadhaar front/back detection will be limited (relies on face detection only)
- ⚠️ OCR validation for Aadhaar cards will be disabled

For full functionality, especially Aadhaar card processing, Tesseract installation is **highly recommended**.

## Troubleshooting

### Tesseract not found

- See [TESSERACT_SETUP.md](TESSERACT_SETUP.md) for detailed troubleshooting
- Check app logs for specific error messages
- Verify Tesseract is installed and accessible

### Cloud deployment issues

- See [DEPLOYMENT.md](DEPLOYMENT.md) for cloud-specific troubleshooting
- Verify `packages.txt` exists and contains required packages
- Check deployment logs for errors

## Author

Aiclex Technologies (aiclex.in)

## License

[Add your license here]

