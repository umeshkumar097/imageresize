# Deployment Guide

## Deploying to Streamlit Cloud

### Prerequisites
1. GitHub repository with your code
2. Streamlit Cloud account (free at https://streamlit.io/cloud)

### Step 1: Prepare Your Repository

Make sure your repository has:
- ✅ `streamlit_zip_processor_aiclex.py` (main app)
- ✅ `requirements.txt` (Python dependencies)
- ✅ `packages.txt` (system dependencies - Tesseract, Poppler)
- ✅ `README.md` (documentation)

### Step 2: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Connect your GitHub repository
4. Select your repository and branch
5. Set main file path: `streamlit_zip_processor_aiclex.py`
6. Click "Deploy"

### Step 3: System Dependencies

Streamlit Cloud will automatically install packages from `packages.txt`:
- `tesseract-ocr` - For OCR functionality
- `poppler-utils` - For PDF support

**Important**: The `packages.txt` file must be in the root of your repository.

### Step 4: Verify Deployment

After deployment:
1. Check the app logs for Tesseract status
2. You should see: `✅ Tesseract X.X found in PATH`
3. If you see warnings, check the logs for errors

## Troubleshooting Cloud Deployment

### Issue: "Tesseract OCR is not available"

**Solution**: 
1. Verify `packages.txt` exists in repository root
2. Verify it contains: `tesseract-ocr`
3. Redeploy the app (Streamlit Cloud will reinstall packages)

### Issue: App crashes on startup

**Check logs for**:
- Missing Python packages → Update `requirements.txt`
- Missing system packages → Update `packages.txt`
- Path errors → Check Tesseract detection code

### Issue: Tesseract not found in PATH

The code should auto-detect Tesseract on Linux. If not:
1. Check app logs for error messages
2. Verify `packages.txt` is correct
3. Try redeploying

## Alternative: Deploy on Other Platforms

### Heroku

1. Create `Procfile`:
```
web: streamlit run streamlit_zip_processor_aiclex.py --server.port=$PORT --server.address=0.0.0.0
```

2. Create `Aptfile` (for system packages):
```
tesseract-ocr
poppler-utils
```

3. Deploy using Heroku buildpacks:
```bash
heroku buildpacks:add heroku/python
heroku buildpacks:add --index 1 https://github.com/heroku/heroku-buildpack-apt
```

### Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_zip_processor_aiclex.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Verifying Tesseract Installation

After deployment, the app will show:
- ✅ `Tesseract OCR is available and configured` (if working)
- ⚠️ `Tesseract OCR is not available...` (if not installed)

Check the app logs in Streamlit Cloud dashboard for detailed Tesseract status.

