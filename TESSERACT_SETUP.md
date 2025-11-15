# How to Set Tesseract in PATH (Windows)

## Method 1: Install Tesseract and Add to PATH

### Step 1: Download Tesseract
1. Download Tesseract installer for Windows from:
   - **Official**: https://github.com/UB-Mannheim/tesseract/wiki
   - Direct download: https://digi.bib.uni-mannheim.de/tesseract/
   - Choose the latest version (e.g., `tesseract-ocr-w64-setup-5.x.x.exe`)

### Step 2: Install Tesseract
1. Run the installer
2. **Important**: During installation, note the installation path (usually `C:\Program Files\Tesseract-OCR`)
3. Complete the installation

### Step 3: Add to PATH (Windows 10/11)

#### Option A: Using System Properties (Recommended)
1. Press `Win + X` and select **System**
2. Click **Advanced system settings** (on the right)
3. Click **Environment Variables**
4. Under **System variables**, find and select **Path**, then click **Edit**
5. Click **New** and add: `C:\Program Files\Tesseract-OCR`
   - (Replace with your actual installation path if different)
6. Click **OK** on all dialogs
7. **Restart your terminal/IDE** for changes to take effect

#### Option B: Using Command Prompt (Admin)
1. Open Command Prompt as Administrator
2. Run:
   ```cmd
   setx /M PATH "%PATH%;C:\Program Files\Tesseract-OCR"
   ```
3. **Restart your terminal/IDE**

### Step 4: Verify Installation
Open a new Command Prompt or PowerShell and run:
```cmd
tesseract --version
```

You should see version information. If you get "command not found", PATH is not set correctly.

---

## Method 2: Set Tesseract Path in Code (Alternative)

If you can't add Tesseract to PATH, you can set it programmatically in your Python code.

### Find Your Tesseract Installation Path
Common locations:
- `C:\Program Files\Tesseract-OCR\tesseract.exe`
- `C:\Users\<YourUsername>\AppData\Local\Programs\Tesseract-OCR\tesseract.exe`

### Add to Your Code
Add this at the top of your Python file (after imports):

```python
import pytesseract

# Set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**Note**: Replace the path with your actual Tesseract installation path.

---

## Quick Test

After setting up, test in Python:
```python
import pytesseract

# If using Method 2, set path first:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    print(pytesseract.get_tesseract_version())
    print("✅ Tesseract is working!")
except Exception as e:
    print(f"❌ Error: {e}")
```

---

## Troubleshooting

### "tesseract is not installed or it's not in your PATH"
- Make sure Tesseract is installed
- Verify PATH is set correctly
- Restart your terminal/IDE after setting PATH
- Use Method 2 (set path in code) as a workaround

### "TesseractNotFoundError"
- Check the installation path
- Use Method 2 to set the path explicitly in code

### Still Not Working?
1. Find `tesseract.exe` on your system (use Windows Search)
2. Copy the full path
3. Use Method 2 to set it in your code

