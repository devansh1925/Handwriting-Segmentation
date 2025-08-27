HANDWRITING MASKING PROJECT
============================

This project extracts handwritten text from images by masking out printed text regions.
It uses PaddleOCR for text detection and Tesseract for confidence scoring to distinguish
between printed and handwritten text.

PREREQUISITES
============

1. Python 3.7 or higher
2. Tesseract OCR engine (version 4.0 or higher, must be installed separately)

INSTALLATION
============

1. Install Tesseract OCR:
   
   **Windows:**
   - Download Tesseract 4.0+ from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install with default settings (usually C:\Program Files\Tesseract-OCR\)
   - Add Tesseract to your system PATH, or use --tesseract-cmd flag
   
   **macOS:**
   - Install via Homebrew: brew install tesseract
   - Or download from: https://github.com/tesseract-ocr/tesseract
   
   **Linux (Ubuntu/Debian):**
   - sudo apt-get update
   - sudo apt-get install tesseract-ocr
   - sudo apt-get install tesseract-ocr-eng (for English language)
   
   **Linux (CentOS/RHEL):**
   - sudo yum install tesseract
   - sudo yum install tesseract-langpack-eng
   
   **Verify Installation:**
   - Run: tesseract --version
   - Should show version 4.0 or higher

2. Install Python dependencies:
   
   **Using requirements.txt (recommended):**
   pip install -r requirements.txt
   
   **Or install manually:**
   pip install paddleocr==2.7.3 paddlepaddle==2.5.2 pytesseract==0.3.10 opencv-python==4.6.0.66 numpy==1.26.4 tqdm==4.66.4
   
   **Note:** These are the exact versions tested with this project. Using different versions may cause compatibility issues.

USAGE
=====

Basic usage:
python handwriting_mask.py --inputs image.jpg --outdir outputs

Process multiple images:
python handwriting_mask.py --inputs image1.jpg image2.jpg --outdir outputs

Process all images in a directory:
python handwriting_mask.py --inputs /path/to/images/ --outdir outputs

Process images using glob pattern:
python handwriting_mask.py --inputs "*.jpg" --outdir outputs

COMMAND LINE ARGUMENTS
=====================

--inputs: Image files, directories, or glob patterns (required)
--outdir: Output directory (default: "outputs")
--psm: Tesseract page segmentation mode (default: 6)
--use-angle-cls: Enable PaddleOCR angle classifier
--lang: Language for PaddleOCR (default: "en")
--gpu: Use GPU for PaddleOCR if available
--tesseract-cmd: Path to tesseract executable if not in PATH

EXAMPLES
========

# Process a single image
python handwriting_mask.py --inputs document.jpg --outdir results

# Process all JPG files in current directory
python handwriting_mask.py --inputs "*.jpg" --outdir results

# Process images with GPU acceleration
python handwriting_mask.py --inputs images/ --outdir results --gpu

# Use custom Tesseract path (Windows)
python handwriting_mask.py --inputs image.jpg --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"

# Process images in different language
python handwriting_mask.py --inputs image.jpg --lang es

OUTPUT FILES
============

For each input image, the script generates:
1. {filename}_handwriting_only.png - Image with printed text masked out
2. {filename}_overlay.png - Original image with printed regions highlighted in red

TROUBLESHOOTING
==============

1. "Tesseract not found" error:
   - Ensure Tesseract 4.0+ is installed on your system
   - Add Tesseract to your system PATH
   - Use --tesseract-cmd to specify the path to tesseract.exe
   - Windows example: --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
   - Verify installation: tesseract --version

2. "No module named 'paddleocr'" error:
   - Run: pip install paddleocr==2.7.3 paddlepaddle==2.5.2
   - Or use: pip install -r requirements.txt

3. Poor detection results:
   - Try different --psm values (3, 6, 8, 13)
   - Ensure images are clear and well-lit
   - Check if language setting matches your text

4. GPU not working:
   - Install CUDA and cuDNN for GPU support
   - Use --gpu flag only if you have compatible GPU

PERFORMANCE TIPS
===============

- Use --gpu flag if you have a compatible NVIDIA GPU
- Process images in batches for better efficiency
- Ensure sufficient RAM (at least 4GB recommended)
- For large images, consider resizing them first

SUPPORTED IMAGE FORMATS
======================

- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tif, .tiff)
- BMP (.bmp)
- WebP (.webp)
- PDF (.pdf) - Note: PDF processing may require additional setup

NOTES
=====

- The script uses confidence thresholds and heuristics to classify text
- Printed text is masked using estimated background color
- Results may vary depending on image quality and text characteristics
- For best results, use high-quality scans with good contrast

AUTHOR
======

This project is designed for extracting handwritten content from mixed documents
containing both printed and handwritten text.
