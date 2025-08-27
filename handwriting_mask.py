#!/usr/bin/env python3
"""
handwriting_mask.py

Route-B pipeline to **mask printed text** and keep **handwriting**:

1) Text detection with PaddleOCR (DBNet/CRAFT under the hood)
2) Printed-vs-handwritten decision using Tesseract confidence
3) Create & save a printed-text mask + handwriting-only image

Usage:
    python handwriting_mask.py \
        --inputs /path/to/img_or_dir1 /path/to/img_or_dir2 ... \
        --outdir outputs \
        --conf-thres 80

Requirements:
    pip install paddleocr paddlepaddle pytesseract opencv-python numpy tqdm
    (and install Tesseract binary on your system)
"""

import argparse
import os
import sys
import glob
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import pytesseract
from pytesseract import Output
from paddleocr import PaddleOCR


# ---------- HEURISTIC CHECKER ----------
common_keywords = ["hospital", "doctor", "rx", "date", "phone", "reg", "dept", "dr.", "age", "sex", "name", "city", "drug", "illness", "allergy", "email", "gender", "DR", "MR", "MS", "MRS", "MISS", "DR.", "MR.", "MS.", "MRS.", "MISS.", "nav"]

def is_computer_generated(text, bbox, angle):
    text_lower = text.lower()
    # Allow just one keyword match
    if any(keyword in text_lower for keyword in common_keywords):
        return True

    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox
    width = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
    height = np.linalg.norm(np.array([x1, y1]) - np.array([x4, y4]))
    if width == 0 or height == 0:
        return False
    aspect_ratio = width / height
    # Lower aspect ratio
    if aspect_ratio > 6:
        return True

    # Wider height range
    if 12 <= height <= 50:
        return True

    return False


def list_images(paths):
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for ext in exts:
                files.extend(glob.glob(str(p / f"*{ext}")))
        elif p.is_file():
            if p.suffix.lower() in exts:
                files.append(str(p))
        else:
            # glob pattern
            files.extend(glob.glob(str(p)))
    # remove duplicates, keep order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def crop_from_polygon(img, polygon):
    """
    Simple & fast: use bounding rectangle of the polygon.
    (You can replace with a perspective-warp crop if needed.)
    """
    pts = np.array(polygon, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    return img[y:y + h, x:x + w], (x, y, w, h)


def estimate_background_color(img, polygon, margin=5):
    """
    Estimate the background color around the polygon by sampling pixels just outside its bounding rectangle.
    Returns a tuple (B, G, R) or (gray,) depending on image type.
    """
    pts = np.array(polygon, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    h_img, w_img = img.shape[:2]
    # Expand the rectangle by margin, but keep within image bounds
    x0 = max(x - margin, 0)
    y0 = max(y - margin, 0)
    x1 = min(x + w + margin, w_img)
    y1 = min(y + h + margin, h_img)
    # Mask for the expanded region
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    # Draw the original polygon (shifted) as 1, then invert to get outside
    shifted_pts = pts - [x0, y0]
    cv2.fillPoly(mask, [shifted_pts], 1)
    mask = 1 - mask  # outside polygon is 1
    # Get pixels from the expanded region, outside the polygon
    region = img[y0:y1, x0:x1]
    if len(region.shape) == 2:
        pixels = region[mask == 1]
        if len(pixels) == 0:
            return 255  # fallback to white
        return int(np.median(pixels))
    else:
        pixels = region[mask == 1]
        if len(pixels) == 0:
            return (255, 255, 255)
        return tuple([int(np.median(pixels[:, i])) for i in range(region.shape[2])])


def tesseract_max_confidence(crop, psm=6):
    """
    Returns the maximum word-level confidence for the crop.
    """
    if crop.size == 0:
        return 0
    # Convert to RGB if needed
    if len(crop.shape) == 2:
        rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    data = pytesseract.image_to_data(
        rgb,
        output_type=Output.DICT,
        config=f'--psm {psm}'
    )
    confs = [int(c) for c in data.get('conf', []) if c != '-1']
    return max(confs) if confs else 0


def mask_printed_text(img, detections):
    """
    detections: PaddleOCR detections for a single page (results[0])
    Returns:
        mask (uint8), handwriting_only_img (BGR), printed_boxes_count, handwritten_boxes_count
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    printed, handwritten = 0, 0

    # Prepare handwriting_only as a copy of img
    handwriting_only = img.copy()

    for det in detections:
        polygon = det[0]  # 4-point polygon [[x1,y1], ...]
        text = det[1][0] if len(det) > 1 and len(det[1]) > 0 else ""
        angle = det[2] if len(det) > 2 else 0
        crop, _ = crop_from_polygon(img, polygon)
        conf = tesseract_max_confidence(crop)

        # More inclusive thresholds and permissive heuristic
        if conf >= 85:
            printed += 1
            cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
            bg_color = estimate_background_color(img, polygon)
            cv2.fillPoly(handwriting_only, [np.array(polygon, dtype=np.int32)], bg_color)
        elif 75 <= conf < 85 and is_computer_generated(text, polygon, angle):
            printed += 1
            cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
            bg_color = estimate_background_color(img, polygon)
            cv2.fillPoly(handwriting_only, [np.array(polygon, dtype=np.int32)], bg_color)
        else:
            handwritten += 1

    return mask, handwriting_only, printed, handwritten


def main():
    parser = argparse.ArgumentParser(description="Mask printed text using PaddleOCR + Tesseract confidence.")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Image files, directories or glob patterns.")
    parser.add_argument("--outdir", type=str, default="outputs",
                        help="Directory to save results.")
    parser.add_argument("--psm", type=int, default=6,
                        help="Tesseract page segmentation mode used on each crop.")
    parser.add_argument("--use-angle-cls", action="store_true",
                        help="Enable PaddleOCR angle classifier.")
    parser.add_argument("--lang", type=str, default="en",
                        help="Language for PaddleOCR (default: en)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for PaddleOCR if available.")
    parser.add_argument("--tesseract-cmd", type=str, default="",
                        help="Path to tesseract executable if not in PATH.")
    args = parser.parse_args()

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    os.makedirs(args.outdir, exist_ok=True)

    # Initialize PaddleOCR
    print("[*] Loading PaddleOCR...")
    ocr = PaddleOCR(
        use_angle_cls=args.use_angle_cls,
        lang=args.lang,
        use_gpu=args.gpu
    )

    files = list_images(args.inputs)
    if not files:
        print("No images found. Check your --inputs.", file=sys.stderr)
        sys.exit(1)

    for f in tqdm(files, desc="Processing"):
        img = cv2.imread(f)
        if img is None:
            print(f"[!] Could not read {f}, skipping.", file=sys.stderr)
            continue

        # Run detection (and recognition ignored, we only need boxes)
        results = ocr.ocr(f, cls=args.use_angle_cls)
        if not results or len(results[0]) == 0:
            print(f"[!] No text detected in {f}")
            continue

        mask, handwriting_only, printed_count, handwritten_count = mask_printed_text(
            img, results[0]
        )

        base = Path(f).stem
        hw_path = os.path.join(args.outdir, f"{base}_handwriting_only.png")
        overlay_path = os.path.join(args.outdir, f"{base}_overlay.png")

        # Save results
        cv2.imwrite(hw_path, handwriting_only)

        # Optional overlay visualization
        overlay = img.copy()
        overlay[mask == 255] = (0, 0, 255)  # mark printed areas red
        alpha = 0.35
        vis = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
        cv2.imwrite(overlay_path, vis)

        print(f"[+] {f}: printed={printed_count}, handwritten={handwritten_count}")
        print(f"    -> {hw_path}")
        print(f"    -> {overlay_path}")

    print("Done.")


if __name__ == "__main__":
    main()
