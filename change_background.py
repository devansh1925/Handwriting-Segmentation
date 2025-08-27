import cv2
import numpy as np
import os

def process_image(input_image_path):
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Could not load: {input_image_path}")
        return None

    # 1️⃣ Gaussian Blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 2️⃣ Sharpen
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, sharpening_kernel)

    # 3️⃣ Adaptive Thresholding
    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=10
    )

    # 4️⃣ Morphological Closing
    kernel = np.ones((6, 6), np.uint8)
    closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)

    # 5️⃣ Invert: black text on white
    result = cv2.bitwise_not(closed)

    return result

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            result = process_image(input_path)

            if result is not None:
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, result)
                print(f"✅ Saved: {output_path}")

# Run it:
process_folder('Dataset/only_handwritten', 'Dataset/only_handwritten_Output')