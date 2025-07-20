import cv2
import numpy as np
import pytesseract
import pandas as pd
import os

INPUT_ROOT = "trainingData/hsf_page/"
OUTPUT_DIR = "trainingData/hsf_data/hsf_handwritten_boxes"
CSV_PATH = "trainingData/hsf_data/hsf_labels.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def deskew(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    angle = 0.0
    if lines is not None:
        angles = []
        for rho, theta in lines[:,0]:
            angle_deg = (theta * 180 / np.pi) - 90
            if -45 < angle_deg < 45:
                angles.append(angle_deg)
        if angles:
            angle = np.median(angles)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

def find_boxes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,15,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h)
            area = w * h
            if 10000 < area < 40000 and 2.5 < aspect < 5.5:
                boxes.append((x, y, w, h))
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes

def find_bottom_big_box(img):
    """Detect the largest rectangular box near the bottom of the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,25,15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    candidates = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, bw, bh = cv2.boundingRect(approx)
            area = bw * bh
            # Heuristic: very large, near the bottom quarter of the image
            if area > 0.15 * w * h and y > h * 0.6:
                candidates.append((x, y, bw, bh, area))
    if not candidates:
        return None
    # Return the largest candidate
    candidates = sorted(candidates, key=lambda b: b[4], reverse=True)
    x, y, bw, bh, _ = candidates[0]
    return (x, y, bw, bh)

def group_boxes_into_grid(boxes, num_cols=4):
    rows = []
    boxes = sorted(boxes, key=lambda b: b[1])
    while boxes:
        row = [boxes.pop(0)]
        y0 = row[0][1]
        to_remove = []
        for i, b in enumerate(boxes):
            if abs(b[1] - y0) < 30:
                row.append(b)
                to_remove.append(i)
        for i in reversed(to_remove):
            boxes.pop(i)
        row = sorted(row, key=lambda b: b[0])
        rows.append(row)
    grid = [b for row in rows for b in row]
    return grid, rows

def safe_crop(img, x, y, w, h):
    img_h, img_w = img.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(x + w, img_w)
    y2 = min(y + h, img_h)
    if y2 > y1 and x2 > x1:
        return img[y1:y2, x1:x2]
    else:
        return np.zeros((1,1,3), dtype=np.uint8)

def is_box_empty(crop, border=4, ink_thresh=0.01):
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    h, w = gray.shape
    if h <= 2*border or w <= 2*border:
        return True
    inner = gray[border:h-border, border:w-border]
    _, bw = cv2.threshold(inner, 220, 255, cv2.THRESH_BINARY_INV)
    ink_pixels = np.count_nonzero(bw)
    total_pixels = bw.size
    return (ink_pixels / total_pixels) < ink_thresh

def extract_boxes_and_labels(img, boxes, rows, base_name, folder_name, csv_rows, label_height=45):
    for row_idx, row in enumerate(rows):
        for col_idx, box in enumerate(row):
            x, y, w, h = box
            handwritten_crop = safe_crop(img, x, y, w, h)
            # Dynamically expand label region above the box
            expand = int(0.2 * h)
            label_y = max(0, y - label_height - expand)
            label_h = label_height + expand
            label_region = safe_crop(img, x, label_y, w, label_h)
            # Preprocess label region for better OCR
            label_gray = cv2.cvtColor(label_region, cv2.COLOR_BGR2GRAY)
            label_gray = cv2.equalizeHist(label_gray)
            label_gray = cv2.adaptiveThreshold(label_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)
            # OCR with whitelist and single-line mode
            custom_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
            label = pytesseract.image_to_string(label_gray, config=custom_config)
            label = ''.join(filter(str.isalnum, label)).strip()
            fname = f"{folder_name}_{base_name}_r{row_idx}_c{col_idx}.png"
            out_path = os.path.join(OUTPUT_DIR, fname)
            cv2.imwrite(out_path, handwritten_crop)
            empty = is_box_empty(handwritten_crop)
            csv_rows.append({
                "IMAGE": fname,
                "MEDICINE_NAME": label,
                "GENERIC_NAME": "",
                "EMPTY": empty
            })

def extract_bottom_box(img, base_name, folder_name, csv_rows):
    bottom_box = find_bottom_big_box(img)
    if bottom_box is not None:
        x, y, w, h = bottom_box
        crop = safe_crop(img, x, y, w, h)
        # Preprocess for OCR
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_gray = cv2.equalizeHist(crop_gray)
        crop_gray = cv2.adaptiveThreshold(crop_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 15)
        # OCR with multi-line mode
        custom_config = r'--psm 6'
        text = pytesseract.image_to_string(crop_gray, config=custom_config).strip()
        fname = f"{folder_name}_{base_name}_bottom_box.png"
        out_path = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(out_path, crop)
        csv_rows.append({
            "IMAGE": fname,
            "MEDICINE_NAME": "BOTTOM_BOX",
            "GENERIC_NAME": text,
            "EMPTY": False
        })

def main():
    csv_rows = []
    for folder_name in sorted(os.listdir(INPUT_ROOT)):
        folder_path = os.path.join(INPUT_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            continue
        print(f"Processing folder: {folder_name}")
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, fname)
                img = cv2.imread(image_path)
                img = deskew(img)
                boxes = find_boxes(img)
                grid, rows = group_boxes_into_grid(boxes)
                base_name = os.path.splitext(fname)[0]
                extract_boxes_and_labels(img, grid, rows, base_name, folder_name, csv_rows)
                extract_bottom_box(img, base_name, folder_name, csv_rows)
    df = pd.DataFrame(csv_rows)
    df.to_csv(CSV_PATH, index=False, columns=["IMAGE", "MEDICINE_NAME", "GENERIC_NAME", "EMPTY"])
    print(f"Saved {len(csv_rows)} crops and labels to {CSV_PATH}")

if __name__ == "__main__":
    main()