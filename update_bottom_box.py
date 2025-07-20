import os
import cv2
import pytesseract

IMG_DIR = "trainingData/hsf_data/hsf_handwritten_boxes"
# Use a unique phrase from the NIST consent paragraph for detection
UNIQUE_PHRASE = "The National Institute of Standards and Technology (NIST) will include"

def is_nist_consent_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config="--psm 6")
    return UNIQUE_PHRASE in text

def main():
    deleted = 0
    for fname in os.listdir(IMG_DIR):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            fpath = os.path.join(IMG_DIR, fname)
            if is_nist_consent_image(fpath):
                os.remove(fpath)
                print(f"Deleted: {fname}")
                deleted += 1
    print(f"Deleted {deleted} NIST consent images.")

if __name__ == "__main__":
    main()