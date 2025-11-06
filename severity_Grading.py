import os
import cv2
import numpy as np
import pandas as pd
from skimage import morphology
from skimage.filters import threshold_otsu

# ---------- SETTINGS ----------
dataset_path = r"C:\Users\deswa\Downloads\glaucoma\RIM-ONE_DL_images\partitioned_by_hospital\test_set\glaucoma"
output_csv = r"C:\Users\deswa\OneDrive\Desktop\capstone\glaucoma_severity_results.csv"

print("🩺 Automatic Glaucoma Severity Estimation (Cup–Disc Ratio)\n")

# ---------- HELPER FUNCTION ----------
def estimate_cdr(image_path):
    """Estimate cup–disc ratio (CDR) and severity level from a fundus image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # --- Detect optic disc ---
    try:
        thresh_disc = threshold_otsu(gray)
        disc_mask = gray > thresh_disc * 0.9
        disc_mask = morphology.remove_small_objects(disc_mask, 400)
        disc_mask = morphology.binary_closing(disc_mask, morphology.disk(5))
    except Exception:
        print("⚠️ No optic disc detected.")
        return None

    if disc_mask.sum() == 0:
        print("⚠️ No optic disc detected.")
        return None

    # --- Adaptive CLAHE + local threshold for cup ---
    try:
        disc_region = gray * disc_mask
        disc_norm = cv2.normalize(disc_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(disc_norm)

        # adaptive threshold (locally sensitive)
        cup_mask = cv2.adaptiveThreshold(enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, -8)

        
        cup_mask = morphology.remove_small_objects(cup_mask.astype(bool), 100)
        cup_mask = morphology.binary_opening(cup_mask, morphology.disk(3))
    except Exception:
        print("⚠️ No cup detected.")
        return None

    if cup_mask.sum() == 0:
        print("⚠️ No cup detected.")
        return None

    # --- Compute cup–disc ratio ---
    disc_area = np.sum(disc_mask)
    cup_area = np.sum(cup_mask)
    cdr = min(cup_area / disc_area, 1.0)

    # --- Severity grading ---
    if cdr < 0.3:
        severity = "Low"
    elif cdr < 0.6:
        severity = "Mid"
    else:
        severity = "High"

    print(f"🖼️ {os.path.basename(image_path)} → CDR: {cdr:.2f} → Severity: {severity}")
    return cdr, severity


# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    image_paths = [
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if not image_paths:
        print("⚠️ No images found in dataset folder.")
    else:
        print(f"📂 Found {len(image_paths)} images for severity grading.\n")

    results = []
    for img_path in image_paths:
        res = estimate_cdr(img_path)
        if res is not None:
            cdr, severity = res
            results.append({
                "Image": os.path.basename(img_path),
                "CDR": round(cdr, 3),
                "Severity": severity
            })

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Results saved to: {output_csv}")
    else:
        print("⚠️ No valid results computed.")
