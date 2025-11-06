import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# --- Define severity levels ---
def get_severity(cdr):
    if cdr < 0.4:
        return "Low"
    elif cdr < 0.7:
        return "Moderate"
    else:
        return "High"

# --- Main CDR estimation function ---
def estimate_cdr(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not read image.")
        return None, None

    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # --- Disc detection ---
    disc_thresh = cv2.adaptiveThreshold(enhanced, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        51, -10)
    contours, _ = cv2.findContours(disc_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("⚠️ No optic disc detected.")
        return None, None

    disc_contour = max(contours, key=cv2.contourArea)
    disc_area = cv2.contourArea(disc_contour)

    # --- Cup detection ---
    # Focus on the brightest central region within the disc
    mask = np.zeros_like(enhanced)
    cv2.drawContours(mask, [disc_contour], -1, 255, -1)
    disc_region = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    cup_region = cv2.GaussianBlur(disc_region, (9, 9), 0)

    # Use higher intensity threshold for cup
    _, cup_thresh = cv2.threshold(cup_region, 210, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(cup_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("⚠️ Cup not clearly visible — using fallback estimation.")
        cup_area = disc_area * 0.3  # fallback assumption
    else:
        cup_contour = max(contours, key=cv2.contourArea)
        cup_area = cv2.contourArea(cup_contour)

    # --- Compute CDR ---
    cdr = round(cup_area / disc_area, 2)
    severity = get_severity(cdr)

    # --- Draw contours for visualization ---
    vis = img.copy()
    cv2.drawContours(vis, [disc_contour], -1, (0, 255, 0), 2)
    if contours:
        cv2.drawContours(vis, [cup_contour], -1, (0, 0, 255), 2)
    cv2.putText(vis, f"CDR: {cdr}  Severity: {severity}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"CDR = {cdr} → Severity = {severity}")
    plt.axis('off')
    plt.show()

    return cdr, severity

# --- Image picker ---
def choose_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Fundus Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    return file_path

# --- Main ---
if __name__ == "__main__":
    print("🩺 Automatic Glaucoma Severity Estimation (Cup–Disc Ratio)")
    image_path = choose_image()
    if not image_path:
        print("⚠️ No image selected.")
    else:
        print(f"📂 Selected image: {image_path}")
        cdr, severity = estimate_cdr(image_path)
        if cdr is not None:
            print(f"\n✅ Result:\nCDR = {cdr}\nSeverity = {severity}")
        else:
            print("❌ Could not estimate CDR for this image.")
