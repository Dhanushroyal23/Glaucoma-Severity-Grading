import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# --- Configuration ---
dataset_path = r"C:\Users\deswa\Downloads\glaucoma\RIM-ONE_DL_images\partitioned_by_hospital"
img_size = 224
apply_clahe = True

X = []
y = []
image_paths = []

# --- Helper: CLAHE contrast enhancement ---
def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

# --- Load all images ---
for subset in ["training_set", "test_set"]:
    subset_path = os.path.join(dataset_path, subset)
    if not os.path.exists(subset_path):
        continue

    print(f"\n📂 Processing subset: {subset}")

    for label_name in ["glaucoma", "normal"]:
        class_path = os.path.join(subset_path, label_name)
        if not os.path.exists(class_path):
            print(f"⚠️ Folder '{label_name}' not found in {subset}")
            continue

        print(f"   ➜ Loading {label_name} images...")
        for file_name in os.listdir(class_path):
            if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            file_path = os.path.join(class_path, file_name)
            img = cv2.imread(file_path)
            if img is None:
                continue

            if apply_clahe:
                img = enhance_contrast(img)

            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0  # normalize
            X.append(img)
            y.append(1 if label_name.lower() == "glaucoma" else 0)
            image_paths.append(file_path)

# --- Convert to NumPy arrays ---
X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\n✅ Preprocessing complete!")
print("Total images loaded:", len(X))
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Label distribution: glaucoma =", np.sum(y==1), ", normal =", np.sum(y==0))

# --- Randomly preview one image ---
idx = random.randint(0, len(X) - 1)
img = X[idx]
label = "Glaucoma" if y[idx] == 1 else "Normal"

plt.imshow(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title(f"Random Sample - {label}")
plt.axis('off')
plt.show()
