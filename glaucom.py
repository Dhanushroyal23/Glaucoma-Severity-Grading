import os
import cv2

# ✅ Main dataset folder
dataset_path = r"C:\Users\deswa\Downloads\glaucoma\RIM-ONE_DL_images\partitioned_by_hospital"

print("📂 Dataset path:", dataset_path)

if not os.path.exists(dataset_path):
    print("❌ Error: Dataset path not found!")
    exit()

print("📁 Contents:", os.listdir(dataset_path))

# --- Look inside each subset (train/test) ---
subfolders = ["training_set", "test_set"]
found = False

for subset in subfolders:
    subset_path = os.path.join(dataset_path, subset)
    if not os.path.exists(subset_path):
        continue

    print(f"\n🔍 Checking inside '{subset}' folder...")
    classes = os.listdir(subset_path)
    print("   Classes found:", classes)

    if "Glaucoma" in classes or "glaucoma" in classes:
        # handle case-insensitivity
        glaucoma_folder = "Glaucoma" if "Glaucoma" in classes else "glaucoma"
        glaucoma_path = os.path.join(subset_path, glaucoma_folder)

        images = os.listdir(glaucoma_path)
        if images:
            first_image = images[0]
            img_path = os.path.join(glaucoma_path, first_image)
            print(f"✅ Found sample image in {subset}: {img_path}")

            # Try reading the image
            img = cv2.imread(img_path)
            if img is not None:
                print("🖼️ Image loaded successfully! Shape:", img.shape)
            else:
                print("⚠️ Could not read image file.")
            found = True
            break

if not found:
    print("❌ No 'Glaucoma' folder found inside training_set or test_set.")

#preprocess
import os
import cv2
import numpy as np

# --- Configuration ---
dataset_path = r"C:\Users\deswa\Downloads\glaucoma\RIM-ONE_DL_images\partitioned_by_hospital"
img_size = 224  # Resize to 224x224 for CNNs (like ResNet, VGG, etc.)
apply_clahe = True  # Apply CLAHE for better contrast

# --- Initialize lists ---
X = []
y = []

# --- Helper function for CLAHE ---
def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

# --- Load images from training_set and test_set ---
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
            file_path = os.path.join(class_path, file_name)
            if not (file_name.lower().endswith(('.jpg', '.jpeg', '.png'))):
                continue

            img = cv2.imread(file_path)
            if img is None:
                print(f"⚠️ Could not read {file_name}")
                continue

            if apply_clahe:
                img = enhance_contrast(img)

            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0  # normalize to [0, 1]
            X.append(img)
            y.append(1 if label_name.lower() == "glaucoma" else 0)

# --- Convert to NumPy arrays ---
X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\n✅ Preprocessing complete!")
print("Total images loaded:", len(X))
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Label distribution: glaucoma =", np.sum(y==1), ", normal =", np.sum(y==0))
cv2.imshow("Sample Image", img)
cv2.waitKey(0)      # waits until you press any key
cv2.destroyAllWindows()

