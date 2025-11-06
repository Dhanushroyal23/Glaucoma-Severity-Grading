import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tkinter import Tk, filedialog  # 👈 for file picker

# =====================================================
# 1️⃣ LOAD & PREPROCESS DATA
# =====================================================
dataset_path = r"C:\Users\deswa\Downloads\glaucoma\RIM-ONE_DL_images\partitioned_by_hospital"
img_size = 224
apply_clahe = True

X, y = [], []

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

print("📂 Loading images...")
for subset in ["training_set", "test_set"]:
    for label_name in ["glaucoma", "normal"]:
        class_path = os.path.join(dataset_path, subset, label_name)
        if not os.path.exists(class_path):
            continue
        for file in os.listdir(class_path):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(class_path, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                if apply_clahe:
                    img = enhance_contrast(img)
                img = cv2.resize(img, (img_size, img_size))
                X.append(img / 255.0)
                y.append(1 if label_name == "glaucoma" else 0)

X = np.array(X, dtype=np.float32)
y = np.array(y)
print("✅ Data loaded:", X.shape, "Labels:", y.shape)
print("   Glaucoma =", np.sum(y==1), ", Normal =", np.sum(y==0))

# =====================================================
# 2️⃣ SPLIT DATA
# =====================================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_train = to_categorical(y_train, 2)
y_val = to_categorical(y_val, 2)

# =====================================================
# 3️⃣ DEFINE / LOAD MODEL
# =====================================================
model_path = r"C:\Users\deswa\OneDrive\Desktop\capstone\glaucoma_model.keras"

if os.path.exists(model_path):
    print("📦 Loading existing model...")
    model = load_model(model_path)
else:
    print("🧠 Creating new model...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # =====================================================
    # 4️⃣ TRAIN MODEL
    # =====================================================
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    model.save(model_path)
    print(f"💾 Model saved as {model_path}")

    # Plot accuracy/loss
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# =====================================================
# 5️⃣ EVALUATE MODEL
# =====================================================
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"\n🎯 Validation Accuracy: {val_acc*100:.2f}%")

# =====================================================
# 6️⃣ LIVE PREDICTION FUNCTION
# =====================================================
def predict_image(image_path):
    print(f"\n🔍 Predicting: {image_path}")
    if not os.path.exists(image_path):
        print("❌ File not found!")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not read image.")
        return
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    class_names = ["Normal", "Glaucoma"]
    result = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    print(f"🧾 Prediction: {result} ({confidence:.2f}% confidence)")

    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title(f"{result} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

# =====================================================
# 7️⃣ FILE PICKER (AUTOMATIC)
# =====================================================
print("\n📁 Please select an image to predict...")
Tk().withdraw()  # Hide the root Tk window
file_path = filedialog.askopenfilename(
    title="Select a Fundus Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if file_path:
    predict_image(file_path)
else:
    print("⚠️ No file selected.")
