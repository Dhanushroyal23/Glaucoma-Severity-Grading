import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

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
print("Train:", X_train.shape, "Validation:", X_val.shape)

# =====================================================
# 3️⃣ DEFINE MODEL (TRANSFER LEARNING)
# =====================================================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
for layer in base_model.layers:
    layer.trainable = False  # freeze pretrained layers

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

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

# =====================================================
# 5️⃣ EVALUATE
# =====================================================
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"\n🎯 Validation Accuracy: {val_acc*100:.2f}%")

# =====================================================
# 6️⃣ PLOT ACCURACY & LOSS
# =====================================================
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
