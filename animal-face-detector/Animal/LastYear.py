import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Set random seed
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load MobileNetV2 as feature extractor
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
base_model.trainable = False

# Load dataset
dataset_dir = r"C:\Users\Avani\Downloads\animal-face-detector\animal-face-detector\animal_dataset\train"

classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
data, labels = [], []

for label in classes:
    folder_path = os.path.join(dataset_dir, label)
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        try:
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            data.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

data = np.array(data, dtype=np.float32)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=SEED)

# Build and compile model
model = Sequential([
    base_model,
    Dense(256, activation='relu'),
    Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save Model and Label Encoder
model.save("animal_classifier.h5")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Model and label encoder saved successfully.")

# ==========================
# Webcam Prediction with OpenCV
# ==========================
print("Starting webcam...")

# Load trained model and label encoder
model = load_model("animal_classifier.h5", compile=False)
label_encoder = joblib.load("label_encoder.pkl")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    class_index = np.argmax(preds)
    confidence = np.max(preds) * 100
    label = label_encoder.inverse_transform([class_index])[0]

    # Display prediction on frame
    text = f"{label} ({confidence:.2f}%)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Animal Classifier", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
