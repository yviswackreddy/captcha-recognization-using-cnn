import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import random
from sklearn.preprocessing import LabelEncoder
import string

# 1. Preprocessing Functions
def preprocess_image(image_path):
    # Read and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Noise reduction
    denoised = cv2.GaussianBlur(thresh, (3, 3), 0)
    
    return denoised

def segment_characters(image, num_chars=4):
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by x-coordinate
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    
    char_images = []
    for contour in contours[:num_chars]:
        x, y, w, h = cv2.boundingRect(contour)
        # Extract character
        char = image[y:y+h, x:x+w]
        # Resize to 28x28
        char = cv2.resize(char, (28, 28))
        char = char / 255.0  # Normalize
        char_images.append(char Acheter le code completchar)
    
    return np.array(char_images)

# 2. Model Creation
def create_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 3. Data Preparation
def prepare_data(data_dir):
    characters = list(string.ascii_uppercase + string.digits)
    label_encoder = LabelEncoder()
    label_encoder.fit(characters)
    
    X, y = [], []
    for char in characters:
        char_path = os.path.join(data_dir, char)
        for img_name in os.listdir(char_path):
            img_path = os.path.join(char_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            img = img / 255.0
            X.append(img)
            y.append(char)
    
    X = np.array(X).reshape(-1, 28, 28, 1)
    y = label_encoder.transform(y)
    
    return X, y, label_encoder

# 4. Main CAPTCHA Recognition Function
def recognize_captcha(image_path, model, label_encoder, num_chars=4):
    # Preprocess and segment
    processed_image = preprocess_image(image_path)
    char_images = segment_characters(processed_image, num_chars)
    
    # Reshape for model
    char_images = char_images.reshape(-1, 28, 28, 1)
    
    # Predict
    predictions = model.predict(char_images)
    predicted_chars = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    
    return ''.join(predicted_chars)

# 5. Training Function
def train_model(data_dir, epochs=10):
    X, y, label_encoder = prepare_data(data_dir)
    
    model = create_model(len(label_encoder.classes_))
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    
    return model, label_encoder

# 6. Main Execution
if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/captcha/character/dataset"
    model, label_encoder = train_model(data_dir)
    
    # Save model
    model.save("captcha_model.h5")
    
    # Test on a CAPTCHA image
    captcha_image = "path/to/test/captcha.png"
    result = recognize_captcha(captcha_image, model, label_encoder)
    print(f"Predicted CAPTCHA: {result}")