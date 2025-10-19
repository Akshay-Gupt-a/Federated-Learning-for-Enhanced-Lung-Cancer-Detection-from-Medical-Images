import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ------------------------------
# Step 1: Check for Unreadable Images
# ------------------------------
def check_bad_images(folder_path):
    bad_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.png'):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    img.verify()
                except Exception as e:
                    print(f"Bad image: {path} -> {e}")
                    bad_files.append(path)
    print(f"✅ Image check complete. Bad images found: {len(bad_files)}")
    return bad_files

# Run image check
bad_files = check_bad_images('D:/Project/central_training')
# Optionally remove them
for file in bad_files:
    os.remove(file)

# ------------------------------
# Step 2: Image Data Generators
# ------------------------------
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'D:/Project/central_training',
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    'D:/Project/central_training',
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# ------------------------------
# Step 3: Build CNN Model
# ------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ------------------------------
# Step 4: Train the Model
# ------------------------------
try:
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator
    )
except Exception as e:
    print("❌ Training failed due to:", e)

# ------------------------------
# Step 5: Save the Model
# ------------------------------
model.save('central_model.h5')
print("✅ Model training complete and saved as 'central_model.h5'")
