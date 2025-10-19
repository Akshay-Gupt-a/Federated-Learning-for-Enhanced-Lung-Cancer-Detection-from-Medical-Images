import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set paths
train_dir = 'central_training_split/train'
val_dir = 'central_training_split/val'

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Model architecture (deeper CNN)
def build_model():
    model = Sequential([
        Input(shape=(224, 224, 3)),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

# Hyperparameter tuning
learning_rates = [0.0001, 0.0005, 0.001, 0.005]

best_val_acc = 0.0
best_model = None

for lr in learning_rates:
    print(f"\n🔍 Training with learning rate: {lr}")
    
    model = build_model()
    optimizer = Adam(learning_rate=lr)

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    history = model.fit(
        train_data,
        epochs=35,
        validation_data=val_data,
        verbose=1
    )

    val_accuracy = history.history['val_accuracy'][-1]
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_model = model
        best_model.save('central_model_tuned.h5')
        print(f"✅ New best model saved with val_accuracy = {val_accuracy:.4f}")

print("🎯 Hyperparameter tuning complete.")
