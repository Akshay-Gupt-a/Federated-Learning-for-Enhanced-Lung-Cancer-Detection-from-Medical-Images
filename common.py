# common.py
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_model():
    """Build the same architecture you used for central training."""
    model = Sequential([
        Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
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

def build_or_load_model(weights_path="central_model_tuned.h5", lr=5e-4):
    """Load pretrained model if exists, otherwise build & compile a fresh model."""
    if os.path.exists(weights_path):
        print(f"Loading pretrained model from: {weights_path}")
        model = load_model(weights_path)
        # Ensure compiled (recompile to be safe)
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
    else:
        print("Pretrained model not found. Building a fresh model.")
        model = build_model()
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
    return model

def load_client_data(client_dir, img_size=(224,224), batch_size=32, shuffle=True):
    labels_path = os.path.join(client_dir, "labels.csv")
    df = pd.read_csv(labels_path)

    # Ensure correct columns
    # Expecting: filename,label
    df["filepath"] = df["Image Index"].apply(lambda x: os.path.join(client_dir, "images", x))
    df["label"] = df["Finding Labels"].apply(lambda x: 0 if x == "No Finding" else 1)

    def process_row(filepath, label):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = img / 255.0
        return img, label

    filepaths = df["filepath"].values
    labels = df["label"].values

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(lambda x, y: process_row(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds, len(df)



def load_central_val(val_dir="central_training_split/val", batch_size=BATCH_SIZE, target_size=IMG_SIZE):
    """
    Load the central validation set (expects folder structure val/Cancer, val/Normal).
    """
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Central val directory not found at: {val_dir}")

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
    val_gen = datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    return val_gen
