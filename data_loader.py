# data_loader.py
import tensorflow as tf
import numpy as np
from transformers import AutoImageProcessor
from config import IMAGE_SIZE, BATCH_SIZE, MODEL_NAME, NUM_CLASSES

# Hugging Face se processor load karein
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

def _preprocess(image_np, label_np):
    """Image ko model ke format mein convert karta hai."""
    # Processor numpy array expect karta hai
    # Ab image_np pehle se hi uint8 format mein hai
    processed = processor(images=image_np.numpy(), return_tensors="np")["pixel_values"][0]
    return processed, label_np

def preprocess_image(image, label):
    """Preprocessing ko TensorFlow graph mein wrap karta hai."""
    image, label = tf.py_function(
        _preprocess,
        [image, label],
        [tf.float32, tf.float32]
    )
    # Shape set karna zaroori hai
    image.set_shape([3, IMAGE_SIZE, IMAGE_SIZE])
    label.set_shape([NUM_CLASSES])
    return image, label

def load_datasets(train_dir="dataset/train", val_dir="dataset/val"):
    """Train aur Validation datasets ko load karta hai."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    # ✅ FIX: Image data type ko float32 se uint8 mein convert karein
    def cast_to_uint8(image, label):
        return tf.cast(image, tf.uint8), label

    train_ds = train_ds.map(cast_to_uint8, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(cast_to_uint8, num_parallel_calls=tf.data.AUTOTUNE)

    # Preprocessing apply karein
    train_ds = train_ds.unbatch().map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.unbatch().map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("✅ Datasets successfully loaded and preprocessed.")
    return train_ds, val_ds