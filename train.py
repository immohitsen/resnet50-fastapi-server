# train.py
import tensorflow as tf
from transformers import TFResNetModel
from config import IMAGE_SIZE, NUM_CLASSES, EPOCHS, MODEL_NAME, MODEL_WEIGHTS_PATH

def build_finetuned_model():
    """Pre-trained ResNet-50 model ko load karke uspar naya classifier head lagata hai."""
    # Base model load karein
    base_model = TFResNetModel.from_pretrained(MODEL_NAME, from_pt=True)
    
    # Base model ki layers ko freeze karein taaki unke weights update na ho
    base_model.trainable = False

    # Naya model banayein
    input_layer = tf.keras.Input(shape=(3, IMAGE_SIZE, IMAGE_SIZE), name="pixel_values")
    
    # Base model se features nikalein
    x = base_model(input_layer).pooler_output
    x = tf.keras.layers.Flatten()(x)
    
    # Hamara custom classification layer
    # Dense layer hamare 2 classes (Cancer/Normal) ke liye
    output_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Model ko compile karein
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model built and compiled successfully.")
    model.summary()
    return model

def train_model(model, train_ds, val_ds):
    """Model ko train aur save karta hai."""
    print("🚀 Starting model fine-tuning...")
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    
    # Trained weights ko save karein
    model.save_weights(MODEL_WEIGHTS_PATH)
    print(f"✅ Model weights saved to {MODEL_WEIGHTS_PATH}")
    return model