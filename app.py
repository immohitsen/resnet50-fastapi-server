import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # fix for Transformers + tf-keras
import tensorflow as tf
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor
from PIL import Image
import io

# uvicorn app:app --reload --port 8000

from train import build_finetuned_model
from config import MODEL_WEIGHTS_PATH, MODEL_NAME, IMAGE_SIZE

# ---- FastAPI App ----
app = FastAPI(title="Cancer Detection API")

# Allow requests from frontend (Next.js on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Global variables for model ----
model = None
processor = None
CLASS_LABELS = ["Cancer", "Normal"]

@app.on_event("startup")
async def load_model():
    global model, processor
    try:
        print("🚀 Loading model...")
        model = build_finetuned_model()
        model.load_weights(MODEL_WEIGHTS_PATH)
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        model = None
        processor = None

def preprocess(file: UploadFile):
    """Convert uploaded file to model input"""
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    # HuggingFace AutoImageProcessor handles resizing & normalization
    inputs = processor(images=image, return_tensors="np")["pixel_values"]
    return inputs

def predict(file: UploadFile):
    """Run inference"""
    inputs = preprocess(file)
    preds = model.predict(inputs, verbose=0)
    scores = tf.nn.softmax(preds[0])
    class_index = int(np.argmax(scores))
    return {
        "prediction": CLASS_LABELS[class_index],
        "confidence": float(np.max(scores)) * 100,
        "probabilities": {
            CLASS_LABELS[0]: float(scores[0]) * 100,
            CLASS_LABELS[1]: float(scores[1]) * 100
        }
    }

# ---- API Routes ----
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Cancer Detection API is running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None or processor is None:
        return {"error": "Model is still loading. Please try again in a few moments."}
    result = predict(file)
    return result
