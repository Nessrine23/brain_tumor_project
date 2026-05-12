from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
import base64
from gradcam import make_gradcam_heatmap, overlay_gradcam

app = FastAPI(title="Brain Tumor AI Diagnostic API")

# Chargement des modèles
model = tf.keras.models.load_model('../models/efficientnet_best.h5', compile=False)
autoencoder = tf.keras.models.load_model('../models/autoencoder.h5', compile=False)

CLASS_NAMES = ["Stade 0: Pas de tumeur", "Stade I: Gliome", "Stade II: Méningiome", "Stade III: Pituitaire"]
CONFIDENCE_THRESHOLD = 0.7

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array) # ✅ BON PRÉTRAITEMENT
    return np.expand_dims(img_array, axis=0), img_array / 255.0 # On garde img_array/255.0 pour l'affichage GradCAM

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array, raw_img = preprocess_image(image_bytes)
    
    # 1. Classification
    preds = model.predict(img_array)
    confidence = float(np.max(preds))
    class_idx = int(np.argmax(preds))
    
    # 2. Détection d'anomalie (Erreur de reconstruction)
    reconstructed = autoencoder.predict(img_array)
    mse = float(np.mean(np.square(img_array - reconstructed)))
    is_anomaly = mse > 0.01 # Seuil à ajuster selon vos tests
    
    # 3. Incertitude probabiliste
    requires_manual_review = (confidence < CONFIDENCE_THRESHOLD) or is_anomaly
    
    # 4. Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model)
    gradcam_img = overlay_gradcam(raw_img, heatmap)
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))
    gradcam_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "diagnostic": CLASS_NAMES[class_idx],
        "confidence": confidence,
        "anomaly_mse_score": mse,
        "requires_manual_review": requires_manual_review,
        "gradcam_image_base64": gradcam_base64
    }