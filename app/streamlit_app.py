import streamlit as st
import requests
import base64
from PIL import Image
import io

st.title("🧠 Système d'Aide au Diagnostic - Tumeurs Cérébrales")
st.write("Upload une IRM cérébrale pour obtenir le stade et l'explication Grad-CAM.")

uploaded_file = st.file_uploader("Choisir une image IRM...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image Uploadée', width=250)
    
    if st.button("Lancer le Diagnostic"):
        with st.spinner("Analyse en cours par l'IA..."):
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post("http://localhost:8000/predict", files=files)
                response.raise_for_status()
                result = response.json()
                
                st.subheader("Résultat du Diagnostic")
                st.success(f"**{result['diagnostic']}**")
                st.info(f"Confiance (Softmax) : {result['confidence']*100:.2f}%")
                st.warning(f"Score d'Anomalie (Reconstruction) : {result['anomaly_mse_score']:.4f}")
                
                if result['requires_manual_review']:
                    st.error("⚠️ ALERTE : Cas suspect ou incertain. Révision manuelle prioritaire requise !")
                else:
                    st.success("✅ Prédiction fiable.")
                    
                st.subheader("Explicabilité (Zones d'activation)")
                gradcam_bytes = base64.b64decode(result['gradcam_image_base64'])
                gradcam_img = Image.open(io.BytesIO(gradcam_bytes))
                st.image(gradcam_img, caption='Carte de chaleur superposée', width=250)
            except Exception as e:
                st.error(f"Erreur de connexion à l'API : {e}")