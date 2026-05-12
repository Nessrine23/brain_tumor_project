🧠 Système d'Aide au Diagnostic - Tumeurs Cérébrales
Projet de Fin d'Année (PFA) développé par Nessrine FELAH dans le cadre de la méthodologie CRISP-DM. Ce système utilise le Deep Learning (Transfer Learning avec EfficientNetB0) pour détecter et classifier les stades de tumeurs cérébrales à partir d'images IRM, avec une précision et une sensibilité dépassant les 95%.

🛡️ Fonctionnalités Principales
Classification multi-classes : Stade 0 (Sain), Stade I (Gliome), Stade II (Méningiome), Stade III (Pituitaire).
Explicabilité (XAI) : Génération de cartes de chaleur Grad-CAM pour transparenter la décision de l'IA.
Détection d'anomalies : Auto-encodeur pour identifier les cas atypiques ou inconnus.
Alertes cliniques : Système de double vérification (Confiance Softmax + Erreur de reconstruction) demandant une révision manuelle en cas de doute.
API & Dashboard : Backend asynchrone FastAPI et interface utilisateur Streamlit.