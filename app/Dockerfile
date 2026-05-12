# Utiliser une image Python légère
FROM python:3.9-slim

# Installer OpenCV dépendances système
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Définir le dossier de travail
WORKDIR /app

# Copier requirements.txt et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code et les modèles
COPY . .
# Note: En vrai déploiement, les modèles .h5 seraient dans un cloud bucket, 
# mais pour le projet on les inclus dans l'image.

# Exposer le port FastAPI
EXPOSE 8000

# Lancer le serveur
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]