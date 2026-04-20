# Image de base Python
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers requirements (si existant)
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code du projet
COPY . .

# Exposer le port par défaut de Flask
EXPOSE 5000

# Variable d'environnement pour Flask
ENV FLASK_APP=api_abandon_predict.py
ENV FLASK_RUN_HOST=0.0.0.0

# Commande de démarrage
CMD ["flask", "run"]