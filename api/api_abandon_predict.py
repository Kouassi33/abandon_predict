# app.py
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Chargement du modèle (à faire une seule fois au démarrage)
try:
    with open('model/abandon_predict.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None

app = Flask(__name__)

def preprocess_input(data):
    """
    À partir des données brutes JSON, calcule les features dérivées
    et retourne un DataFrame prêt pour la prédiction.
    """
    # Colonnes attendues dans la requête
    required_fields = [
        'age', 'average_grade', 'absenteeism_rate', 'study_time_hours',
        'gender', 'internet_access', 'extra_activities'
    ]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Champ manquant : {field}")

    # Récupération des valeurs
    age = float(data['age'])
    avg_grade = float(data['average_grade'])
    absenteeism = float(data['absenteeism_rate'])
    study_hours = float(data['study_time_hours'])
    gender = data['gender']
    internet = data['internet_access']
    extra = data['extra_activities']

    # Calcul des features dérivées (exactement comme dans l'entraînement)
    presence_ratio = 1 - absenteeism
    global_score = (avg_grade * 0.5 + study_hours * 0.3 - absenteeism * 0.2)

    # Construction du DataFrame avec les mêmes noms de colonnes qu'à l'entraînement
    input_df = pd.DataFrame([{
        'age': age,
        'average_grade': avg_grade,
        'absenteeism_rate': absenteeism,
        'study_time_hours': study_hours,
        'presence_ratio': presence_ratio,
        'global_score': global_score,
        'gender': gender,
        'internet_access': internet,
        'extra_activities': extra
    }])

    return input_df

@app.route('/health', methods=['GET'])
def health():
    """Vérification que l'API et le modèle sont opérationnels."""
    if model is None:
        return jsonify({'status': 'error', 'message': 'Modèle non chargé'}), 500
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction."""
    if model is None:
        return jsonify({'error': 'Modèle non disponible'}), 500

    # Récupération des données JSON
    if not request.is_json:
        return jsonify({'error': 'Le contenu doit être au format JSON'}), 400
    data = request.get_json()

    try:
        # Prétraitement
        input_df = preprocess_input(data)

        # Prédiction
        prediction = model.predict(input_df)[0]          # 0 ou 1
        proba = model.predict_proba(input_df)[0].tolist() # [prob_classe_0, prob_classe_1]

        # Construction de la réponse
        response = {
            'dropout_risk': int(prediction),
            'probabilities': {
                'no_risk': proba[0],
                'risk': proba[1]
            }
        }
        return jsonify(response)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Erreur interne : {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)