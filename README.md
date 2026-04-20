# Prédiction du risque d’abandon scolaire

Projet de Machine Learning visant à prédire si un étudiant risque d’abandonner ses études à partir de données académiques et comportementales.

## Objectif

Construire un modèle fiable et interprétable pour identifier précocement les élèves en difficulté, permettant aux établissements de mettre en place des actions de soutien ciblées.

## Jeu de données

- **Source** : fichier synthétique `student_dropout_dataset.csv`
- **Taille** : 300 lignes, 8 colonnes
- **Variables** :
  - `age` (15–24 ans)
  - `gender` (Male/Female)
  - `average_grade` (note sur 20)
  - `absenteeism_rate` (0 à 0.5)
  - `internet_access` (Yes/No)
  - `study_time_hours` (heures par jour)
  - `extra_activities` (Yes/No)
  - `dropout_risk` (cible : 0 = non, 1 = risque)

**Règle de construction de la cible** :  
`dropout_risk = 1` si au moins deux conditions sont vraies :
- moyenne générale < 10
- absentéisme > 30%
- temps d’étude < 1 heure

## Technologies utilisées

| Domaine | Bibliothèques |
|---------|----------------|
| Langage | Python 3 |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| ML & pipeline | Scikit-learn (StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline) |
| Modèles | LogisticRegression, RandomForestClassifier, SVC |
| Optimisation | GridSearchCV, cross_val_score |
| Évaluation | accuracy, precision, recall, f1-score, matrice de confusion |
| Interface de démonstration | Streamlit |
| Sauvegarde | Pickle |

## Structure du projet

projet_abandon_scolaire/
│
├── data/
│ └── student_dropout_dataset.csv
│
├── notebooks/
│ └── modelisation.ipynb # Notebook Jupyter complet
│
├── app/
│ └── streamlit_app.py # Interface de démonstration
│
├── models/
│ └── best_rf_model.pkl # Modèle Random Forest optimisé
│
├── requirements.txt
└── README.md


## Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/Kouassi33/abandon_predict.git
   cd projet_abandon_scolaire

## Créer un environnement virtuel (optionnel mais recommandé)

python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

## Installer les dépendances

pip install -r requirements.txt

**Contenu de requirements.txt :**

pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
pickle-mixin
