## IMPORTATION DES LIBRARIES
import warnings
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap

# import the metrics class
from sklearn import metrics
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,accuracy_score,precision_score, recall_score,f1_score,classification_report
warnings.filterwarnings('ignore')

## CHARGEMENT DU JEU DE DONNÉES
df = pd.read_csv("data/student_dropout_dataset.csv")
df

df.info()
df.describe()
df.dtypes

## Prétraitement des données (Data Cleaning)
# Gestion des valeurs manques

df.isnull().sum()
# Presence de valuer negatif dans la colone age
nega = (df["age"] < 0).sum()
nega

# Doublon
db = df.duplicated().sum()
db

## Analyse exploratoire
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
variable = ["age","average_grade","absenteeism_rate","study_time_hours"]
for i, var in enumerate(variable):
    axes[i].hist(df[var], bins=30, edgecolor='black')
    axes[i].set_title(f"Histogramme de {var}")
    axes[i].set_xlabel(var)
    axes[i].set_ylabel("Fréquence")

plt.tight_layout()
plt.show()

# Matrice de correlation
corr_matrice = df.corr(numeric_only=True)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrice, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation")
plt.show() # Affiche la heatmap [10]

## Feature Engineering
# Ratio présence/absence
df['presence_ratio'] = 1 - df['absenteeism_rate']

# Score global (pondération fixe, arbitraire)
df['global_score'] = (df['average_grade'] * 0.5 + 
                      df['study_time_hours'] * 0.3 - 
                      df['absenteeism_rate'] * 0.2)

X = df.drop('dropout_risk', axis=1)
y = df['dropout_risk']
col_num = ["age","average_grade","absenteeism_rate","study_time_hours","presence_ratio","global_score"]
col_categorial = ["gender","internet_access","extra_activities"]
ct = ColumnTransformer(
    [("numerique", StandardScaler(),col_num),
     ("categorial", OneHotEncoder(drop="first"),col_categorial)])

## SPLIT DU DATASET
# Split (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

## MODELISATION
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}

results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ("prep", ct),
        ("clf", model)
    ])
    
    # Validation croisée
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    results[name] = {
        "CV F1 moyen": cv_scores.mean(),
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Test Precision": precision_score(y_test, y_pred),
        "Test Recall": recall_score(y_test, y_pred),
        "Test F1": f1_score(y_test, y_pred)
    }
    
    print(f"\n{'='*50}")
    print(f"Modèle : {name}")
    print(f"{'='*50}")
    print(f"Validation croisée F1 (moy ± std) : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"\nClassification Report sur test :")
    print(classification_report(y_test, y_pred))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))


## OPTIMISATION

param_grid = {
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [5, 10, None],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4]
}

rf_base = RandomForestClassifier(random_state=42)
pipeline_rf = Pipeline([("prep", ct), ("clf", rf_base)])

grid_search = GridSearchCV(
    pipeline_rf, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\nMeilleurs paramètres : {grid_search.best_params_}")
print(f"Meilleur score F1 (validation) : {grid_search.best_score_:.3f}")

# Évaluation du modèle optimisé
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f"F1 score sur test : {f1_score(y_test, y_pred_best):.3f}")
print(f"Accuracy sur test : {accuracy_score(y_test, y_pred_best):.3f}")

# SÉLECTION DES VARIABLES IMPORTANTES 
best_rf = best_model.named_steps["clf"]
feature_names = (col_num + 
                 list(best_model.named_steps["prep"].named_transformers_["categorial"].get_feature_names_out(col_categorial)))

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": best_rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 5 des variables les plus importantes :")
print(importance_df.head(5))

## SAUVEGARDE DU MODEL
# Sauvegarder le modèle
with open('model/abandon_predict.pkl', 'wb') as f:
    pickle.dump(best_model, f)