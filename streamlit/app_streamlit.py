import streamlit as st
import requests
import json
import pandas as pd

# Configuration de la page
st.set_page_config(page_title="Prédiction risque d'abandon", layout="centered")
st.title("🎓 Prédiction du risque d'abandon scolaire")
st.markdown("Remplissez les informations ci-dessous pour évaluer le risque d'abandon de l'étudiant.")

# Formulaire de saisie
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Âge", min_value=16, max_value=100, value=20, step=1)
        avg_grade = st.number_input("Moyenne générale (/20)", min_value=0.0, max_value=20.0, value=12.0, step=0.5)
        absenteeism = st.number_input("Taux d'absentéisme (0 à 0.5)", min_value=0.0, max_value=0.5, value=0.1, step=0.01, format="%.2f")
        study_hours = st.number_input("Temps d'étude journalier (heures)", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
    with col2:
        gender = st.selectbox("Sexe", options=["Male", "Female"])
        internet = st.selectbox("Accès Internet", options=["Yes", "No"])
        extra = st.selectbox("Activités extrascolaires", options=["Yes", "No"])
    
    submitted = st.form_submit_button("Prédire le risque")

# Lors de la soumission, appel à l'API Flask
if submitted:
    # Construction du payload
    payload = {
        "age": age,
        "average_grade": avg_grade,
        "absenteeism_rate": absenteeism,
        "study_time_hours": study_hours,
        "gender": gender,
        "internet_access": internet,
        "extra_activities": extra
    }
    
    # Adresse de l'API (à adapter si besoin)
    api_url = "http://localhost:5000/predict"
    
    try:
        response = requests.post(api_url, json=payload, timeout=5)
        if response.status_code == 200:
            result = response.json()
            risk = result["dropout_risk"]
            prob_no_risk = result["probabilities"]["no_risk"]
            prob_risk = result["probabilities"]["risk"]
            
            # Affichage des résultats
            st.subheader("📊 Résultat de la prédiction")
            if risk == 1:
                st.error(f"⚠️ Risque d'abandon élevé (probabilité : {prob_risk:.1%})")
            else:
                st.success(f"✅ Faible risque d'abandon (probabilité : {prob_no_risk:.1%})")
            
            # Barre de probabilité
            st.progress(prob_risk)
            st.write(f"**Probabilité de non‑abandon** : {prob_no_risk:.1%}")
            st.write(f"**Probabilité d’abandon** : {prob_risk:.1%}")
            
            # Rappel des règles métier (optionnel)
            with st.expander("🔍 Logique métier (aide à l'interprétation)"):
                conditions = []
                if avg_grade < 10:
                    conditions.append("Moyenne < 10")
                if absenteeism > 0.3:
                    conditions.append("Absentéisme > 30%")
                if study_hours < 1:
                    conditions.append("Temps d'étude < 1h")
                if len(conditions) >= 2:
                    st.warning(f"Selon les règles métier, l'étudiant remplit {len(conditions)} conditions : {', '.join(conditions)} → risque élevé.")
                else:
                    st.info(f"Selon les règles métier, l'étudiant remplit {len(conditions)} condition(s) : {', '.join(conditions) or 'aucune'} → risque faible.")
        else:
            st.error(f"Erreur API : {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Impossible de contacter l'API. Vérifiez que le serveur Flask est lancé (python app.py).")
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")