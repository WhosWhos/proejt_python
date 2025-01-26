import streamlit as st
import pandas as pd
import joblib
import requests
from io import StringIO

# Chargement des données avec cache
@st.cache_data
def data():
    url = 'https://raw.githubusercontent.com/WhosWhos/projet_python/refs/heads/main/covid19_data_test.csv'
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        st.error("Failed to load data from GitHub.")
        return None
        
# Chargement du modèle sauvegardé
loaded_model = joblib.load(r"C:\Users\Lenovo\Downloads\logistic_model.pkl")

# Définition de la fonction de prédiction
def predict_intubation(input_data):
    prediction = loaded_model.predict(input_data)
    return prediction

# Interface utilisateur
st.title("Détection de cas COVID-19 - Risque d'Intubation")
st.write("Veuillez entrer les informations suivantes :")

# Saisie des données par l'utilisateur
age = st.number_input("Âge", min_value=0, max_value=120, value=30)
sex = st.selectbox("Sexe", options=["Homme", "Femme"])
diabetes = st.selectbox("Diabète", options=["Oui", "Non"])
obesity = st.selectbox("Obésité", options=["Oui", "Non"])
cardiovascular = st.selectbox("Maladies cardiovasculaires", options=["Oui", "Non"])
hypertension = st.selectbox("Hypertension", options=["Oui", "Non"])
copd = st.selectbox("Bronchopneumopathie chronique obstructive", options=["Oui", "Non"])
asthma = st.selectbox("Asthme", options=["Oui", "Non"])
inmsupr = st.selectbox("Immunodépression", options=["Oui", "Non"])
pneumonia = st.selectbox("Pneumonie", options=["Oui", "Non"])
pregnant = st.selectbox("Grossesse", options=["Oui", "Non"])
tobacco = st.selectbox("Tabagisme", options=["Oui", "Non"])
icu = st.selectbox("Admission en unité de soins intensifs", options=["Oui", "Non"])
intubed = st.selectbox("Intubé", options=["Oui", "Non"])
usmr = st.selectbox("USMR (Unité de soins médicaux)", options=["1", "2"])  # Exemple de valeurs possibles

# Préparation des données pour la prédiction
input_data = pd.DataFrame({
    'AGE': [age],
    'SEX': [1 if sex == "Homme" else 0],
    'DIABETES': [1 if diabetes == "Oui" else 0],
    'OBESITY': [1 if obesity == "Oui" else 0],
    'CARDIOVASCULAR': [1 if cardiovascular == "Oui" else 0],
    'HYPERTENSION': [1 if hypertension == "Oui" else 0],
    'COPD': [1 if copd == "Oui" else 0],
    'ASTHMA': [1 if asthma == "Oui" else 0],
    'INMSUPR': [1 if inmsupr == "Oui" else 0],
    'PNEUMONIA': [1 if pneumonia == "Oui" else 0],
    'PREGNANT': [1 if pregnant == "Oui" else 0],
    'TOBACCO': [1 if tobacco == "Oui" else 0],
    'ICU': [1 if icu == "Oui" else 0],
    'INTUBED': [1 if intubed == "Oui" else 0],
    'USMR': [int(usmr)]  # Convertir en entier si nécessaire
})

# Bouton pour faire la prédiction
if st.button("Prédire le risque d'intubation"):
    prediction = loaded_model.predict(input_data)
    if prediction[0] == 1:
        st.success("Le patient est à risque d'intubation.")
    else:
        st.success("Le patient n'est pas à risque d'intubation.")
