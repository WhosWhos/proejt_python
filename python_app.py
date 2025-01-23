import streamlit as st
import joblib

# Sauvegarder le modèle
joblib.dump(rf_model, 'random_forest_model.pkl')

# Charger le modèle pré-entraîné
model = joblib.load('random_forest_model.pkl')

# Fonction pour faire des prédictions
def predict_intubation(input_data):
    # Normaliser les données
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    prediction = model.predict(input_data_scaled)
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
    'INMSUPR': [1 if inmsupr == "Oui" else 0]
})

# Bouton pour faire la prédiction
if st.button("Prédire le risque d'intubation"):
    prediction = predict_intubation(input_data)
    if prediction[0] == 1:
        st.success("Le patient est à risque d'intubation.")
    else:
        st.success("Le patient n'est pas à risque d'intubation.")
