import streamlit as st
import pandas as pd
import joblib

# Load model dan preprocessing
rf_model = joblib.load("random_forest_model.pkl")
imputer = joblib.load("median_imputer.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Water Quality Class Prediction")
st.write(
    "Aplikasi demo Machine Learning menggunakan "
    "Random Forest Classifier untuk memprediksi kelas kualitas air."
)

# Input pengguna berdasarkan fitur yang dilatih
temperatur = st.number_input("Temperatur", min_value=15.0, max_value=45.0, value=28.0)
tds = st.number_input("TDS (Total Dissolved Solids)", min_value=0.0, max_value=2000.0, value=300.0)
tss = st.number_input("TSS (Total Suspended Solids)", min_value=0.0, max_value=500.0, value=50.0)
bod = st.number_input("BOD (Biological Oxygen Demand)", min_value=0.0, max_value=400.0, value=10.0)
cod = st.number_input("COD (Chemical Oxygen Demand)", min_value=0.0, max_value=1500.0, value=30.0)
curah_hujan = st.number_input("Curah Hujan", min_value=0.0, max_value=100.0, value=5.0)

if st.button("Predict"):
    # Membuat dataframe input sesuai nama kolom saat training
    input_data = pd.DataFrame([{
        "Temperatur": temperatur,
        "TDS": tds,
        "TSS": tss,
        "BOD": bod,
        "COD": cod,
        "CurahHujan": curah_hujan
    }])
    
    # Pastikan urutan feature sama seperti training
    input_data = input_data[feature_columns]
    
    # Gunakan imputer yang sudah di-fit pada training data
    input_imputed = pd.DataFrame(
        imputer.transform(input_data),
        columns=feature_columns
    )
    
    # Prediksi
    prediction = rf_model.predict(input_imputed)
    st.subheader("Hasil Prediksi")
    st.write(f"Predicted Class: {prediction[0]}")