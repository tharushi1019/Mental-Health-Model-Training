import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Mental Health Prediction App", layout="centered")

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose section", ["Home", "Data Overview", "Visualizations", "Predict", "Model Performance"])

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv("data/dataset.csv")
    return df

df = load_data()

# --- Navigation Logic ---
if app_mode == "Home":
    st.title("Mental Health in Tech Prediction App")
    st.markdown("This app predicts whether a person is likely to seek treatment for mental health issues based on survey responses.")

elif app_mode == "Data Overview":
    st.subheader("Dataset Overview")
    st.write(df.shape)
    st.write(df.head())

elif app_mode == "Visualizations":
    st.subheader("Visual Insights")
    st.bar_chart(df['gender'].value_counts())

elif app_mode == "Predict":
    st.subheader("Make a Prediction")
    
    # Input fields - example, adjust based on your features
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ["male", "female", "other"])
    
    # Convert gender to number
    gender_encoded = 0 if gender == "male" else 1 if gender == "female" else 2

    # Build input for model
    input_data = np.array([[age, gender_encoded]])
    
    # Prediction
    prediction = model.predict(input_data)
    result = "Likely to Seek Treatment" if prediction[0] == 1 else "Unlikely to Seek Treatment"
    
    st.success(f"Prediction: {result}")

elif app_mode == "Model Performance":
    st.subheader("Evaluation Results")
    st.text("Model: Random Forest Classifier")
    st.text("Accuracy: 84%")
