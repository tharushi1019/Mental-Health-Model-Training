import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# ------------------------ Config ------------------------
st.set_page_config(
    page_title="Mental Health Predictor",
    page_icon="🧠",
    layout="centered"
)

# ------------------------ Load Model ------------------------
try:
    model = pickle.load(open("model.pkl", "rb"))
except:
    st.error("❌ Model file not found. Please train and export model.pkl first.")

# ------------------------ Load Dataset ------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/dataset.csv")

df = load_data()

# ------------------------ Sidebar Navigation ------------------------
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data Overview", "📈 Visualizations", "🔮 Predict", "📉 Model Performance"])

# ------------------------ Page: Home ------------------------
if app_mode == "🏠 Home":
    st.markdown("<h1 style='text-align:center; color:purple;'>Mental Health in Tech</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align:center;'>This app predicts whether an individual in the tech industry is likely to seek mental health treatment based on survey data.</p>
        <br>
        <p style='text-align:center;'>Developed by <b>Tharushi</b> | Final Year Project</p>
    """, unsafe_allow_html=True)

# ------------------------ Page: Data Overview ------------------------
elif app_mode == "📊 Data Overview":
    st.header("📊 Dataset Overview")
    st.dataframe(df.head(20))
    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.markdown("### Summary")
    st.dataframe(df.describe())

# ------------------------ Page: Visualizations ------------------------
elif app_mode == "📈 Visualizations":
    st.header("📈 Visual Insights")

    st.markdown("### 🎯 Gender Distribution")
    fig1 = px.histogram(df, x="gender", color="gender", title="Gender Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 🎯 Age vs Treatment")
    fig2, ax = plt.subplots()
    sns.boxplot(x="treatment", y="age", data=df, ax=ax)
    st.pyplot(fig2)

    st.markdown("### 🎯 Country vs Treatment Count")
    fig3 = px.histogram(df, x="country", color="treatment", barmode="group", title="Treatment by Country")
    st.plotly_chart(fig3, use_container_width=True)

# ------------------------ Page: Predict ------------------------
elif app_mode == "🔮 Predict":
    st.header("🔮 Mental Health Treatment Prediction")

    st.markdown("### 👤 Input Your Info")

    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ["male", "female", "other"])
    family_history = st.radio("Family History of Mental Illness", ["Yes", "No"])
    benefits = st.selectbox("Employer Provides Mental Health Benefits", ["Yes", "No"])
    care_options = st.radio("Care Options Available", ["Yes", "No", "Not sure"])
    anonymity = st.radio("Anonymity at Workplace", ["Yes", "No", "Don't know"])

    # Encode Inputs
    gender_map = {"male": 0, "female": 1, "other": 2}
    binary_map = {"Yes": 1, "No": 0}
    unsure_map = {"Yes": 1, "No": 0, "Not sure": 2, "Don't know": 2}

    input_data = np.array([[
        age,
        gender_map[gender],
        binary_map[family_history],
        binary_map[benefits],
        unsure_map[care_options],
        unsure_map[anonymity]
    ]])

    if st.button("Predict"):
        prediction = model.predict(input_data)
        result = "🟢 Likely to Seek Treatment" if prediction[0] == 1 else "🔴 Unlikely to Seek Treatment"
        st.success(f"**Prediction:** {result}")

# ------------------------ Page: Model Performance ------------------------
elif app_mode == "📉 Model Performance":
    st.header("📉 Model Evaluation Metrics")

    try:
        y_test = pickle.load(open("data/y_test.pkl", "rb"))
        X_test = pickle.load(open("data/X_test.pkl", "rb"))
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Accuracy", f"{acc:.2f}")
        col2.metric("🎯 Precision", f"{prec:.2f}")
        col3.metric("📞 Recall", f"{rec:.2f}")

        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        st.pyplot(fig_cm)

    except Exception as e:
        st.warning("❌ Model performance data not available. Please ensure X_test.pkl and y_test.pkl are saved.")

