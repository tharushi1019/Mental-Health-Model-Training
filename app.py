import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# -------------------- Page Config -------------------- #
st.set_page_config(
    page_title="🧠 Mental Health Predictor",
    layout="wide",
    page_icon="💡",
    initial_sidebar_state="expanded"
)

# -------------------- Load Model & Data -------------------- #
@st.cache_data
def load_dataset():
    return pd.read_csv("data/cleaned_survey.csv")

@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

# -------------------- Sidebar Navigation -------------------- #
st.sidebar.title("🔎 App Navigation")
section = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data Overview", "📈 Visualizations", "🔮 Predict", "📉 Model Performance", "ℹ️ About App"])

# -------------------- Load Data -------------------- #
df = load_dataset()
model = load_model()

# -------------------- HOME -------------------- #
if section == "🏠 Home":
    st.markdown("""
        <h1 style='text-align:center; color:#4B8BBE;'>Mental Health in Tech</h1>
        <p style='text-align:center; font-size:18px;'>
        This app predicts whether an individual is likely to seek treatment for mental health issues, based on workplace survey data.
        </p>
    """, unsafe_allow_html=True)
    st.image("https://www.amsa.org/wp-content/uploads/2021/02/Artboard-1-1.png", use_container_width=True)

# -------------------- DATA OVERVIEW -------------------- #
elif section == "📊 Data Overview":
    st.header("🧾 Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.write("\n**Dataset Shape:**", df.shape)
    st.markdown("**Column Types:**")
    st.dataframe(df.dtypes.reset_index().rename(columns={0: "Type", "index": "Column"}))

# -------------------- VISUALIZATIONS -------------------- #
elif section == "📈 Visualizations":
    st.header("📈 Exploratory Visualizations")

    st.markdown("### Gender Distribution")
    fig1 = px.histogram(df, x="Gender", color="Gender")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### Age vs Treatment")
    fig2 = px.box(df, x="treatment", y="Age", color="treatment", title="Treatment by Age")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Country-wise Treatment")
    fig3 = px.histogram(df, x="Country", color="treatment", barmode="group")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------- PREDICT -------------------- #
elif section == "🔮 Predict":
    st.header("🔮 Mental Health Treatment Prediction")
    st.markdown("Enter your details below:")

    # Raw Inputs
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ["male", "female", "other"])
    self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
    family_history = st.radio("Family history of mental illness?", ["Yes", "No"])
    work_interfere = st.selectbox("Mental health interference with work?", ["Never", "Rarely", "Sometimes", "Often"])
    no_employees = st.selectbox("Company size?", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
    tech_company = st.selectbox("Is it a tech company?", ["Yes", "No"])
    benefits = st.radio("Mental health benefits available?", ["Yes", "No"])
    care_options = st.radio("Are care options available?", ["Yes", "No", "Not sure"])
    wellness_program = st.radio("Wellness program available?", ["Yes", "No", "Don't know"])
    seek_help = st.radio("Do you know how to seek help?", ["Yes", "No", "Don't know"])
    anonymity = st.radio("Is anonymity protected?", ["Yes", "No", "Don't know"])
    leave = st.selectbox("Ease of leave for mental health?", ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"])
    mental_health_consequence = st.selectbox("Mental health consequences?", ["Yes", "No", "Maybe"])
    phys_health_consequence = st.selectbox("Physical health consequences?", ["Yes", "No", "Maybe"])
    coworkers = st.selectbox("Comfort discussing with coworkers?", ["Yes", "No", "Some of them"])
    supervisor = st.selectbox("Comfort discussing with supervisor?", ["Yes", "No", "Some of them"])
    mental_health_interview = st.selectbox("Discuss mental health in interview?", ["Yes", "No", "Maybe"])
    phys_health_interview = st.selectbox("Discuss physical health in interview?", ["Yes", "No", "Maybe"])
    mental_vs_physical = st.selectbox("Mental vs Physical importance?", ["Yes", "No", "Don't know"])
    obs_consequence = st.selectbox("Observed mental health consequences?", ["Yes", "No"])

    # Manual Encoders
    yes_no = {"Yes": 1, "No": 0}
    gender_map = {"male": 0, "female": 1, "other": 2}
    interfere_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}
    employee_map = {
        "1-5": 0, "6-25": 1, "26-100": 2, "100-500": 3,
        "500-1000": 4, "More than 1000": 5
    }
    unsure_map = {"Yes": 1, "No": 0, "Not sure": 2, "Don't know": 2}
    leave_map = {
        "Very easy": 0, "Somewhat easy": 1, "Don't know": 2,
        "Somewhat difficult": 3, "Very difficult": 4
    }
    consequence_map = {"Yes": 1, "No": 0, "Maybe": 2}
    coworker_map = {"Yes": 1, "No": 0, "Some of them": 2}
    interview_map = {"Yes": 1, "No": 0, "Maybe": 2}

    # Convert inputs
    input_data = np.array([[
        age,
        gender_map[gender],
        yes_no[self_employed],
        yes_no[family_history],
        interfere_map[work_interfere],
        employee_map[no_employees],
        yes_no[remote_work],
        yes_no[tech_company],
        yes_no[benefits],
        unsure_map[care_options],
        unsure_map[wellness_program],
        unsure_map[seek_help],
        unsure_map[anonymity],
        leave_map[leave],
        consequence_map[mental_health_consequence],
        consequence_map[phys_health_consequence],
        coworker_map[coworkers],
        coworker_map[supervisor],
        interview_map[mental_health_interview],
        interview_map[phys_health_interview],
        unsure_map[mental_vs_physical],
        yes_no[obs_consequence]
    ]])

    # Prediction
    if st.button("Predict Treatment Likelihood"):
        try:
            pred = model.predict(input_data)[0]
            result = "🟢 Likely to Seek Treatment" if pred == 1 else "🔴 Unlikely to Seek Treatment"
            st.success(f"Prediction Result: {result}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------- MODEL PERFORMANCE -------------------- #
elif section == "📉 Model Performance":
    st.header("📉 Model Evaluation")

    try:
        y_test = pickle.load(open("data/y_test.pkl", "rb"))
        X_test = pickle.load(open("data/X_test.pkl", "rb"))
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.metric("Accuracy", f"{acc:.2f}")
        st.metric("Precision", f"{prec:.2f}")
        st.metric("Recall", f"{rec:.2f}")

        st.markdown("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig)

    except FileNotFoundError:
        st.warning("Performance data not available. Please add y_test.pkl and X_test.pkl to the data folder.")

# -------------------- ABOUT -------------------- #
elif section == "ℹ️ About App":
    st.header("ℹ️ About This Project")
    st.markdown("""
    This Streamlit app is part of Tharushi's final-year Intelligent Systems assignment. It uses a trained machine learning model to predict mental health treatment likelihood among tech professionals.

    **Technologies Used:**
    - Streamlit
    - Scikit-learn
    - Pandas, Numpy
    - Plotly, Seaborn, Matplotlib
    - Pickle (for model deployment)

    **Dataset:** Mental Health in Tech Survey (Kaggle)
    """)
