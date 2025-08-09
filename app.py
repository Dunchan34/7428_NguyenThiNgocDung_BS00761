import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

st.title("Product Quality Prediction - P7 Demo")

@st.cache_resource
def load_model():
    return joblib.load('quality_model.pkl')

# Load model
try:
    model = load_model()
except:
    model = None
    st.warning("Model file quality_model.pkl not found. Upload model to repo for predictions.")

uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # EDA plots
    st.subheader("EDA Charts")
    if 'Air temperature [K]' in df.columns:
        plt.figure()
        sns.histplot(df['Air temperature [K]'], kde=True)
        st.pyplot()

    if 'Process temperature [K]' in df.columns:
        plt.figure()
        sns.histplot(df['Process temperature [K]'], kde=True)
        st.pyplot()

    if 'Rotational speed [rpm]' in df.columns:
        plt.figure()
        sns.histplot(df['Rotational speed [rpm]'], kde=True)
        st.pyplot()

    if 'Torque [Nm]' in df.columns:
        plt.figure()
        sns.histplot(df['Torque [Nm]'], kde=True)
        st.pyplot()

    if 'Tool wear [min]' in df.columns:
        plt.figure()
        sns.histplot(df['Tool wear [min]'], kde=True)
        st.pyplot()

    # Prediction
    if model is not None:
        try:
            preds = model.predict(df)
            df['Predicted'] = preds
            st.subheader("Prediction Results")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Upload a CSV file to start analysis.")
