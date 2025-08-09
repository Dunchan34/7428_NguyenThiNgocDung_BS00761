import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz


st.title('Machine Failure Analysis')

uploaded_file = st.file_uploader("Upload your CSV data file (ai4i2020.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Đổi tên cột để dùng cho sau
    df = df.rename(columns={'Air temperature [K]': 'machine_temperature'})

    # Tạo cột defect_status
    df['defect_status'] = df['Machine failure']

    # Phân loại chất lượng vật liệu dựa trên Torque
    def classify_quality(torque):
        if torque < 30:
            return 'L'  # Low
        elif torque < 50:
            return 'M'  # Medium
        else:
            return 'H'  # High

    df['material_quality'] = df['Torque [Nm]'].apply(classify_quality)

    # Tạo dữ liệu giả lập cho độ ẩm và thời gian chờ (lead_time)
    df['humidity'] = np.random.uniform(40, 80, size=len(df))
    df['lead_time'] = np.random.randint(10, 60, size=len(df))

    # --- Figure 37 ---
    st.header("Figure 37: Distribution of Machine Temperature")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['machine_temperature'], kde=True, color='blue', ax=ax)
    ax.set_xlabel('Air Temperature (K)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # --- Figure 38 ---
    st.header("Figure 38: Defect Rates by Material Quality")
    defect_rates = df.groupby('material_quality')['defect_status'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='material_quality', y='defect_status', data=defect_rates,
                palette=['#1f77b4', '#ff7f0e', '#2ca02c'], ax=ax)
    ax.set_xlabel('Material Quality (L: Low, M: Medium, H: High)')
    ax.set_ylabel('Defect Rate')
    st.pyplot(fig)

    # --- Figure 39 ---
    st.header("Figure 39: Correlation Heatmap")
    corr = df[['machine_temperature', 'humidity', 'lead_time', 'defect_status']].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)

    # --- Figure 40 ---
    st.header("Figure 40: Boxplot of Defects vs. Machine Temperature")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='defect_status', y='machine_temperature', data=df, ax=ax)
    ax.set_xlabel('Defect Status (0: Pass, 1: Fail)')
    ax.set_ylabel('Machine Temperature (K)')
    st.pyplot(fig)

    # --- Figure 42 ---
    st.header("Figure 42: Errors vs. Humidity Box Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='defect_status', y='humidity', data=df, ax=ax)
    ax.set_xlabel('Defect Status (0: Pass, 1: Fail)')
    ax.set_ylabel('Humidity (%)')
    st.pyplot(fig)

    # --- Logistic Regression ---
    st.header("Figure 43: Logistic Regression Model")
    features = ['machine_temperature', 'Torque [Nm]', 'Tool wear [min]']
    X = df[features]
    y = df['defect_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # --- Decision Tree ---
    st.header("Figure 44: Decision Tree Structure")
    # Train lại Decision Tree với cùng dữ liệu
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    dot_data = export_graphviz(dt_model, out_file=None,
                               feature_names=features,
                               class_names=['Pass', 'Fail'],
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    st.graphviz_chart(graph)

else:
    st.info("Please upload the ai4i2020.csv file to start the analysis.")
