import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

st.title('Machine Failure Analysis')

# Bạn có thể thay phần upload bằng đọc file trực tiếp từ Github nếu muốn
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

    st.header("Figure 37: Distribution of Machine Temperature")
    plt.figure(figsize=(8, 6))
    sns.histplot(df['machine_temperature'], kde=True, color='blue')
    plt.xlabel('Air Temperature (K)')
    plt.ylabel('Frequency')
    st.pyplot()

    st.header("Figure 38: Defect Rates by Material Quality")
    defect_rates = df.groupby('material_quality')['defect_status'].mean().reset_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(x='material_quality', y='defect_status', data=defect_rates,
                palette=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xlabel('Material Quality (L: Low, M: Medium, H: High)')
    plt.ylabel('Defect Rate')
    st.pyplot()

    st.header("Figure 39: Correlation Heatmap")
    corr = df[['machine_temperature', 'humidity', 'lead_time', 'defect_status']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    st.pyplot()

    st.header("Figure 40: Boxplot of Defects vs. Machine Temperature")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='defect_status', y='machine_temperature', data=df)
    plt.xlabel('Defect Status (0: Pass, 1: Fail)')
    plt.ylabel('Machine Temperature (K)')
    st.pyplot()

    st.header("Figure 42: Errors vs. Humidity Box Plot")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='defect_status', y='humidity', data=df)
    plt.xlabel('Defect Status (0: Pass, 1: Fail)')
    plt.ylabel('Humidity (%)')
    st.pyplot()

    # Logistic Regression model
    st.header("Logistic Regression Model")

    features = ['machine_temperature', 'Torque [Nm]', 'Tool wear [min]']
    X = df[features]
    y = df['defect_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("Confusion Matrix")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot()

else:
    st.info("Please upload the ai4i2020.csv file to start the analysis.")
