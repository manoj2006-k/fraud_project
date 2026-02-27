import streamlit as st
import pandas as pd
import pickle

# ===== LOAD MODEL =====
model, feature_names = pickle.load(open("model.pkl", "rb"))

st.title("💳 AI Transaction Fraud Detection")

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    if st.button("Run Detection"):

        data = df.copy()

        # ENCODE TEXT
        for col in data.select_dtypes(include=["object"]).columns:
            data[col] = data[col].astype("category").cat.codes

        # KEEP ONLY TRAINED FEATURES
        data = data[feature_names]

        # ===== PREDICTIONS =====
        predictions = model.predict(data)
        risk_scores = model.predict_proba(data)[:, 1]

        df["Prediction"] = predictions
        df["Risk Score"] = risk_scores

        # ALERT SYSTEM
        df["Alert"] = df["Risk Score"].apply(
            lambda x: "⚠️ HIGH RISK" if x > 0.8 else "Safe"
        )

        st.subheader("Prediction Results")
        st.write(df)

        st.subheader("Risk Score Chart")
        st.bar_chart(df["Risk Score"])

        st.subheader("High Risk Transactions")
        st.write(df[df["Risk Score"] > 0.8])