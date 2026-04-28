import streamlit as st
import numpy as np
import requests
import pandas as pd
from datetime import datetime

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# -------------------------------
# SESSION STATE
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# HEADER
# -------------------------------
st.title("🏦 Credit Card Fraud Detection System")
st.caption("Real-time transaction monitoring")

# -------------------------------
# METRICS
# -------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Transactions", len(st.session_state.history))

with col2:
    frauds = sum(1 for h in st.session_state.history if h["risk"] == "HIGH")
    st.metric("High Risk Alerts", frauds)

with col3:
    avg_prob = np.mean([h["prob"] for h in st.session_state.history]) if st.session_state.history else 0
    st.metric("Avg Risk Score", f"{avg_prob:.2f}")

# -------------------------------
# LAYOUT
# -------------------------------
left, right = st.columns([1, 2])

# -------------------------------
# INPUT PANEL
# -------------------------------
with left:
    st.subheader("💳 Transaction Input")

    amount = st.number_input("Amount (₹)", min_value=0.0, value=100.0)
    location = st.selectbox("Location", ["Domestic", "International"])
    device = st.selectbox("Device", ["Mobile", "Laptop", "ATM"])
    hour = st.slider("Transaction Hour", 0, 23, 12)

    if st.button("🔍 Analyze Transaction", use_container_width=True):

        # -------------------------------
        # FEATURE GENERATION (NO OVERRIDE)
        # -------------------------------
        if location == "International" or amount > 3000:
            features = list(np.random.normal(2, 1.5, 28))
        elif amount > 1000:
            features = list(np.random.normal(1, 1, 28))
        else:
            features = list(np.random.normal(0, 1, 28))

        features.append(amount)

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"data": features}
            )

            result = response.json()
            prob = result["fraud_probability"]

            # -------------------------------
            # RISK CLASSIFICATION (ONLY UI)
            # -------------------------------
            if prob > 0.7:
                risk = "HIGH"
            elif prob > 0.3:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            # Save history
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "amount": amount,
                "location": location,
                "device": device,
                "prob": prob,
                "risk": risk
            })

            st.success("Transaction analyzed")

        except Exception as e:
            st.error(f"API Error: {e}")

# -------------------------------
# RIGHT PANEL
# -------------------------------
with right:

    st.subheader("📊 Risk Analysis")

    if st.session_state.history:

        latest = st.session_state.history[-1]

        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("Amount", f"₹{latest['amount']}")

        with colB:
            st.metric("Fraud Probability", f"{latest['prob']:.4f}")

        with colC:
            if latest["risk"] == "HIGH":
                st.error("🚨 HIGH RISK")
            elif latest["risk"] == "MEDIUM":
                st.warning("⚠️ MEDIUM RISK")
            else:
                st.success("✅ SAFE")

        st.markdown("---")

        # -------------------------------
        # CHARTS
        # -------------------------------
        df = pd.DataFrame(st.session_state.history)

        st.markdown("### 📈 Risk Trend")
        st.line_chart(df["prob"])

        st.markdown("### 📊 Risk Distribution")
        st.bar_chart(df["risk"].value_counts())

        # -------------------------------
        # EXPLAINABILITY
        # -------------------------------
        st.markdown("### 🧠 Why This Prediction?")

        reasons = []

        if latest["amount"] > 2000:
            reasons.append("High transaction amount")

        if latest["location"] == "International":
            reasons.append("International transaction")

        if latest["prob"] > 0.7:
            reasons.append("Pattern similar to fraud cases")

        if not reasons:
            reasons.append("Transaction behavior appears normal")

        for r in reasons:
            st.write(f"- {r}")

        # -------------------------------
        # HISTORY
        # -------------------------------
        st.markdown("### 📜 Transaction History")
        st.dataframe(df[::-1], use_container_width=True)

    else:
        st.info("No transactions yet.")