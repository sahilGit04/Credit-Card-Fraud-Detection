# 💳 Credit Card Fraud Detection System

## 📌 Overview

An end-to-end machine learning system to detect fraudulent credit card transactions using imbalanced classification techniques. The project includes a trained model, FastAPI backend, and an interactive Streamlit dashboard.

---

## ⚠️ Problem

Credit card fraud is rare but highly impactful. Traditional models fail due to **extreme class imbalance (~0.17% fraud cases)**, leading to poor detection of fraudulent transactions.

---

## 💡 Solution

* Applied **SMOTE** to handle class imbalance
* Trained a **Random Forest classifier**
* Built a **FastAPI backend** for predictions
* Developed a **Streamlit dashboard** for real-time monitoring and visualization

---

## 🛠 Tech Stack

* Python, Pandas, NumPy
* Scikit-learn, Imbalanced-learn
* FastAPI (backend API)
* Streamlit (dashboard UI)

---

## 🏗 Architecture

```
Streamlit Dashboard → FastAPI → ML Model → Prediction → UI
```

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python main.py

# Run API
uvicorn src.api:app --reload

# Run dashboard (new terminal)
streamlit run dashboard.py
```

---

## 📊 Features

* Fraud probability prediction
* Real-time dashboard with:

  * Risk levels (Low / Medium / High)
  * Transaction history
  * Risk trend charts
* API endpoint for external integration

---

## 🧪 Sample API Input

```json
{
  "data": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,500]
}
```

---

## 📸 Output

* Confusion Matrix 
* Precision-Recall Curve
* Interactive dashboard with fraud alerts

---

## 🔮 Future Improvements

* Model explainability (SHAP)
* Real-time streaming (Kafka)
* Cloud deployment

---

## 🎯 Use Case

Applicable in banking, fintech, and payment systems for **real-time fraud detection and risk scoring**.
