# ================================
# Credit Card Fraud Detection
# ================================

from src.simulation import generate_transactions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("data/creditcard.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# -------------------------------
# 2. Data Preprocessing
# -------------------------------
# Check missing values
print(df.isnull().sum())

# Scale Amount
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Drop Time column (optional)
df = df.drop(['Time'], axis=1)

# -------------------------------
# 3. Feature & Target Split
# -------------------------------
X = df.drop('Class', axis=1)
y = df['Class']

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. Handle Imbalance using SMOTE
# -------------------------------
smote = SMOTE(random_state=42, k_neighbors=1)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", pd.Series(y_train_res).value_counts())

# -------------------------------
# 6. Model Training
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)


importances = model.feature_importances_
features = X.columns

feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop Features:")
print(feat_df.head(10))

# Plot
plt.figure()
sns.barplot(x="Importance", y="Feature", data=feat_df.head(10))
plt.title("Top 10 Important Features")
plt.savefig("images/feature_importance.png")
plt.show()

# -------------------------------
# 7. Predictions
# -------------------------------
y_pred = model.predict(X_test)

def alert_system(prediction):
    if prediction == 1:
        print("🚨 FRAUD ALERT!")
    else:
        print("✅ Transaction Approved")



sample = X_test.iloc[0].values.reshape(1, -1)

prob = model.predict_proba(sample)[0][1]
pred = model.predict(sample)[0]

print(f"Fraud Probability: {prob:.4f}")

if prob > 0.7:
    print("🚨 HIGH RISK FRAUD")
elif prob > 0.3:
    print("⚠️ MEDIUM RISK")
else:
    print("✅ LOW RISK")


# -------------------------------
# 8. Evaluation
# -------------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

# -------------------------------
# Precision-Recall Curve
# -------------------------------
from sklearn.metrics import precision_recall_curve

y_scores = model.predict_proba(X_test)[:, 1]

precision, recall, _ = precision_recall_curve(y_test, y_scores)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig("images/pr_curve.png")
plt.show()

# -------------------------------
# 9. Visualization
# -------------------------------
plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("images/confusion_matrix.png")
plt.show()

# -------------------------------
# 10. Save Model
# -------------------------------
import joblib
joblib.dump(model, "models/fraud_model.pkl")

print("Model saved successfully!")


print("\n--- Simulated Transactions ---")
transactions = generate_transactions(5)

for t in transactions:
    print(t)