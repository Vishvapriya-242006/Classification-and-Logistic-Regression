import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# 1. Load and preprocess the data
data = pd.read_csv('C:\\Users\\SNU\\Downloads\\data.csv')

# Drop unnecessary columns
data = data.drop(['id', 'Unnamed: 32'], axis=1)

# Convert 'diagnosis' to binary: M=1, B=0
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Split into features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# 2. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Fit logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 5. Evaluate model
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Confusion matrix and classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.2f}")

threshold = 0.6
y_pred_thresh = (y_prob >= threshold).astype(int)
print(f"\nConfusion Matrix at threshold {threshold}:\n", confusion_matrix(y_test, y_pred_thresh))# 6. Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# 7. Threshold tuning example


# 8. Plot sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
sig_vals = sigmoid(z)

plt.figure(figsize=(6,4))
plt.plot(z, sig_vals)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.grid()
plt.show()
