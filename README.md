# Classification-and-Logistic-Regression

##  Objective
Build a binary classifier using *Logistic Regression* to predict whether a tumor is *Malignant (M)* or *Benign (B)* based on diagnostic measurements.

---

##  Tools & Libraries
- *Scikit-learn* – model, metrics, preprocessing
- *Pandas* – data handling
- *Matplotlib* & *Seaborn* – visualization
- *Numpy* – math operations

---

##  Dataset
*Breast Cancer Wisconsin (Diagnostic) Dataset*

- Features: 30 numeric columns (e.g., radius, texture, smoothness)
- Target: diagnosis (B = 0, M = 1)

---

## Workflow

### 1. Import & Preprocess
- Dropped columns: id, Unnamed: 32
- Converted diagnosis to numeric:
  - M → 1 (Malignant)
  - B → 0 (Benign)

### 2. Train/Test Split
- Used train_test_split from scikit-learn (80/20 split)

### 3. Feature Scaling
- Standardized features using StandardScaler

### 4. Model Training
- Trained a *Logistic Regression* classifier on the scaled training data

### 5. Model Evaluation
- Evaluated using:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1)
  - ROC Curve & AUC Score

### 6. Threshold Tuning
- Showed how to manually set a probability threshold (e.g., 0.6) for prediction

### 7. Sigmoid Function
- Visualized the *Sigmoid Function* used in logistic regression:
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]

---

## 📊 Example Outputs

- *Accuracy*: ~95%
- *ROC-AUC Score*: > 0.98
- *Confusion Matrix*:
