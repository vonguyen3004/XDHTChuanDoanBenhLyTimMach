# =========================================
# Evaluate MIT-BIH RandomForest Model
# =========================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================================
# 1. DEFINE BASE DIRECTORY
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Dataset paths
train_path = os.path.join(PROJECT_ROOT, "ml", "ECG_Diagnosis_System", "mitbih_train.csv")
test_path = os.path.join(PROJECT_ROOT, "ml", "ECG_Diagnosis_System", "mitbih_test.csv")

# Model path
model_path = os.path.join(PROJECT_ROOT, "backend", "model.joblib")

# =========================================
# 2. CHECK PATHS (DEBUG SAFE)
# =========================================

print("Train path:", train_path)
print("Test path:", test_path)
print("Model path:", model_path)

if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found! Check backend/model.joblib")

# =========================================
# 3. LOAD DATA
# =========================================

print("\nLoading dataset...")

test_df = pd.read_csv(test_path, header=None)

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

print("Dataset loaded successfully!")
print("Test shape:", X_test.shape)

# =========================================
# 4. LOAD MODEL
# =========================================

print("\nLoading model...")
model = joblib.load(model_path)
print("Model loaded successfully!")

# =========================================
# 5. PREDICT
# =========================================

print("\nPredicting...")
y_pred = model.predict(X_test)

# =========================================
# 6. EVALUATION
# =========================================

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n===== RESULTS =====")
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)

# =========================================
# 7. SAVE RESULTS
# =========================================

result_txt_path = os.path.join(BASE_DIR, "mitbih_results.txt")

with open(result_txt_path, "w", encoding="utf-8") as f:
    f.write("=== MIT-BIH RandomForest Evaluation ===\n")
    f.write("Time: " + str(datetime.now()) + "\n\n")
    f.write("Accuracy: " + str(accuracy) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("Results saved to mitbih_results.txt")

# =========================================
# 8. SAVE CONFUSION MATRIX
# =========================================

plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - MIT-BIH RandomForest")
plt.tight_layout()

cm_path = os.path.join(BASE_DIR, "mitbih_confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

print("Confusion matrix saved as mitbih_confusion_matrix.png")

print("\n===== DONE =====")