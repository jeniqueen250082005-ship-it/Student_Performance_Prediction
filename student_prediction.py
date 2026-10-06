# ============================================================
# STUDENT PERFORMANCE PREDICTION
# Thiranex - Predictive Modeling Using Machine Learning
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

import joblib


# ============================================================
# 2. CREATE PROJECT FOLDERS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# 3. CREATE STUDENT DATASET
# ============================================================

np.random.seed(42)

number_of_students = 500

study_hours = np.round(
    np.random.uniform(1, 10, number_of_students), 2
)

attendance = np.round(
    np.random.uniform(55, 100, number_of_students), 2
)

previous_score = np.round(
    np.random.uniform(40, 100, number_of_students), 2
)

assignment_score = np.round(
    np.random.uniform(40, 100, number_of_students), 2
)

sleep_hours = np.round(
    np.random.uniform(5, 9, number_of_students), 2
)


# Calculate final score

final_score = (
    (study_hours * 4)
    + (attendance * 0.20)
    + (previous_score * 0.25)
    + (assignment_score * 0.20)
    + (sleep_hours * 1.5)
)

# Add small random variation
final_score = final_score + np.random.normal(
    0, 5, number_of_students
)

# Keep score between 0 and 100
final_score = np.clip(final_score, 0, 100)

final_score = np.round(final_score, 2)


# ============================================================
# 4. CREATE PASS / FAIL TARGET
# ============================================================

# Using median makes sure both classes have enough records.

threshold = np.median(final_score)

performance = np.where(
    final_score >= threshold,
    "Pass",
    "Fail"
)


# ============================================================
# 5. CREATE DATAFRAME
# ============================================================

data = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_score": previous_score,
    "assignment_score": assignment_score,
    "sleep_hours": sleep_hours,
    "final_score": final_score,
    "performance": performance
})


# ============================================================
# 6. SAVE DATASET
# ============================================================

dataset_path = os.path.join(
    DATASET_DIR,
    "student_data.csv"
)

data.to_csv(
    dataset_path,
    index=False
)

print("\n==============================================")
print("STUDENT PERFORMANCE PREDICTION")
print("==============================================")

print("\nDataset created successfully!")
print("Number of students:", len(data))

print("\nFirst 5 records:")
print(data.head())

print("\nClass distribution:")
print(data["performance"].value_counts())

print("\nDataset saved at:")
print(dataset_path)


# ============================================================
# 7. LOAD DATASET
# ============================================================

df = pd.read_csv(dataset_path)


# ============================================================
# 8. REGRESSION MODEL
#    Predict Final Score
# ============================================================

print("\n==============================================")
print("REGRESSION MODEL")
print("==============================================")


# Input features
X_reg = df[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "assignment_score",
        "sleep_hours"
    ]
]

# Target
y_reg = df["final_score"]


# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 9. LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train_reg,
    y_train_reg
)

linear_predictions = linear_model.predict(
    X_test_reg
)


# Regression metrics

linear_mae = mean_absolute_error(
    y_test_reg,
    linear_predictions
)

linear_mse = mean_squared_error(
    y_test_reg,
    linear_predictions
)

linear_rmse = np.sqrt(
    linear_mse
)

linear_r2 = r2_score(
    y_test_reg,
    linear_predictions
)


print("\nLinear Regression Results")

print("MAE :", round(linear_mae, 2))
print("RMSE:", round(linear_rmse, 2))
print("R2 Score:", round(linear_r2, 2))


# Save Linear Regression model

linear_model_path = os.path.join(
    MODEL_DIR,
    "linear_regression_model.pkl"
)

joblib.dump(
    linear_model,
    linear_model_path
)


# ============================================================
# 10. REGRESSION VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_reg,
    linear_predictions,
    alpha=0.7
)

plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")

plt.title(
    "Linear Regression - Actual vs Predicted"
)

plt.grid(True)

plt.tight_layout()

regression_plot_path = os.path.join(
    RESULT_DIR,
    "linear_regression_prediction.png"
)

plt.savefig(
    regression_plot_path,
    dpi=300
)

plt.show()


# ============================================================
# 11. CLASSIFICATION MODEL
#    Predict Pass / Fail
# ============================================================

print("\n==============================================")
print("CLASSIFICATION MODEL")
print("==============================================")


# Input features
X_cls = df[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "assignment_score",
        "sleep_hours"
    ]
]

# Target
y_cls = df["performance"]


# Split data
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls,
    y_cls,
    test_size=0.20,
    random_state=42,
    stratify=y_cls
)


# ============================================================
# 12. DECISION TREE CLASSIFIER
# ============================================================

decision_tree_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

decision_tree_model.fit(
    X_train_cls,
    y_train_cls
)

dt_predictions = decision_tree_model.predict(
    X_test_cls
)

dt_probabilities = decision_tree_model.predict_proba(
    X_test_cls
)


dt_accuracy = accuracy_score(
    y_test_cls,
    dt_predictions
)


print("\nDecision Tree Results")

print(
    "Accuracy:",
    round(dt_accuracy * 100, 2),
    "%"
)


# Save Decision Tree model

dt_model_path = os.path.join(
    MODEL_DIR,
    "decision_tree_model.pkl"
)

joblib.dump(
    decision_tree_model,
    dt_model_path
)


# ============================================================
# 13. DECISION TREE CONFUSION MATRIX
# ============================================================

cm_dt = confusion_matrix(
    y_test_cls,
    dt_predictions,
    labels=["Fail", "Pass"]
)

disp_dt = ConfusionMatrixDisplay(
    confusion_matrix=cm_dt,
    display_labels=["Fail", "Pass"]
)

disp_dt.plot()

plt.title(
    "Decision Tree - Confusion Matrix"
)

plt.tight_layout()

dt_cm_path = os.path.join(
    RESULT_DIR,
    "decision_tree_confusion_matrix.png"
)

plt.savefig(
    dt_cm_path,
    dpi=300
)

plt.show()


# ============================================================
# 14. RANDOM FOREST CLASSIFIER
# ============================================================

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(
    X_train_cls,
    y_train_cls
)

rf_predictions = random_forest_model.predict(
    X_test_cls
)

rf_probabilities = random_forest_model.predict_proba(
    X_test_cls
)


rf_accuracy = accuracy_score(
    y_test_cls,
    rf_predictions
)


print("\nRandom Forest Results")

print(
    "Accuracy:",
    round(rf_accuracy * 100, 2),
    "%"
)


# Save Random Forest model

rf_model_path = os.path.join(
    MODEL_DIR,
    "random_forest_model.pkl"
)

joblib.dump(
    random_forest_model,
    rf_model_path
)


# ============================================================
# 15. RANDOM FOREST CONFUSION MATRIX
# ============================================================

cm_rf = confusion_matrix(
    y_test_cls,
    rf_predictions,
    labels=["Fail", "Pass"]
)

disp_rf = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["Fail", "Pass"]
)

disp_rf.plot()

plt.title(
    "Random Forest - Confusion Matrix"
)

plt.tight_layout()

rf_cm_path = os.path.join(
    RESULT_DIR,
    "random_forest_confusion_matrix.png"
)

plt.savefig(
    rf_cm_path,
    dpi=300
)

plt.show()


# ============================================================
# 16. ROC CURVE - RANDOM FOREST
# ============================================================

# Find probability column corresponding to Pass

pass_index = list(
    random_forest_model.classes_
).index("Pass")

rf_pass_probability = rf_probabilities[
    :, pass_index
]

y_test_binary = (
    y_test_cls == "Pass"
).astype(int)


fpr, tpr, thresholds = roc_curve(
    y_test_binary,
    rf_pass_probability
)

rf_auc = roc_auc_score(
    y_test_binary,
    rf_pass_probability
)


plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {rf_auc:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "Random Forest - ROC Curve"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

roc_path = os.path.join(
    RESULT_DIR,
    "random_forest_roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300
)

plt.show()


# ============================================================
# 17. MODEL COMPARISON
# ============================================================

print("\n==============================================")
print("MODEL COMPARISON")
print("==============================================")


print(
    "\nLinear Regression R2 Score:",
    round(linear_r2, 2)
)

print(
    "Decision Tree Accuracy:",
    round(dt_accuracy * 100, 2),
    "%"
)

print(
    "Random Forest Accuracy:",
    round(rf_accuracy * 100, 2),
    "%"
)

print(
    "Random Forest ROC-AUC:",
    round(rf_auc, 2)
)


# ============================================================
# 18. SAVE MODEL RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Metric": [
        "R2 Score",
        "Accuracy",
        "Accuracy"
    ],
    "Score": [
        round(linear_r2, 4),
        round(dt_accuracy, 4),
        round(rf_accuracy, 4)
    ]
})


results_path = os.path.join(
    RESULT_DIR,
    "model_results.csv"
)

results.to_csv(
    results_path,
    index=False
)


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n==============================================")
print("PROJECT COMPLETED SUCCESSFULLY!")
print("==============================================")

print("\nDataset:")
print(dataset_path)

print("\nModels saved in:")
print(MODEL_DIR)

print("\nResults saved in:")
print(RESULT_DIR)

print("\nGenerated files include:")

print("1. student_data.csv")
print("2. linear_regression_model.pkl")
print("3. decision_tree_model.pkl")
print("4. random_forest_model.pkl")
print("5. linear_regression_prediction.png")
print("6. decision_tree_confusion_matrix.png")
print("7. random_forest_confusion_matrix.png")
print("8. random_forest_roc_curve.png")
print("9. model_results.csv")

print("\n==============================================")
print("READY FOR THIRANEX SUBMISSION")
print("==============================================")