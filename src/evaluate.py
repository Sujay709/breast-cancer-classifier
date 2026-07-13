import os
 
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
 
MODELS_DIR = "models"
RESULTS_DIR = "results"
 
 
def load_artifacts():
    """Load every model, the scaler, and the held-out test data saved by train.py."""
    models = {
        "Logistic Regression": joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib")),
        "Random Forest": joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib")),
        "Random Forest (Tuned)": joblib.load(os.path.join(MODELS_DIR, "random_forest_tuned.joblib")),
        "SVM": joblib.load(os.path.join(MODELS_DIR, "svm.joblib")),
    }
    test_data = joblib.load(os.path.join(MODELS_DIR, "test_data.joblib"))
    return models, test_data
 
 
def evaluate_model(model, X_test_scaled, y_test, target_names):
    """Return predictions, the classification report (as a dict), and the confusion matrix."""
    y_pred = model.predict(X_test_scaled)
    report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)
    return y_pred, report, cm
 
 
def compare_malignant_recall(reports, save_path):
    """Bar chart of malignant recall across all evaluated models, saved to disk."""
    models = list(reports.keys())
    recalls = [reports[name]["malignant"]["recall"] for name in models]
 
    plt.figure()
    plt.bar(models, recalls)
    plt.ylabel("Malignant Recall")
    plt.title("Model Comparison: Malignant Recall")
    plt.ylim(0.8, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
 
 
def find_common_misclassifications(y_test, preds_by_model):
    """
    Check whether different models misclassified the exact same test sample(s).
    preds_by_model: dict of {model_name: y_pred array}
    """
    mismatch_masks = {
        name: (y_test != preds) for name, preds in preds_by_model.items()
    }
    names = list(mismatch_masks.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            same = np.array_equal(mismatch_masks[names[i]], mismatch_masks[names[j]])
            print(f"{names[i]} vs {names[j]} — same misclassifications: {same}")
 
 
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    models, test_data = load_artifacts()
 
    X_test_scaled = test_data["X_test_scaled"]
    y_test = test_data["y_test"]
    target_names = test_data["target_names"]
 
    reports = {}
    preds_by_model = {}
 
    for name, model in models.items():
        y_pred, report, cm = evaluate_model(model, X_test_scaled, y_test, target_names)
        reports[name] = report
        preds_by_model[name] = y_pred
 
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, target_names=target_names))
        print(f"Confusion matrix:\n{cm}")
 
    print("\n=== Cross-model misclassification check ===")
    find_common_misclassifications(y_test, preds_by_model)
 
    chart_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    compare_malignant_recall(reports, chart_path)
    print(f"\nSaved comparison chart to {chart_path}")
 
 
if __name__ == "__main__":
    main()