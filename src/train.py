import os
 
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
 
MODELS_DIR = "models"
 
 
def load_data():
    """Load the breast cancer dataset and split into train/test sets."""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    return X_train, X_test, y_train, y_test, data.feature_names, data.target_names
 
 
def train_dummy_baseline(X_train, y_train):
    """Fit the majority-class baseline used to establish the floor a real model must beat."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    return dummy
 
 
def train_logistic_regression(X_train_scaled, y_train):
    """Fit logistic regression on scaled features."""
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model
 
 
def train_random_forest(X_train_scaled, y_train):
    """Fit a default Random Forest classifier."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model
 
 
def tune_random_forest(X_train_scaled, y_train):
    """
    Run GridSearchCV over Random Forest hyperparameters, scoring on malignant
    recall specifically (pos_label=0) rather than sklearn's default, since a
    missed malignant case (false negative) is far worse than a false positive.
    """
    malignant_recall_scorer = make_scorer(recall_score, pos_label=0)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring=malignant_recall_scorer,
    )
    grid_search.fit(X_train_scaled, y_train)
    print(f"Best Random Forest params: {grid_search.best_params_}")
    return grid_search.best_estimator_
 
 
def train_svm(X_train_scaled, y_train):
    """Fit an SVM classifier with probability estimates enabled."""
    model = SVC(probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model
 
 
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
 
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
    print(f"Train/test sizes: {X_train.shape[0]} / {X_test.shape[0]}")
    print(f"Class balance (train): {np.bincount(y_train)}")
 
    # Scale features once, reuse for every model that needs it.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
 
    dummy = train_dummy_baseline(X_train, y_train)
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    rf_model = train_random_forest(X_train_scaled, y_train)
    rf_tuned_model = tune_random_forest(X_train_scaled, y_train)
    svm_model = train_svm(X_train_scaled, y_train)
 
    # Persist everything evaluate.py needs.
    joblib.dump(dummy, os.path.join(MODELS_DIR, "dummy.joblib"))
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest.joblib"))
    joblib.dump(rf_tuned_model, os.path.join(MODELS_DIR, "random_forest_tuned.joblib"))
    joblib.dump(svm_model, os.path.join(MODELS_DIR, "svm.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(
        {
            "X_test": X_test,
            "X_test_scaled": X_test_scaled,
            "y_test": y_test,
            "feature_names": feature_names,
            "target_names": target_names,
        },
        os.path.join(MODELS_DIR, "test_data.joblib"),
    )
 
    print(f"\nSaved 5 models + scaler + test data to {MODELS_DIR}/")
    print("Run evaluate.py next to see the metrics and comparison chart.")
 
 
if __name__ == "__main__":
    main()