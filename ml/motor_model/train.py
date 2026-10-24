"""
Training Pipeline for Motor Predictive Condition Classifier.

Trains and serializes a scikit-learn classification pipeline to classify
induction motor operational health as Normal, Warning, or Critical.

[IMPORTANT DISCLAIMER]
All training and evaluation metrics are computed strictly on SYNTHETIC data
generated from idealized physical equations and fault assumptions.
These synthetic test metrics MUST NEVER be represented or cited as
real-world operational performance or field-validated accuracy.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure root package directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.motor_model.data_generator import (
    generate_motor_synthetic_data,
    get_motor_features_and_target,
    MOTOR_FEATURE_NAMES,
    MOTOR_CLASSES
)

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = MODEL_DIR / "motor_model.joblib"


def train_motor_model(
    model_save_path: Path = DEFAULT_MODEL_PATH,
    n_samples_per_class: int = 1500,
    random_seed: int = 42
) -> dict:
    """
    Train and save the motor condition classification pipeline.
    """
    print("=" * 70)
    print("Starting Motor ML Model Training Pipeline")
    print(f"Features: {MOTOR_FEATURE_NAMES}")
    print(f"Target classes: {MOTOR_CLASSES}")
    print("=" * 70)

    # 1. Generate synthetic data
    print("\n[Step 1/5] Generating reproducible synthetic motor telemetry...")
    df = generate_motor_synthetic_data(
        n_samples_per_class=n_samples_per_class,
        random_seed=random_seed
    )
    X, y = get_motor_features_and_target(df)
    print(f"Dataset generated: {len(df)} samples across {df['condition'].nunique()} classes.")

    # 2. Strict train-test split before scaling (ML Best Practice: avoid data leakage)
    print("\n[Step 2/5] Performing stratified 80/20 train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=random_seed,
        stratify=y
    )

    # 3. Build scikit-learn Pipeline with StandardScaler and RandomForestClassifier
    print("\n[Step 3/5] Constructing and fitting scikit-learn Pipeline...")
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=4,
            random_state=random_seed,
            class_weight="balanced"
        ))
    ])

    pipeline.fit(X_train, y_train)

    # 4. Evaluate strictly on held-out test split
    print("\n[Step 4/5] Evaluating model on synthetic held-out test split...")
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, labels=MOTOR_CLASSES)
    report_str = classification_report(y_test, y_pred, labels=MOTOR_CLASSES)
    cm = confusion_matrix(y_test, y_pred, labels=MOTOR_CLASSES)

    print("\n--- SYNTHETIC DATA CLASSIFICATION REPORT ---")
    print(report_str)
    print("Confusion Matrix (Normal, Warning, Critical):")
    print(cm)
    print(f"Synthetic Test Accuracy: {test_accuracy * 100:.2f}%")

    print("\n" + "!" * 70)
    print("[MANDATORY DISCLAIMER]")
    print("The above scores are evaluated on SYNTHETIC data generated from")
    print("simulated operating boundaries. They DO NOT represent real-world")
    print("motor operational accuracy or production performance.")
    print("!" * 70)

    # 5. Save model bundle with explicit metadata
    print(f"\n[Step 5/5] Saving model artifact to {model_save_path}...")
    model_bundle = {
        "pipeline": pipeline,
        "feature_names": MOTOR_FEATURE_NAMES,
        "target_classes": MOTOR_CLASSES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "n_samples": len(df),
        "synthetic_test_accuracy": float(test_accuracy),
        "disclaimer": "Metrics are evaluated strictly on synthetic benchmark data and must not be represented as real-world performance."
    }

    joblib.dump(model_bundle, model_save_path)
    print(f"Model artifact successfully saved to: {model_save_path.resolve()}")

    return model_bundle


if __name__ == "__main__":
    train_motor_model()
