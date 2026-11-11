import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
SEED = 42


def evaluate(name, model, features, target):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    metrics = {
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "average_precision": average_precision_score(target, probabilities),
        "confusion_matrix": confusion_matrix(target, predictions, labels=[0, 1]).tolist(),
    }
    print(f"\n{name}")
    for key in ("precision", "recall", "f1", "average_precision"):
        print(f"{key}: {metrics[key]:.4f}")
    print("Confusion matrix (rows=true, columns=predicted; order 0, 1):")
    print(np.array(metrics["confusion_matrix"]))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate the selected model on the held-out test set")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "random_forest.joblib")
    args = parser.parse_args()
    model_path = args.model_path
    metadata_path = model_path.with_suffix(".json")
    if model_path.exists() or metadata_path.exists():
        parser.error("Model or metadata already exists; choose a new --model-path")
    dataset_path = ROOT / "data" / "raw" / "ai4i2020.csv"
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    data = pd.read_csv(dataset_path)
    x_train, x_temp, y_train, y_temp = train_test_split(
        data[FEATURES], data["Machine failure"], test_size=0.30,
        stratify=data["Machine failure"], random_state=SEED,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED,
    )
    for name, target in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        print(f"{name}: {len(target)} observations, {int(target.sum())} failures")

    candidates = {}
    for name, weight in [("standard", None), ("balanced", "balanced")]:
        preprocessing = ColumnTransformer(
            [("type", OneHotEncoder(handle_unknown="ignore"), ["Type"])],
            remainder="passthrough",
        )
        model = Pipeline([
            ("preprocessing", preprocessing),
            ("classifier", RandomForestClassifier(class_weight=weight, random_state=SEED)),
        ])
        model.fit(x_train, y_train)
        metrics = evaluate(f"Validation: {name}", model, x_val, y_val)
        candidates[name] = (metrics, model)

    # max preserves insertion order on ties, choosing the standard model.
    best_name = max(candidates, key=lambda name: candidates[name][0]["average_precision"])
    best_metrics, best_model = candidates[best_name]
    print(f"\nSelected: {best_name} (validation Average Precision: {best_metrics['average_precision']:.4f})")
    if args.evaluate_test:
        evaluate(f"Test: {best_name}", best_model, x_test, y_test)
    metadata = {
        "selected_model": best_name,
        "classifier_parameters": best_model.named_steps["classifier"].get_params(),
        "features": FEATURES,
        "seed": SEED,
        "dataset_sha256": dataset_hash,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
        "validation_metrics": {name: result[0] for name, result in candidates.items()},
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing artifacts even if another run starts concurrently.
    with model_path.open("xb") as model_file:
        joblib.dump(best_model, model_file)
    try:
        with metadata_path.open("x", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
            metadata_file.write("\n")
    except Exception:
        model_path.unlink()
        raise
    print(f"\nSaved pipeline: {model_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
