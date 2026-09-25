"""Train and evaluate digit classifiers without touching the test set during selection."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, stratify=digits.target,
        random_state=SEED,
    )
    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=1, random_state=SEED, n_jobs=-1
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {
        name: cross_validate(model, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)["test_score"]
        for name, model in candidates.items()
    }
    winner = max(candidates, key=lambda name: np.mean(scores[name]))
    model = candidates[winner].fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = {
        "dataset": "scikit-learn load_digits (8x8 grayscale images)",
        "seed": SEED,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "selection": "5-fold stratified cross-validation on training split only",
        "cv_accuracy": {
            name: {"mean": float(np.mean(value)), "std": float(np.std(value))}
            for name, value in scores.items()
        },
        "selected_model": winner,
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    joblib.dump(model, output / "model.joblib")
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax, cmap="Blues")
    ax.set_title(f"Held-out test set: {winner}")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    report = run(args.output)
    print(f"Selected: {report['selected_model']}; test accuracy: {report['test_accuracy']:.3f}")
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
