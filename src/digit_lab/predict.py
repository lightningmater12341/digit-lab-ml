"""Predict a bundled dataset sample or a JSON list of 64 pixel intensities."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_digits


def predict(model_path: Path, pixels: list[float]) -> dict:
    values = np.asarray(pixels, dtype=float)
    if values.shape != (64,) or not np.isfinite(values).all() or not ((0 <= values) & (values <= 16)).all():
        raise ValueError("Expected exactly 64 finite pixel values between 0 and 16")
    model = joblib.load(model_path)  # Load only artifacts you created or trust.
    probabilities = model.predict_proba(values.reshape(1, -1))[0]
    label = int(model.classes_[np.argmax(probabilities)])
    return {"digit": label, "confidence": float(np.max(probabilities)),
            "probabilities": {str(int(k)): float(v) for k, v in zip(model.classes_, probabilities)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("reports/model.joblib"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample", type=int, help="Index in bundled digits dataset (0-1796)")
    source.add_argument("--pixels", type=Path, help="JSON file containing 64 pixel intensities")
    args = parser.parse_args()
    if args.sample is not None:
        digits = load_digits()
        if not 0 <= args.sample < len(digits.data):
            parser.error("sample index must be between 0 and 1796")
        pixels = digits.data[args.sample].tolist()
    else:
        pixels = json.loads(args.pixels.read_text())
    print(json.dumps(predict(args.model, pixels), indent=2))


if __name__ == "__main__":
    main()
