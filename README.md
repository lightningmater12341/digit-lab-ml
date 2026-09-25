# Digit Lab: explainable handwritten digit recognition

A portfolio-ready Python project that compares two classifiers on the [scikit-learn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), selects one using cross-validation, evaluates it on an untouched test split, and serves an interactive Streamlit demo. No external dataset download or API key is needed.

## What it demonstrates

- Stratified 80/20 train/test split with a fixed random seed (42).
- Five-fold cross-validation **on training data only** to compare scaled logistic regression with random forest.
- Held-out accuracy, per-class precision/recall/F1, and a confusion matrix.
- Saved model, validated CLI predictions, interactive sample explorer, and CI test.

## Run locally

Python 3.10+ recommended.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
export PYTHONPATH=src  # PowerShell: $env:PYTHONPATH="src"
python -m digit_lab.train
python -m digit_lab.predict --sample 12
streamlit run app.py
```

For tests: `PYTHONPATH=src python -m unittest discover -s tests` (PowerShell: set `$env:PYTHONPATH="src"` first).

## Evaluation

The generated `reports/metrics.json` contains exact run results and selection scores. The confusion matrix is at `reports/confusion_matrix.png`. Results vary with dependency versions and hardware. The dataset consists of 1,797 small, 8×8 grayscale images; performance here does **not** establish accuracy on phone photos, full-resolution handwriting, or other populations. The demo browses the entire source dataset, including training examples, so its individual examples do not measure generalization.

## Structure

- `src/digit_lab/train.py`: split, model comparison, held-out evaluation, artifacts.
- `src/digit_lab/predict.py`: validated prediction interface.
- `app.py`: interactive Streamlit explorer.
- `tests/`: end-to-end smoke test.
- `.github/workflows/ci.yml`: automated test on pushes and pull requests.

## Next experiments

Try a support vector classifier or augment training images with small transformations; keep the same held-out test split and choose settings using training-only cross-validation.

## License

MIT (see `LICENSE`).
