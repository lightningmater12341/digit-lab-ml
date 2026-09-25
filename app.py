"""Explore predictions for handwritten digits from the bundled dataset."""
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_digits

from digit_lab.predict import predict

st.set_page_config(page_title="Digit Lab", page_icon="🔢")
st.title("Digit Lab")
st.caption("Compare real handwritten digit samples with a reproducibly trained classifier.")
model_path = Path(__file__).parent / "reports/model.joblib"
if not model_path.exists():
    st.error("Train the model first: python -m digit_lab.train")
    st.stop()
digits = load_digits()
index = st.slider("Sample index", 0, len(digits.data) - 1, 12)
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(digits.images[index], cmap="gray_r", vmin=0, vmax=16)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)
result = predict(model_path, digits.data[index].tolist())
st.metric("Predicted digit", result["digit"])
st.write(f"True label: **{digits.target[index]}** · Model confidence: **{result['confidence']:.1%}**")
st.bar_chart(result["probabilities"])
st.caption("Confidence is a model score, not a calibrated guarantee. The bundled sample may include training images; use the held-out test report for performance estimates.")
