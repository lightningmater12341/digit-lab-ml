import tempfile
import unittest
from pathlib import Path

from sklearn.datasets import load_digits

from digit_lab.predict import predict
from digit_lab.train import run


class PipelineTest(unittest.TestCase):
    def test_training_and_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = run(output)
            self.assertEqual(report["train_samples"] + report["test_samples"], 1797)
            self.assertGreater(report["test_accuracy"], 0.90)
            sample = load_digits().data[0].tolist()
            result = predict(output / "model.joblib", sample)
            self.assertIn(result["digit"], range(10))
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1)
            with self.assertRaises(ValueError):
                predict(output / "model.joblib", [0] * 63)


if __name__ == "__main__":
    unittest.main()
