import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Set up isolated MLflow environment before importing the module

# Ensure the project root is on the Python path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

#_tmpdir = tempfile.mkdtemp()
#os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")     
#_artifact_dir = os.path.join(_tmpdir, "artifacts")
#os.environ["MLFLOW_ARTIFACT_URI"] = "artifacts"
#os.environ["REGISTERED_MODEL_NAME"] = "TestModel"
#os.environ["MLFLOW_EXPERIMENT"] = "TestExperiment"
#os.makedirs(_artifact_dir, exist_ok=True)

#from src.train_wti_mlflow_fallback import make_lags, make_rolls


import unittest
from src.calculator import add, subtract

class TestCalculator(unittest.TestCase):

    def test_add(self):
        """Test the addition function."""
        result = add(10, 5)
        self.assertEqual(result, 15)  # Assertion 1
        
        result_negatives = add(-1, -1)
        self.assertEqual(result_negatives, -2) # Assertion 2

    def test_subtract(self):
        """Test the subtraction function."""
        result = subtract(10, 5)
        self.assertEqual(result, 5)   # Assertion 1

        result_negative = subtract(5, 10)
        self.assertEqual(result_negative, -5) # Assertion 2

if __name__ == '__main__':
    unittest.main()