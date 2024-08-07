import sys
import os

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from train import load_data, preprocess_data, split_data, train_baseline_model, evaluate_model

class TestTrain(unittest.TestCase):

    def test_load_data(self):
        data = load_data('data/winequality-red.csv')
        self.assertEqual(data.shape[0], 1599)
        self.assertEqual(data.shape[1], 12)

    def test_preprocess_data(self):
        data = load_data('data/winequality-red.csv')
        X, y = preprocess_data(data)
        self.assertEqual(X.shape[1], 11)
        self.assertEqual(y.shape[0], 1599)

    def test_split_data(self):
        data = load_data('data/winequality-red.csv')
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = split_data(X, y)
        self.assertEqual(X_train.shape[0], 1279)
        self.assertEqual(X_test.shape[0], 320)

if __name__ == '__main__':
    unittest.main()