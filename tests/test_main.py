import unittest
from flask import Flask
from flask.testing import FlaskClient
import json
from app.main import app

class TestMain(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()

    def test_predict(self):
        response = self.app.post('/predict', json={'features': [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('prediction', data)

if __name__ == '__main__':
    unittest.main()