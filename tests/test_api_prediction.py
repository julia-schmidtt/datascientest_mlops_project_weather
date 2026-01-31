"""
Test API prediction endpoint
Usage: python tests/test_api_prediction.py
"""

import pandas as pd
import requests
import json

print('Test Rain Prediction API')

# 1. Check if API is healthy
token = "super-secret-token"
header = {
    "authorization": f"Bearer {token}"
}
print("1. Checking API health.")
health = requests.get('http://localhost:8000/health', headers=header)
print(f"Status: {health.json()['status']}")
print(f"Model loaded: {health.json()['model_loaded']}")
print(f"Model version: {health.json()['model_version']}\n")

# 2. Get model info
print("2. Getting production model info.")
model_info = requests.get('http://localhost:8000/model/info', headers=header)
info = model_info.json()['model']
print(f"Version: {info['version']}")
print(f"F1 Score: {info['metrics']['f1_score']:.4f}")
print(f"Trained on: {info['params']['years']}\n")

# 3. Load test sample
print("3. Loading test sample from test set.")
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

#sample = X_test.iloc[0].to_dict()
#actual_label = y_test.iloc[0]['RainTomorrow']
sample_idx = X_test.sample(n=1, random_state=None).index[0] # load random sample

sample = X_test.loc[sample_idx].to_dict()
actual_label = y_test.loc[sample_idx, 'RainTomorrow']

print(f"Sample index: {sample_idx}")
print(f"Features: {len(sample)}")
print(f"Actual label: {'Rain' if actual_label == 1 else 'No Rain'}\n")

# 4. Make prediction
print("4. Making prediction.")
response = requests.post('http://localhost:8000/predict', json=sample, headers=header)

if response.status_code == 200:
    result = response.json()
    print(f"Prediction: {result['label']}")
    print(f"Probability Rain: {result['probability_rain']:.2%}")
    print(f"Probability No Rain: {result['probability_no_rain']:.2%}")
    print(f"Model Version: {result['model_version']}")

    # Compare with actual
    predicted = result['prediction']
    correct = "CORRECT" if predicted == actual_label else "INCORRECT"
    print(f"Actual: {'Rain' if actual_label == 1 else 'No Rain'}")
    print(f"Result: {correct}")
else:
    print(f"Error: {response.status_code}")
    print(f"{response.json()}")

print("\n=== Test Complete ===")
