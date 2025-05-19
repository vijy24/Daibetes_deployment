import os
import requests

# Get the endpoint from the environment variable (set by GitHub Actions)
AZUREML_ENDPOINT = os.getenv("AZUREML_ENDPOINT")

# Sample payload for your diabetes model (adjust to match your model's input schema)
sample_input = {
    "data": [
        [0.038, 0.05, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]
    ]
}

headers = {"Content-Type": "application/json"}

def test_azureml_endpoint_smoke():
    response = requests.post(AZUREML_ENDPOINT, json=sample_input, headers=headers, timeout=10)
    print("Status Code:", response.status_code)
    print("Response:", response.text)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    # Optionally, you can also check the JSON response for the presence of expected fields
    prediction = response.json()
    assert isinstance(prediction, dict), "Response is not a JSON object!"
    print("Smoke test passed.")

if __name__ == "__main__":
    test_azureml_endpoint_smoke()
