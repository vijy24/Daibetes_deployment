import sys
import os
import pandas as pd
from sklearn.linear_model import Ridge
import joblib

# Optional: Print environment info (helpful for debugging in Azure ML logs)
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Python packages path:", sys.path)

def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} does not exist.")
    data = pd.read_csv(filepath)
    if "target" not in data.columns:
        raise KeyError("Expected column 'target' not found in data.")
    if data.empty:
        raise ValueError("Input data is empty.")
    X = data.drop(columns=["target"])
    y = data["target"]
    return X, y

def train_model(X, y):
    if X.empty or len(y) == 0:
        raise ValueError("No data to train on.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("All features must be numeric.")
    model = Ridge()
    model.fit(X, y)
    return model

def save_model(model, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(model, output_path)
    except Exception as e:
        raise RuntimeError(f"Failed to save model to {output_path}: {str(e)}")

if __name__ == "__main__":
    import argparse
    from azureml.core import Run

    # Parse command line arguments (Azure ML will use these when submitting a job)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="diabetes.csv")
    parser.add_argument("--output", type=str, default="outputs/model.pkl")
    args = parser.parse_args()

    # Load, train, and save model
    X, y = load_data(args.data)
    model = train_model(X, y)
    save_model(model, args.output)

    # Log details to Azure ML run if available
    run = Run.get_context()
    run.log("model_name", "Ridge Regression")
    run.log("train_rows", len(X))
    print("Training and saving model completed successfully.")
