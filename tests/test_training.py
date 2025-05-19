import pytest
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training import load_data, train_model, save_model

def test_load_data_empty(tmp_path):
    df = pd.DataFrame(columns=["feat1", "feat2", "target"])
    csv_path = tmp_path / "empty.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Input data is empty"):
        load_data(csv_path)

def test_load_data_missing_target(tmp_path):
    df = pd.DataFrame({"feat1": [1], "feat2": [2]})
    csv_path = tmp_path / "missing_target.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(KeyError, match="Expected column 'target' not found in data."):
        load_data(csv_path)

def test_train_model_no_data():
    X = pd.DataFrame(columns=["feat1", "feat2"])
    y = []
    with pytest.raises(ValueError, match="No data to train on."):
        train_model(X, y)

def test_train_model_non_numeric():
    X = pd.DataFrame({"feat1": ["a", "b"], "feat2": ["c", "d"]})
    y = [1, 2]
    with pytest.raises(ValueError, match="All features must be numeric."):
        train_model(X, y)

def test_train_model_ok():
    X = pd.DataFrame({"feat1": [1,2], "feat2": [3,4]})
    y = [5, 6]
    model = train_model(X, y)
    preds = model.predict(X)
    assert len(preds) == 2

def test_save_model(tmp_path):
    from sklearn.linear_model import Ridge
    import joblib
    model = Ridge()
    output_path = tmp_path / "model.pkl"
    save_model(model, output_path)
    assert os.path.exists(output_path)
    loaded = joblib.load(output_path)
    assert hasattr(loaded, "predict")
