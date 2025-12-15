import numpy as np
import pandas as pd
import pytest

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

@pytest.fixture
def sample_data():
    '''Load a small sample of the census data for testing.'''
    data = pd.read_csv("data/census.csv")
    return data.sample(n=200, random_state=42)

@pytest.fixture
def processed_data(sample_data):
    '''Process sample data for training'''
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, _, _ = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    return X, y

def test_train_model_returns_model(processed_data):
    """
    Test that train_model returns a trained model object.
    """
    X, y = processed_data
    model = train_model(X, y)
    assert model is not None


def test_inference_output_shape(processed_data):
    """
    Test that inference returns predictions with correct shape.
    """
    X, y = processed_data
    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y)


def test_compute_model_metrics_range():
    """
    Test that computed metrics are within valid range [0, 1].
    """
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
