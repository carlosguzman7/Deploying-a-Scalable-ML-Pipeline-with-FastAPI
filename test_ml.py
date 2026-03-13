import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics


def test_apply_labels():
    """
    Test that the label array is valid and contains binary values.
    """
    y = np.array([0, 1, 1, 0, 1])

    assert len(y) > 0
    assert set(y).issubset({0, 1})


def test_train_model():
    """
    Test that train_model returns a trained RandomForestClassifier.
    """
    X = np.array([
        [0, 1, 2],
        [1, 0, 1],
        [1, 1, 0],
        [0, 0, 1]
    ])

    y = np.array([0, 1, 1, 0])

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns valid precision, recall, and fbeta values.
    """
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1

