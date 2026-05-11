import numpy as np

from segmentation_mlops.metrics import confusion_matrix_payload


def test_confusion_matrix_payload_shape_and_order():
    y_true = np.array(["a", "b", "a", "b"])
    y_pred = np.array(["a", "b", "b", "b"])
    labels = ["a", "b"]
    out = confusion_matrix_payload(y_true, y_pred, labels)
    assert out["labels"] == ["a", "b"]
    assert out["matrix"] == [[1, 1], [0, 2]]
