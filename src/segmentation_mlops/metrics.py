from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss


def calculate_metrics(y_true, y_pred, y_proba, label_whales: str = "whales"):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_whales": float(f1_score(y_true, y_pred, labels=[label_whales], average="weighted")),
        "log_loss": float(log_loss(y_true, y_proba)),
    }


def confusion_matrix_payload(y_true, y_pred, labels: list) -> dict:
    """Матрица ошибок: строки — истинный класс, столбцы — предсказанный (как в sklearn)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": [str(x) for x in labels],
        "matrix": cm.astype(int).tolist(),
    }
