from segmentation_mlops.api.experiments_display import format_mlflow_run_for_ui


def test_format_mlflow_run_human_labels():
    row = {
        "run_id": "9370a5fbcf2543efb6e7ac0870291e3a",
        "status": "FINISHED",
        "start_time": "2026-04-03 13:42:23.222000+00:00",
        "metrics.f1_whales": 0.6,
        "metrics.accuracy": 0.563,
        "metrics.f1_weighted": 0.55,
        "metrics.log_loss": 0.9,
        "params.n_estimators": 200,
        "params.max_depth": 10,
    }
    out = format_mlflow_run_for_ui(row)
    assert out["status_ru"] == "Завершён"
    assert "9370a5fbcf" in out["run_id_short"]
    labels = [x[0] for x in out["metrics_items"]]
    assert "Точность" in labels
    assert "F1 (взвешенный)" in labels
    assert "F1, класс «киты»" in labels
    assert "Log loss" in labels
    plabels = [x[0] for x in out["params_items"]]
    assert any("деревьев" in p for p in plabels)
