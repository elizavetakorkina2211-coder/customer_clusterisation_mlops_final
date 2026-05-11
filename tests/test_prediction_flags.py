from segmentation_mlops.drift.prediction_flags import compute_prediction_flags


def test_zscore_flag():
    ref = {
        "numeric": {"sessions_last_month": {"mean": 5.0, "std": 1.0}},
        "categorical": {},
    }
    row = {"sessions_last_month": 50.0}
    flags = compute_prediction_flags(row, {"a": 0.9, "b": 0.1}, ref, {"zscore_threshold": 3.0})
    assert any("Сильное отклонение" in f for f in flags)


def test_low_confidence_flag():
    ref = {"numeric": {}, "categorical": {}}
    flags = compute_prediction_flags(
        {},
        {"x": 0.2, "y": 0.3, "z": 0.5},
        ref,
        {"min_top_proba": 0.55},
    )
    assert any("Низкая уверенность" in f for f in flags)


def test_rare_category_flag():
    ref = {
        "numeric": {},
        "categorical": {"region": {"EU": 0.5, "US": 0.5}},
    }
    flags = compute_prediction_flags({"region": "Mars"}, {"a": 1.0}, ref, {"rare_category_max_p": 0.03})
    assert any("Редкое значение" in f for f in flags)
