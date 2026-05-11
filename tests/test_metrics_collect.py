"""Снимок Prometheus registry для UI: ключи счётчиков совместимы с именем *_total."""

from segmentation_mlops.api import metrics_prom as prom
from segmentation_mlops.api.metrics_collect import collect_mlops_snapshot


def test_prediction_counter_exposed_under_total_and_base_name():
    prom.PREDICTION_REQUESTS.labels(outcome="ok").inc()
    prom.PREDICTION_REQUESTS.labels(outcome="error").inc()
    snap = collect_mlops_snapshot()
    c_short = snap["counters"].get("mlops_prediction_requests")
    c_total = snap["counters"].get("mlops_prediction_requests_total")
    assert c_short is not None and c_total is not None
    assert c_short == c_total
    assert c_total.get("ok", 0) >= 1
    assert c_total.get("error", 0) >= 1
