from prometheus_client import Counter, Gauge, Histogram

PREDICTION_REQUESTS = Counter(
    "mlops_prediction_requests_total",
    "Inference requests",
    ["outcome"],
)
PREDICTION_LATENCY = Histogram(
    "mlops_prediction_latency_seconds",
    "Inference latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
DRIFT_FLAGS = Gauge(
    "mlops_drift_anomaly_flags",
    "Number of drift flags in last batch report",
)
MODEL_LOADED = Gauge("mlops_model_loaded", "1 if model file is loaded")
BUSINESS_WHALE_RATE = Gauge(
    "mlops_business_whale_share_window",
    "Share of whale predictions in rolling window (UI/metrics)",
)
