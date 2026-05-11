import pandas as pd

from segmentation_mlops.constants import EVENT_TS_COL
from segmentation_mlops.features.build_features import build_features


def test_build_features_adds_engineered_columns():
    df = pd.DataFrame(
        [
            {
                "days_since_last_order": 10.0,
                "sessions_last_month": 5.0,
                "avg_basket_size": 2.0,
                "category_diversity": 3.0,
                "discount_share": 0.1,
                "returns_rate": 0.02,
                "avg_session_minutes": 8.0,
                "device_type": "mobile",
                "platform": "ios",
                "marketing_channel": "organic",
                "region": "EU",
                "segment_truth": "loyal",
            }
        ]
    )
    out = build_features(df)
    assert "order_freq" in out.columns
    assert "discount_per_category" in out.columns
    assert "basket_per_session" in out.columns
    assert "recency_sessions" in out.columns
    assert "basket_x_diversity" in out.columns
    assert "engagement_minutes" in out.columns
    assert "inverse_recency" in out.columns
    assert "value_intensity" in out.columns


def test_event_ts_parsed_to_datetime():
    df = pd.DataFrame(
        [
            {
                EVENT_TS_COL: "2024-08-15T10:30:00+00:00",
                "days_since_last_order": 10.0,
                "sessions_last_month": 5.0,
                "avg_basket_size": 2.0,
                "category_diversity": 3.0,
                "discount_share": 0.1,
                "returns_rate": 0.02,
                "avg_session_minutes": 8.0,
                "device_type": "mobile",
                "platform": "ios",
                "marketing_channel": "organic",
                "region": "EU",
                "segment_truth": "loyal",
            }
        ]
    )
    out = build_features(df)
    assert pd.api.types.is_datetime64_any_dtype(out[EVENT_TS_COL])
