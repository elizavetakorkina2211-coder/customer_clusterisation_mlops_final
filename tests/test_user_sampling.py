import json
from pathlib import Path

import pandas as pd

from segmentation_mlops.api.user_sampling import build_user_sampling_config, sample_random_profile_from_csv


def test_build_user_sampling_from_csv(tmp_path: Path):
    rows = []
    for _ in range(100):
        rows.append(
            {
                "days_since_last_order": 30.0,
                "sessions_last_month": 5.0,
                "avg_basket_size": 3.0,
                "category_diversity": 4.0,
                "discount_share": 0.1,
                "returns_rate": 0.05,
                "avg_session_minutes": 12.0,
                "device_type": "mobile",
                "platform": "ios",
                "marketing_channel": "organic",
                "region": "EU",
                "segment_truth": "loyal",
            }
        )
    df = pd.DataFrame(rows)
    p = tmp_path / "data" / "raw" / "customers_raw.csv"
    p.parent.mkdir(parents=True)
    df.to_csv(p, index=False)

    cfg = build_user_sampling_config(tmp_path)
    assert cfg["source"] == "customers_raw.csv"
    assert "days_since_last_order" in cfg["numeric"]
    lo = cfg["numeric"]["days_since_last_order"]["lo"]
    hi = cfg["numeric"]["days_since_last_order"]["hi"]
    assert lo <= hi
    assert "device_type" in cfg["categorical"]
    assert abs(sum(x["p"] for x in cfg["categorical"]["device_type"]) - 1.0) < 1e-6
    assert cfg.get("joint_profiles") == []

    prof = sample_random_profile_from_csv(tmp_path)
    assert prof is not None
    assert prof["days_since_last_order"] == 30.0
    assert prof["device_type"] == "mobile"
    json.dumps(cfg)
