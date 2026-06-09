"""Generate synthetic customer segmentation dataset and reference profile for drift."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from segmentation_mlops.constants import EVENT_TS_COL
from segmentation_mlops.features.build_features import build_features

ROOT = Path(os.getenv("MLOPS_ROOT") or Path(__file__).resolve().parents[3])
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def _features_for_segment(seg: str, rng: np.random.Generator, drift_shift: float) -> dict[str, float]:
    """Признаки из разных распределений по сегменту — классы разделимы, остаётся частичное перекрытие."""
    if seg == "whales":
        days = float(rng.integers(4, 58))
        sessions = float(rng.integers(8, 24))
        basket = float(rng.lognormal(1.62, 0.26))
        cat_div = float(rng.integers(5, 13))
        returns = float(rng.beta(1.1, 24))
        sess_min = float(rng.gamma(3.2, 7.5))
    elif seg == "loyal":
        days = float(rng.integers(22, 128))
        sessions = float(rng.integers(2, 13))
        basket = float(rng.lognormal(1.2, 0.34))
        cat_div = float(rng.integers(2, 11))
        returns = float(rng.beta(2.0, 14))
        sess_min = float(rng.gamma(2.1, 5.8))
    else:
        days = float(rng.integers(78, 178))
        sessions = float(rng.integers(1, 7))
        basket = float(rng.lognormal(0.85, 0.36))
        cat_div = float(rng.integers(1, 9))
        returns = float(rng.beta(2.8, 11))
        sess_min = float(rng.gamma(1.5, 4.2))

    sessions = max(1.0, sessions + drift_shift * rng.normal(0, 1.0))
    days = float(np.clip(days + drift_shift * rng.normal(0, 5.0), 1.0, 179.0))
    basket = float(max(0.35, basket * (1.0 + drift_shift * rng.normal(0, 0.06))))
    cat_div = float(np.clip(cat_div + rng.normal(0, 0.35), 1.0, 12.0))
    return {
        "days_since_last_order": round(days, 2),
        "sessions_last_month": round(sessions, 2),
        "avg_basket_size": round(basket, 3),
        "category_diversity": round(cat_div, 2),
        "discount_share": round(float(rng.uniform(0, 0.5)), 4),
        "returns_rate": round(float(returns), 4),
        "avg_session_minutes": round(sess_min, 2),
    }


def _synthetic_rows(n: int = 9000, seed: int = 42, drift_shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random.seed(seed + 7)

    devices = ["mobile", "desktop", "tablet"]
    platforms = ["ios", "android", "web"]
    channels = ["organic", "paid_search", "social", "email"]
    regions = ["EU", "US", "APAC", "Unknown", np.nan]

    span_seconds = int(120 * 24 * 3600)
    base = pd.Timestamp("2024-06-01", tz="UTC")

    n_each = n // 3
    labels = np.array(
        ["whales"] * n_each + ["loyal"] * n_each + ["casual"] * (n - 2 * n_each),
        dtype=object,
    )
    rng.shuffle(labels)

    rows = []
    for i, seg in enumerate(labels):
        ev_ts = base + pd.Timedelta(seconds=int(rng.integers(0, span_seconds)))
        feat = _features_for_segment(str(seg), rng, drift_shift)
        rows.append(
            {
                "customer_id": f"c_{i}",
                EVENT_TS_COL: ev_ts.isoformat(),
                **feat,
                "device_type": random.choice(devices),
                "platform": random.choice(platforms),
                "marketing_channel": random.choice(channels),
                "region": random.choice(regions),
                "segment_truth": str(seg),
                "orders_last_90d": float(rng.poisson(2)),
                "avg_order_value": float(rng.lognormal(3, 0.3)),
                "gmv_last_90d": float(rng.lognormal(5, 0.5)),
                "customer_value": float(rng.lognormal(4, 0.6)),
            }
        )
    return pd.DataFrame(rows)


def reference_profile(df: pd.DataFrame) -> dict:
    """Summary stats for drift baseline (post feature engineering)."""
    num = [
        "days_since_last_order",
        "sessions_last_month",
        "avg_basket_size",
        "category_diversity",
        "discount_share",
        "returns_rate",
        "avg_session_minutes",
        "order_freq",
        "discount_per_category",
        "basket_per_session",
        "recency_sessions",
        "basket_x_diversity",
        "engagement_minutes",
        "inverse_recency",
        "value_intensity",
    ]
    cat = ["device_type", "platform", "marketing_channel", "region"]
    out = {"numeric": {}, "categorical": {}, "target_dist": {}}
    for c in num:
        if c in df.columns:
            s = df[c].astype(float)
            out["numeric"][c] = {"mean": float(s.mean()), "std": float(s.std() or 1e-6)}
    for c in cat:
        if c in df.columns:
            vc = df[c].fillna("Unknown").value_counts(normalize=True)
            out["categorical"][c] = {str(k): float(v) for k, v in vc.items()}
    if "segment_truth" in df.columns:
        t = df["segment_truth"].value_counts(normalize=True)
        out["target_dist"] = {str(k): float(v) for k, v in t.items()}
    if EVENT_TS_COL in df.columns:
        ts = pd.to_datetime(df[EVENT_TS_COL], utc=True, errors="coerce").dropna()
        if len(ts) > 0:
            out["time"] = {
                "min_utc": ts.min().isoformat(),
                "max_utc": ts.max().isoformat(),
                "n": int(len(ts)),
            }
    return out


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    df_raw = _synthetic_rows(9000, seed=42, drift_shift=0.0)
    path = RAW / "customers_raw.csv"
    df_raw.to_csv(path, index=False)

    df_feat = build_features(df_raw)
    prof = reference_profile(df_feat)
    prof_path = PROC / "reference_profile.json"
    prof_path.write_text(json.dumps(prof, indent=2), encoding="utf-8")
    print(f"Wrote {path} and {prof_path}")
    print(df_raw["segment_truth"].value_counts())


if __name__ == "__main__":
    main()
