import pandas as pd

from segmentation_mlops.constants import EVENT_TS_COL


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess and engineer features."""
    df = df.copy()
    if EVENT_TS_COL in df.columns:
        df[EVENT_TS_COL] = pd.to_datetime(df[EVENT_TS_COL], utc=True, errors="coerce")
    df["region"] = df["region"].fillna("Unknown")

    leak_features = ["orders_last_90d", "avg_order_value", "gmv_last_90d", "customer_value"]
    present = [c for c in leak_features if c in df.columns]
    if present:
        df = df.drop(columns=present)

    d = df["days_since_last_order"].clip(lower=0) + 1.0
    s = df["sessions_last_month"].clip(lower=0)
    df["order_freq"] = s / d
    df["discount_per_category"] = df["discount_share"] / (df["category_diversity"] + 1e-5)
    df["basket_per_session"] = df["avg_basket_size"] / (s + 1e-5)
    # Доп. признаки под сегментацию: активность с учётом давности, «ценность» корзины, вовлечённость по времени
    df["recency_sessions"] = s / d
    df["basket_x_diversity"] = df["avg_basket_size"] * df["category_diversity"]
    df["engagement_minutes"] = s * df["avg_session_minutes"]
    df["inverse_recency"] = 1.0 / d
    df["value_intensity"] = (df["avg_basket_size"] * s) / (df["days_since_last_order"].clip(lower=0) + 40.0)

    return df
