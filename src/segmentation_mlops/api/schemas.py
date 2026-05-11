from datetime import datetime

from pydantic import BaseModel, Field


class PredictIn(BaseModel):
    days_since_last_order: float = Field(..., ge=0)
    sessions_last_month: float = Field(..., ge=0)
    avg_basket_size: float = Field(..., ge=0)
    category_diversity: float = Field(..., ge=0)
    discount_share: float = Field(..., ge=0, le=1)
    returns_rate: float = Field(..., ge=0, le=1)
    avg_session_minutes: float = Field(..., ge=0)
    device_type: str
    platform: str
    marketing_channel: str
    region: str
    # Время события (UTC) для временного дрейфа; если не задано — подставляется сервером
    event_ts: datetime | None = None


class PredictOut(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    anomaly_flags: list[str]
    prediction_id: int


class RetrainOut(BaseModel):
    status: str
    detail: str | None = None


class DriftRunOut(BaseModel):
    status: str
    report_path: str | None = None
    flags: list[str] = []
    auto_retrain_triggered: bool = False
