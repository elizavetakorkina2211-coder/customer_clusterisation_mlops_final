"""SQLite storage for prediction audit trail."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import String, Text, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class PredictionRow(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[str] = mapped_column(Text())
    prediction: Mapped[str] = mapped_column(String(64))
    proba_json: Mapped[str] = mapped_column(Text(), default="{}")
    anomaly_flags: Mapped[str] = mapped_column(Text(), default="[]")


def get_engine(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", future=True)


SessionLocal = None


def init_db(db_path: Path) -> None:
    global SessionLocal
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def save_prediction(
    payload: dict,
    prediction: str,
    proba: dict,
    anomaly_flags: list[str],
    db_path: Path,
) -> int:
    if SessionLocal is None:
        init_db(db_path)
    assert SessionLocal is not None
    ts = datetime.now(timezone.utc).isoformat()
    row = PredictionRow(
        created_at=ts,
        payload_json=json.dumps(payload, default=str),
        prediction=prediction,
        proba_json=json.dumps(proba),
        anomaly_flags=json.dumps(anomaly_flags),
    )
    with SessionLocal() as s:
        s.add(row)
        s.commit()
        s.refresh(row)
        return int(row.id)


def list_predictions(db_path: Path, limit: int = 50) -> list[dict]:
    if SessionLocal is None:
        init_db(db_path)
    assert SessionLocal is not None
    with SessionLocal() as s:
        stmt = select(PredictionRow).order_by(PredictionRow.id.desc()).limit(limit)
        rows = s.execute(stmt).scalars().all()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r.id,
                    "created_at": r.created_at,
                    "payload": json.loads(r.payload_json),
                    "prediction": r.prediction,
                    "proba": json.loads(r.proba_json),
                    "anomaly_flags": json.loads(r.anomaly_flags),
                }
            )
        return out


def clear_all_predictions(db_path: Path) -> int:
    """Удаляет все строки из таблицы предсказаний. Возвращает число удалённых записей."""
    if SessionLocal is None:
        init_db(db_path)
    assert SessionLocal is not None
    with SessionLocal() as s:
        res = s.execute(delete(PredictionRow))
        s.commit()
        return int(res.rowcount or 0)


def merge_prediction_flags(prediction_ids: list[int], new_flags: list[str], db_path: Path) -> None:
    """Добавляет флаги к существующим записям (например, после расчёта дрейфа по батчу)."""
    if not prediction_ids or not new_flags:
        return
    if SessionLocal is None:
        init_db(db_path)
    assert SessionLocal is not None
    with SessionLocal() as s:
        for pid in prediction_ids:
            r = s.get(PredictionRow, pid)
            if r is None:
                continue
            old = json.loads(r.anomaly_flags or "[]")
            merged: list[str] = []
            seen: set[str] = set()
            for x in old + new_flags:
                if x not in seen:
                    seen.add(x)
                    merged.append(x)
            r.anomaly_flags = json.dumps(merged, ensure_ascii=False)
        s.commit()
