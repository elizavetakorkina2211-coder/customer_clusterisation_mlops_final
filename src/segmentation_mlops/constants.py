"""Shared column names and training exclusions."""

EVENT_TS_COL = "event_ts"

# Не подаём в модель (время — для дрейфа и аудита; customer_id — идентификатор).
EXCLUDE_FROM_MODEL = frozenset({"segment_truth", EVENT_TS_COL, "customer_id"})
