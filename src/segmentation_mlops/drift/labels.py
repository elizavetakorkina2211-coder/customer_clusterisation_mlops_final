"""Русские подписи признаков для отчётов и флагов."""

FEATURE_LABEL_RU: dict[str, str] = {
    "days_since_last_order": "дней с последнего заказа",
    "sessions_last_month": "сессий за последний месяц",
    "avg_basket_size": "средний размер корзины",
    "category_diversity": "разнообразие категорий",
    "discount_share": "доля скидок",
    "returns_rate": "доля возвратов",
    "avg_session_minutes": "средняя длительность сессии, мин",
    "order_freq": "частота заказов",
    "discount_per_category": "скидка на категорию",
    "basket_per_session": "корзина на сессию",
    "recency_sessions": "сессий на день с последнего заказа",
    "basket_x_diversity": "корзина × число категорий",
    "engagement_minutes": "сессии × длительность сессии",
    "inverse_recency": "обратная давность заказа",
    "value_intensity": "интенсивность ценности (корзина×сессии / давность)",
    "device_type": "тип устройства",
    "platform": "платформа",
    "marketing_channel": "канал маркетинга",
    "region": "регион",
    "event_ts": "время события",
}

# (имя признака, значение API) → подпись в UI
_CATEGORICAL_VALUE_RU: dict[tuple[str, str], str] = {
    ("device_type", "mobile"): "мобильное",
    ("device_type", "desktop"): "Десктоп",
    ("device_type", "tablet"): "планшет",
    ("platform", "ios"): "iOS",
    ("platform", "android"): "Android",
    ("platform", "web"): "веб",
    ("marketing_channel", "organic"): "органика",
    ("marketing_channel", "paid_search"): "контекстная реклама",
    ("marketing_channel", "social"): "соцсети",
    ("marketing_channel", "email"): "e-mail",
    ("region", "EU"): "ЕС",
    ("region", "US"): "США",
    ("region", "APAC"): "Азиатско-Тихоокеанский регион",
    ("region", "Unknown"): "не указан",
}


def feature_ru(col: str) -> str:
    return FEATURE_LABEL_RU.get(col, col.replace("_", " "))


def feature_title_ru(col: str) -> str:
    """Подпись поля в форме: с заглавной буквы."""
    t = feature_ru(col)
    if not t:
        return col
    return t[0].upper() + t[1:]


def categorical_value_ru(feature: str, value: str) -> str:
    v = str(value).strip()
    if (feature, v) in _CATEGORICAL_VALUE_RU:
        return _CATEGORICAL_VALUE_RU[(feature, v)]
    return _CATEGORICAL_VALUE_RU.get((feature, v.lower()), v)
