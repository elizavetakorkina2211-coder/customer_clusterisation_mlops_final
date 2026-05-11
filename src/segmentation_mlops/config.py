from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLOPS_", env_file=".env", extra="ignore")

    root: Path = _default_root()
    model_path: Path | None = None
    db_path: Path | None = None
    mlflow_tracking_uri: str = ""

    def resolved_model_path(self) -> Path:
        if self.model_path:
            return Path(self.model_path)
        return self.root / "models" / "model.joblib"

    def resolved_db_path(self) -> Path:
        if self.db_path:
            return Path(self.db_path)
        return self.root / "data" / "predictions.db"

    def resolved_mlflow_uri(self) -> str:
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        return f"file:{self.root / 'mlruns'}"


def get_settings() -> Settings:
    return Settings()
