"""
Глобальная конфигурация проекта.

Значения читаются из переменных окружения (или файла .env) и
пробрасываются в виде модульных констант для совместимости с
существующим кодом.

Поддерживаемые переменные окружения:
- OPENAI_API_KEY (обязательно)
- YOUR_TELEGRAM_BOT_TOKEN (опционально, для Telegram‑бота)
- FAISS_INDEX_PATH (по умолчанию ./faiss_index)
- SQLITE_DB_PATH (по умолчанию ./faiss_index/metadata.db)
- DATA_FILE (по умолчанию bd.xlsx)
- EMBEDDING_MODEL (по умолчанию text-embedding-ada-002)
- GPT_MODEL (по умолчанию gpt-4o-2024-08-06)
- DIMENSION (по умолчанию 1536)
- TOP_K (по умолчанию 5)
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Ключи доступа
    OPENAI_API_KEY: str
    YOUR_TELEGRAM_BOT_TOKEN: Optional[str] = None

    # Пути и параметры данных/индексов
    FAISS_INDEX_PATH: str = "./faiss_index"
    SQLITE_DB_PATH: str = "./faiss_index/metadata.db"
    DATA_FILE: str = "bd.xlsx"

    # Модели
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    GPT_MODEL: str = "gpt-5"

    # Гиперпараметры
    DIMENSION: int = 1536
    TOP_K: int = 5

    # Загрузка из .env при наличии
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Экспорт модульных констант для совместимости с существующим кодом
OPENAI_API_KEY = settings.OPENAI_API_KEY
YOUR_TELEGRAM_BOT_TOKEN = settings.YOUR_TELEGRAM_BOT_TOKEN
FAISS_INDEX_PATH = settings.FAISS_INDEX_PATH
SQLITE_DB_PATH = settings.SQLITE_DB_PATH
DATA_FILE = settings.DATA_FILE
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
GPT_MODEL = settings.GPT_MODEL
DIMENSION = settings.DIMENSION
TOP_K = settings.TOP_K
