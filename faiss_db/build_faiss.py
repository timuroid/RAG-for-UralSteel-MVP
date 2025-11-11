import os
import sys
import time
import argparse
import asyncio
import sqlite3
from typing import List

import faiss
import numpy as np
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from tqdm.asyncio import tqdm as async_tqdm

# Ensure project root is on sys.path to import config when running as a script
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    OPENAI_API_KEY,
    FAISS_INDEX_PATH,
    SQLITE_DB_PATH,
    DATA_FILE,
    EMBEDDING_MODEL,
    DIMENSION,
)


BATCH_SIZE = 1000
MAX_CONCURRENT_TASKS = 1

# Embeddings client (global, may be re-initialized in CLI overrides)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)


async def embed_texts(texts: List[str]):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embeddings.embed_documents, texts)


def initialize_metadata_db():
    db_dir = os.path.dirname(os.path.abspath(SQLITE_DB_PATH))
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY,
            idea_number TEXT,
            status TEXT,
            title TEXT,
            cause TEXT,
            solution TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_metadata(batch: pd.DataFrame, start_id: int):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    for i in range(len(batch)):
        current_id = start_id + i
        cursor.execute("SELECT id FROM metadata WHERE id = ?", (current_id,))
        if cursor.fetchone():
            # row already exists — skip
            continue

        cursor.execute(
            """
            INSERT INTO metadata (id, idea_number, status, title, cause, solution)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                current_id,
                str(batch.iloc[i]["Номер Идеи"]),
                str(batch.iloc[i]["Статус Идеи"]),
                str(batch.iloc[i]["Название"]),
                str(batch.iloc[i]["Причина"]),
                str(batch.iloc[i]["Решение"]),
            ),
        )

    conn.commit()
    conn.close()


def get_max_id() -> int:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(id) FROM metadata")
    max_id = cursor.fetchone()[0]
    conn.close()
    return max_id if max_id is not None else 0


async def process_batch(batch: pd.DataFrame, title_index, cause_index, solution_index, progress_bar, start_id: int):
    start_time = time.time()

    texts_to_vectorize = (
        batch["Название"].tolist() + batch["Причина"].tolist() + batch["Решение"].tolist()
    )
    vectors = await embed_texts(
        ["" if x is None else (x if isinstance(x, str) else str(x)) for x in texts_to_vectorize]
    )

    title_vectors = np.array(vectors[: len(batch)]).astype(np.float32)
    cause_vectors = np.array(vectors[len(batch) : 2 * len(batch)]).astype(np.float32)
    solution_vectors = np.array(vectors[2 * len(batch) :]).astype(np.float32)

    title_index.add(title_vectors)
    cause_index.add(cause_vectors)
    solution_index.add(solution_vectors)

    save_metadata(batch, start_id)

    elapsed_time = time.time() - start_time
    progress_bar.update(len(batch))
    print(f"Batch processed in {elapsed_time:.2f}s")


async def load_data():
    print("📥 Загрузка данных из Excel...")
    df = pd.read_excel(DATA_FILE, header=0, engine="openpyxl")
    total_raw = len(df)
    # Нормализация названий колонок (обрезаем пробелы)
    df.columns = [str(c).strip() for c in df.columns]
    try:
        df = df[["Номер Идеи", "Название", "Причина", "Решение", "Статус Идеи"]]
    except KeyError:
        # Fallback: берём первые 5 колонок и принудительно называем
        df = df.iloc[:, :5]
        df.columns = ["Номер Идеи", "Название", "Причина", "Решение", "Статус Идеи"]

    # Заполняем пропуски пустыми строками и чистим пробелы
    for col in ["Название", "Причина", "Решение", "Статус Идеи"]:
        df[col] = df[col].astype(str).fillna("").apply(lambda x: x.strip())
    df["Номер Идеи"] = df["Номер Идеи"].astype(str).apply(lambda x: x.strip())

    # Фильтр: оставляем строки, где хотя бы одно из текстовых полей непустое
    # Учитываем приоритет операций: оборачиваем каждый булев в скобки
    mask_nonempty = (
        (df["Название"].str.len() > 0)
        | (df["Причина"].str.len() > 0)
        | (df["Решение"].str.len() > 0)
    )
    removed_empty = int((~mask_nonempty).sum())
    df = df[mask_nonempty].reset_index(drop=True)
    total_after_clean = len(df)

    print(
        f"Строк в Excel: {total_raw}. После очистки: {total_after_clean}. "
        f"Удалено пустых (без Названия/Причины/Решения): {removed_empty}."
    )

    title_index = faiss.IndexFlatL2(DIMENSION)
    cause_index = faiss.IndexFlatL2(DIMENSION)
    solution_index = faiss.IndexFlatL2(DIMENSION)

    initialize_metadata_db()

    print("🚀 Генерация эмбеддингов и заполнение индексов...")
    tasks = []
    total_records = len(df)
    start_id = get_max_id() + 1
    with async_tqdm(total=total_records, desc="🔄 Обработка записей", unit="строк") as progress_bar:
        for i in range(0, total_records, BATCH_SIZE):
            batch = df.iloc[i : i + BATCH_SIZE]
            tasks.append(
                process_batch(batch, title_index, cause_index, solution_index, progress_bar, start_id=start_id)
            )
            start_id += len(batch)
            if len(tasks) >= MAX_CONCURRENT_TASKS:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)

    if not os.path.exists(FAISS_INDEX_PATH):
        os.makedirs(FAISS_INDEX_PATH)

    faiss.write_index(title_index, os.path.join(FAISS_INDEX_PATH, "title_index.faiss"))
    faiss.write_index(cause_index, os.path.join(FAISS_INDEX_PATH, "cause_index.faiss"))
    faiss.write_index(solution_index, os.path.join(FAISS_INDEX_PATH, "solution_index.faiss"))
    print("✅ Индексы и метаданные обновлены.")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS indexes and SQLite metadata from an Excel file.")
    parser.add_argument("--data-file", dest="data_file", default=None, help="Path to Excel source (e.g., bd.xlsx)")
    parser.add_argument("--faiss-index-path", dest="faiss_index_path", default=None, help="Directory to write FAISS indexes")
    parser.add_argument("--sqlite-db-path", dest="sqlite_db_path", default=None, help="Path to SQLite metadata DB")
    parser.add_argument("--embedding-model", dest="embedding_model", default=None, help="Embedding model id")
    parser.add_argument("--dimension", dest="dimension", type=int, default=None, help="Embedding vector dimension")
    args = parser.parse_args()

    global DATA_FILE, FAISS_INDEX_PATH, SQLITE_DB_PATH, EMBEDDING_MODEL, DIMENSION, embeddings
    if args.data_file:
        DATA_FILE = args.data_file
    if args.faiss_index_path:
        FAISS_INDEX_PATH = args.faiss_index_path
    if args.sqlite_db_path:
        SQLITE_DB_PATH = args.sqlite_db_path
    if args.embedding_model:
        EMBEDDING_MODEL = args.embedding_model
    if args.dimension is not None:
        DIMENSION = args.dimension

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

    asyncio.run(load_data())


if __name__ == "__main__":
    print("Starting FAISS build script...")
    main()
