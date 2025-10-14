import pandas as pd
import numpy as np
import faiss
import os
import time
import asyncio
import sqlite3
import sys
from langchain_community.embeddings import OpenAIEmbeddings
from tqdm.asyncio import tqdm as async_tqdm
# Ensure project root is on sys.path to import config when running as a script
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OPENAI_API_KEY, FAISS_INDEX_PATH, SQLITE_DB_PATH, DATA_FILE, EMBEDDING_MODEL, DIMENSION

# Конфигурация
BATCH_SIZE = 1000
MAX_CONCURRENT_TASKS = 1

# Инициализация эмбеддингов
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)


async def embed_texts(texts):
    """Асинхронное получение эмбеддингов для списка текстов."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embeddings.embed_documents, texts)


def initialize_metadata_db():
    """Создает таблицу для хранения метаданных, если она не существует."""
    db_dir = os.path.dirname(os.path.abspath(SQLITE_DB_PATH))
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY,
            idea_number TEXT,
            status TEXT,
            title TEXT,
            cause TEXT,
            solution TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_metadata(batch, start_id):
    """Сохраняет метаданные с проверкой существования идентификаторов."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    for i in range(len(batch)):
        current_id = start_id + i
        cursor.execute("SELECT id FROM metadata WHERE id = ?", (current_id,))
        result = cursor.fetchone()

        if result:
            print(f"⚠️ Идентификатор {current_id} уже существует, пропуск...")
            continue

        cursor.execute("""
            INSERT INTO metadata (id, idea_number, status, title, cause, solution)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            current_id,
            str(batch.iloc[i]["Номер Идеи"]),
            str(batch.iloc[i]["Статус Идеи"]),
            str(batch.iloc[i]["Название"]),
            str(batch.iloc[i]["Причина"]),
            str(batch.iloc[i]["Решение"])
        ))

    conn.commit()
    conn.close()



async def process_batch(batch, title_index, cause_index, solution_index, progress_bar, start_id):
    """Асинхронная обработка одного батча данных."""
    start_time = time.time()

    # Векторизация всех трех колонок одновременно
    texts_to_vectorize = batch["Название"].tolist() + batch["Причина"].tolist() + batch["Решение"].tolist()
    vectors = await embed_texts(["" if x is None else (x if isinstance(x, str) else str(x)) for x in texts_to_vectorize])

    # Разделение векторов на три части
    title_vectors = np.array(vectors[:len(batch)]).astype(np.float32)
    cause_vectors = np.array(vectors[len(batch):2*len(batch)]).astype(np.float32)
    solution_vectors = np.array(vectors[2*len(batch):]).astype(np.float32)

    # Добавление векторов в индексы
    title_index.add(title_vectors)
    cause_index.add(cause_vectors)
    solution_index.add(solution_vectors)

    # Сохранение метаданных
    save_metadata(batch, start_id)

    elapsed_time = time.time() - start_time
    progress_bar.update(len(batch))  # Обновляем прогресс-бар
    print(f"✅ Батч обработан за {elapsed_time:.2f} секунд")


def get_max_id():
    """Возвращает максимальный id из таблицы metadata или 0, если таблица пуста."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(id) FROM metadata")
    max_id = cursor.fetchone()[0]
    conn.close()
    return max_id if max_id is not None else 0


async def load_data():
    """Асинхронное наполнение базы данных FAISS и сохранение метаданных в SQLite."""
    print("📥 Загрузка данных из Excel...")
    df = pd.read_excel(DATA_FILE, header=0, engine='openpyxl')
    df = df[['Номер Идеи', 'Название', 'Причина', 'Решение', 'Статус Идеи']].dropna().reset_index(drop=True)
    # Удаляем лишние пробелы в строковых столбцах, избегая устаревшего applymap
    df = df.apply(lambda s: s.str.strip() if s.dtype == object else s)

    # Создание индексов
    title_index = faiss.IndexFlatL2(DIMENSION)
    cause_index = faiss.IndexFlatL2(DIMENSION)
    solution_index = faiss.IndexFlatL2(DIMENSION)

    # Инициализация базы данных SQLite
    initialize_metadata_db()

    print("🚀 Начало наполнения базы данных...")
    tasks = []

    # Прогресс-бар
    total_records = len(df)
    start_id = get_max_id() + 1
    with async_tqdm(total=total_records, desc="🔄 Прогресс загрузки", unit="запись") as progress_bar:
        for i in range(0, total_records, BATCH_SIZE):
            batch = df.iloc[i:i+BATCH_SIZE]
            tasks.append(process_batch(batch, title_index, cause_index, solution_index, progress_bar, start_id=start_id))
            start_id += len(batch)

            # Ограничиваем количество одновременно выполняемых задач
            if len(tasks) >= MAX_CONCURRENT_TASKS:
                await asyncio.gather(*tasks)
                tasks = []

        if tasks:
            await asyncio.gather(*tasks)

    # Сохранение индексов на диск
    if not os.path.exists(FAISS_INDEX_PATH):
        os.makedirs(FAISS_INDEX_PATH)

    faiss.write_index(title_index, os.path.join(FAISS_INDEX_PATH, "title_index.faiss"))
    faiss.write_index(cause_index, os.path.join(FAISS_INDEX_PATH, "cause_index.faiss"))
    faiss.write_index(solution_index, os.path.join(FAISS_INDEX_PATH, "solution_index.faiss"))

    print("✅ Все данные успешно загружены и сохранены!")



if __name__ == "__main__":
    print("📊 Старт асинхронной загрузки данных в базу FAISS...")
    asyncio.run(load_data())
