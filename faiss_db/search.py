import faiss
import os
import numpy as np
import sqlite3
from langchain_community.embeddings import OpenAIEmbeddings  # Обновленная версия
import os
import time
import json
import sys
# Ensure project root is on sys.path to import config when running as a script
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OPENAI_API_KEY, FAISS_INDEX_PATH, SQLITE_DB_PATH, TOP_K, DIMENSION, EMBEDDING_MODEL

# Конфигурация
FAISS_INDEX_PATH = FAISS_INDEX_PATH  # from config
SQLITE_DB_PATH = SQLITE_DB_PATH  # from config
OPENAI_API_KEY =  OPENAI_API_KEY
DIMENSION = DIMENSION  # from config
import config as _cfg
TOP_K = _cfg.TOP_K
TOP_K = 15  # Количество ближайших совпадений


# Инициализация эмбеддингов
TOP_K = _cfg.TOP_K
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)


def convert_to_serializable(data):
    """Преобразует данные в формат, который можно сериализовать в JSON."""
    if isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    else:
        return data


# Глобальный кэш для индексов FAISS
faiss_indices = None  

def load_indices():
    """Загружает индексы FAISS в память один раз при старте."""
    global faiss_indices
    if faiss_indices is None:
        print("📥 Загружаем FAISS индексы в память...")
        faiss_indices = {
            "title": faiss.read_index(os.path.join(FAISS_INDEX_PATH, "title_index.faiss")),
            "cause": faiss.read_index(os.path.join(FAISS_INDEX_PATH, "cause_index.faiss")),
            "solution": faiss.read_index(os.path.join(FAISS_INDEX_PATH, "solution_index.faiss")),
        }
    return faiss_indices


def embed_query(query):
    """Создает векторное представление текстового запроса."""
    query_vector = np.array(embeddings.embed_query(query)).astype(np.float32).reshape(1, -1)
    return query_vector


def search_index(index, query_vector, top_k=TOP_K):
    """Выполняет поиск по индексу FAISS и возвращает ближайшие идентификаторы и расстояния."""
    distances, ids = index.search(query_vector, top_k)
    return ids[0], distances[0]


def get_metadata(ids, distances):
    """Возвращает метаданные для заданных идентификаторов, сортируя по расстояниям."""
    results = []
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    for id_, distance in sorted(zip(ids, distances), key=lambda x: x[1]):
        if id_ == -1:  # Пропускаем недействительные идентификаторы
            continue
        id_ = int(id_) + 1
        cursor.execute("""
            SELECT idea_number, status, title, cause, solution 
            FROM metadata 
            WHERE id = ?
        """, (id_,))
        row = cursor.fetchone()
        # Fallback: если запись не найдена по (faiss_id + 1), пробуем exact faiss_id
        if not row:
            cursor.execute("SELECT idea_number, status, title, cause, solution FROM metadata WHERE id = ?", (id_ - 1,))
            row = cursor.fetchone()
        if row:
            results.append({
                "distance": distance,
                "номер идеи": row[0],
                "статус": row[1],
                "название": row[2],
                "описание": row[3],
                "решение": row[4]
            })

    conn.close()
    return results


def remove_duplicates(metadata):
    """Удаляет дубликаты из списка метаданных."""
    unique_combinations = set()
    unique_results = []

    for record in metadata:
        combination = (record["title"], record["cause"], record["solution"])
        if combination not in unique_combinations:
            unique_combinations.add(combination)
            unique_results.append(record)

    return unique_results


def search_problem(query):
    """Оптимизированная функция поиска по запросу, возвращает JSON-результат."""
    indices = load_indices()  # Используем уже загруженные индексы
    query_vector = embed_query(query)

    # Выполняем поиск по всем индексам
    title_ids, title_distances = search_index(indices["title"], query_vector)
    cause_ids, cause_distances = search_index(indices["cause"], query_vector)
    solution_ids, solution_distances = search_index(indices["solution"], query_vector)

    # Извлекаем метаданные
    title_metadata = get_metadata(title_ids, title_distances)
    cause_metadata = get_metadata(cause_ids, cause_distances)
    solution_metadata = get_metadata(solution_ids, solution_distances)

    # Объединяем результаты и удаляем дубликаты по ключевым полям
    seen = set()
    unique_metadata = []
    for record in title_metadata + cause_metadata + solution_metadata:
        record_tuple = (record["название"], record["описание"], record["решение"])
        if record_tuple not in seen:
            seen.add(record_tuple)
            # Конвертация distance из float32 в float
            record["distance"] = float(record["distance"])
            unique_metadata.append(record)

    # Сортируем по расстоянию (чем меньше, тем лучше)
    sorted_metadata = sorted(unique_metadata, key=lambda x: x["distance"])

    return json.dumps({"проблемы": sorted_metadata}, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    query = input("Введите текстовый запрос: ")
    result_json = search_problem(query)
    print(result_json)
