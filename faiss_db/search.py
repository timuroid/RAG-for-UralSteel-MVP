import faiss
import numpy as np
import sqlite3
from langchain_community.embeddings import OpenAIEmbeddings  # Обновленная версия
import os
import time
import json
from config import OPENAI_API_KEY

# Конфигурация
FAISS_INDEX_PATH = "./faiss_index"
SQLITE_DB_PATH = "./faiss_index/metadata.db"
OPENAI_API_KEY =  OPENAI_API_KEY
DIMENSION = 1536
TOP_K = 5  # Количество ближайших совпадений


# Инициализация эмбеддингов
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)


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


def load_indices():
    """Загружает все три индекса FAISS с диска."""
    title_index = faiss.read_index(os.path.join(FAISS_INDEX_PATH, "title_index.faiss"))
    cause_index = faiss.read_index(os.path.join(FAISS_INDEX_PATH, "cause_index.faiss"))
    solution_index = faiss.read_index(os.path.join(FAISS_INDEX_PATH, "solution_index.faiss"))
    return title_index, cause_index, solution_index


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
    """Функция поиска по запросу, возвращает результат в формате JSON."""
    title_index, cause_index, solution_index = load_indices()
    query_vector = embed_query(query)

    title_ids, title_distances = search_index(title_index, query_vector)
    cause_ids, cause_distances = search_index(cause_index, query_vector)
    solution_ids, solution_distances = search_index(solution_index, query_vector)

    title_metadata = get_metadata(title_ids, title_distances)
    cause_metadata = get_metadata(cause_ids, cause_distances)
    solution_metadata = get_metadata(solution_ids, solution_distances)

    # Удаление дубликатов
    combined_metadata = title_metadata + cause_metadata + solution_metadata
    unique_metadata = {json.dumps(convert_to_serializable(record), ensure_ascii=False): record
                      for record in combined_metadata}.values()

    # Сортировка уникальных записей по расстоянию (от меньшего к большему), затем реверс
    sorted_metadata = sorted(unique_metadata, key=lambda x: x["distance"], reverse=False)

    result = {"проблемы": list(sorted_metadata)}

    # Преобразование результата в сериализуемый формат
    result_serializable = convert_to_serializable(result)

    return json.dumps(result_serializable, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    query = input("Введите текстовый запрос: ")
    result_json = search_problem(query)
    print(result_json)
