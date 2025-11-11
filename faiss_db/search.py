import os
import sys
import json
import sqlite3
from typing import Dict

import faiss
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings

# Ensure project root is on sys.path to import config when running as a script
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    OPENAI_API_KEY,
    FAISS_INDEX_PATH,
    SQLITE_DB_PATH,
    TOP_K,
    DIMENSION,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH_2,
    SQLITE_DB_PATH_2,
)


embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

# lazy cache of indices
_default_indices: Dict[str, faiss.Index] = None
_indices_cache: Dict[str, Dict[str, faiss.Index]] = {}


def load_indices():
    global _default_indices
    if _default_indices is None:
        _default_indices = {
            "title": faiss.read_index(os.path.join(FAISS_INDEX_PATH, "title_index.faiss")),
            "cause": faiss.read_index(os.path.join(FAISS_INDEX_PATH, "cause_index.faiss")),
            "solution": faiss.read_index(os.path.join(FAISS_INDEX_PATH, "solution_index.faiss")),
        }
    return _default_indices


def load_indices_by_path(index_path: str):
    if index_path in _indices_cache:
        return _indices_cache[index_path]
    indices = {
        "title": faiss.read_index(os.path.join(index_path, "title_index.faiss")),
        "cause": faiss.read_index(os.path.join(index_path, "cause_index.faiss")),
        "solution": faiss.read_index(os.path.join(index_path, "solution_index.faiss")),
    }
    _indices_cache[index_path] = indices
    return indices


def embed_query(query: str) -> np.ndarray:
    qv = np.array(embeddings.embed_query(query)).astype(np.float32).reshape(1, -1)
    return qv


def search_index(index: faiss.Index, query_vector: np.ndarray, top_k: int = TOP_K):
    distances, ids = index.search(query_vector, top_k)
    return ids[0], distances[0]


def get_metadata(ids, distances):
    results = []
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    for id_, distance in sorted(zip(ids, distances), key=lambda x: x[1]):
        if id_ == -1:
            continue
        id_db = int(id_) + 1
        cursor.execute(
            "SELECT idea_number, status, title, cause, solution FROM metadata WHERE id = ?",
            (id_db,),
        )
        row = cursor.fetchone()
        if not row:
            cursor.execute(
                "SELECT idea_number, status, title, cause, solution FROM metadata WHERE id = ?",
                (id_db - 1,),
            )
            row = cursor.fetchone()
        if row:
            results.append(
                {
                    "distance": float(distance),
                    "Номер Идеи": row[0],
                    "Статус": row[1],
                    "Название": row[2],
                    "Причина": row[3],
                    "Решение": row[4],
                }
            )
    conn.close()
    return results


def get_metadata_from(sqlite_path: str, ids, distances):
    results = []
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    for id_, distance in sorted(zip(ids, distances), key=lambda x: x[1]):
        if id_ == -1:
            continue
        id_db = int(id_) + 1
        cursor.execute(
            "SELECT idea_number, status, title, cause, solution FROM metadata WHERE id = ?",
            (id_db,),
        )
        row = cursor.fetchone()
        if not row:
            cursor.execute(
                "SELECT idea_number, status, title, cause, solution FROM metadata WHERE id = ?",
                (id_db - 1,),
            )
            row = cursor.fetchone()
        if row:
            results.append(
                {
                    "distance": float(distance),
                    "Номер Идеи": row[0],
                    "Статус": row[1],
                    "Название": row[2],
                    "Причина": row[3],
                    "Решение": row[4],
                }
            )
    conn.close()
    return results


def search_problem(query: str):
    indices = load_indices()
    qv = embed_query(query)
    title_ids, title_distances = search_index(indices["title"], qv)
    cause_ids, cause_distances = search_index(indices["cause"], qv)
    solution_ids, solution_distances = search_index(indices["solution"], qv)

    title_metadata = get_metadata(title_ids, title_distances)
    cause_metadata = get_metadata(cause_ids, cause_distances)
    solution_metadata = get_metadata(solution_ids, solution_distances)

    seen = set()
    unique_metadata = []
    for record in title_metadata + cause_metadata + solution_metadata:
        record_tuple = (record["Название"], record["Причина"], record["Решение"])
        if record_tuple not in seen:
            seen.add(record_tuple)
            unique_metadata.append(record)

    sorted_metadata = sorted(unique_metadata, key=lambda x: x["distance"])
    return json.dumps({"results": sorted_metadata}, ensure_ascii=False, indent=4)


def _search_single(query: str, index_path: str, sqlite_path: str, top_k: int = TOP_K):
    indices = load_indices_by_path(index_path)
    qv = embed_query(query)
    title_ids, title_distances = search_index(indices["title"], qv, top_k)
    cause_ids, cause_distances = search_index(indices["cause"], qv, top_k)
    solution_ids, solution_distances = search_index(indices["solution"], qv, top_k)

    title_metadata = get_metadata_from(sqlite_path, title_ids, title_distances)
    cause_metadata = get_metadata_from(sqlite_path, cause_ids, cause_distances)
    solution_metadata = get_metadata_from(sqlite_path, solution_ids, solution_distances)

    seen = set()
    unique_metadata = []
    for record in title_metadata + cause_metadata + solution_metadata:
        key = (record.get("Название"), record.get("Причина"), record.get("Решение"))
        if key not in seen:
            seen.add(key)
            unique_metadata.append(record)
    unique_metadata.sort(key=lambda x: x.get("distance", 0.0))
    return unique_metadata


def search_problem_dual(query: str):
    results_db1 = []
    results_db2 = []
    try:
        results_db1 = _search_single(query, FAISS_INDEX_PATH, SQLITE_DB_PATH, TOP_K)
    except Exception:
        results_db1 = []
    try:
        results_db2 = _search_single(query, FAISS_INDEX_PATH_2, SQLITE_DB_PATH_2, TOP_K)
    except Exception:
        results_db2 = []
    return json.dumps({"results_db1": results_db1, "results_db2": results_db2}, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    q = input("Введите запрос: ")
    print(search_problem(q))

