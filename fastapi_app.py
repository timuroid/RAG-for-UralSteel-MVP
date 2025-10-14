from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import json
import openai
import sqlite3
import faiss
import os

from config import OPENAI_API_KEY
from faiss_db.search import search_problem
from chatgpt_handler import generate_final_response
from faiss_db.search import load_indices
from config import SQLITE_DB_PATH, FAISS_INDEX_PATH


openai.api_key = OPENAI_API_KEY


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    token_count: int


app = FastAPI(title="RAG QA Service", version="0.1.0")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse)
def ask(req: QueryRequest):
    user_query = (req.query or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="query must be a non-empty string")

    try:
        # 1) Поиск по FAISS + SQLite
        metadata_json = search_problem(user_query)
        parsed = json.loads(metadata_json)
        # Поддержка разных ключей и форматов (берём первый список из корня)
        metadata_list = []
        if isinstance(parsed, list):
            metadata_list = parsed
        elif isinstance(parsed, dict):
            # Явные ключи + авто‑поиск первого list-значения
            candidates = [
                parsed.get("Данные"),
                parsed.get("данные"),
                parsed.get("проблемы"),
                parsed.get("�஡����"),
                parsed.get("data"),
                parsed.get("results"),
            ]
            metadata_list = next((v for v in candidates if isinstance(v, list)), None)
            if metadata_list is None:
                for v in parsed.values():
                    if isinstance(v, list):
                        metadata_list = v
                        break
            if metadata_list is None:
                metadata_list = []

        # 2) Генерация финального ответа GPT
        final_response, token_count = generate_final_response(metadata_list, user_query)

        return QueryResponse(answer=final_response, token_count=token_count)

    except HTTPException:
        raise
    except Exception as e:
        # Прокидываем как 500, чтобы клиент получил понятную ошибку
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/debug/storage")
def debug_storage():
    """Диагностика: размеры индексов и статистика БД."""
    try:
        idxs = load_indices()
        idx_sizes = {k: v.ntotal for k, v in idxs.items()}
    except Exception as e:
        idx_sizes = {"error": str(e)}

    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cur = conn.cursor()
        mn, mx, cnt = cur.execute("select min(id), max(id), count(*) from metadata").fetchone()
        conn.close()
        db_stats = {"min_id": mn, "max_id": mx, "count": cnt}
    except Exception as e:
        db_stats = {"error": str(e)}

    return {
        "faiss_index_path": FAISS_INDEX_PATH,
        "sqlite_db_path": SQLITE_DB_PATH,
        "index_sizes": idx_sizes,
        "db_stats": db_stats,
    }


@app.get("/debug/search")
def debug_search(q: str = Query(..., description="Запрос для тестового поиска"), k: int = 5):
    """Диагностика: сырые id и расстояния из FAISS и что реально нашлось в БД."""
    try:
        # 1) Выполним обычный поиск, но вернём промежуточные данные
        from faiss_db.search import embed_query
        idxs = load_indices()
        qv = embed_query(q)

        out = {}
        for name, index in idxs.items():
            distances, ids = index.search(qv, k)
            out[name] = {
                "ids": ids[0].tolist(),
                "distances": [float(x) for x in distances[0].tolist()],
            }

        # 2) Пробуем получить записи из БД по (id+1) и по exact id
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cur = conn.cursor()
        details = {}
        for name, payload in out.items():
            found = []
            for raw_id in payload["ids"]:
                if raw_id == -1:
                    found.append({"id": -1, "row": None, "mode": None})
                    continue
                row = cur.execute(
                    "SELECT id, idea_number, status, title, cause, solution FROM metadata WHERE id = ?",
                    (int(raw_id) + 1,),
                ).fetchone()
                mode = "id+1"
                if not row:
                    row = cur.execute(
                        "SELECT id, idea_number, status, title, cause, solution FROM metadata WHERE id = ?",
                        (int(raw_id),),
                    ).fetchone()
                    mode = "id"
                found.append({"id": int(raw_id), "mode": mode, "row": row})
            details[name] = found
        conn.close()

        return {"raw": out, "db_lookup": details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"debug_search error: {e}")


# Локальный запуск: uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
