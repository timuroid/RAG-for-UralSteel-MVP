import os
import uvicorn


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = _env_bool("RELOAD", True)
    workers = int(os.getenv("WORKERS", "1"))

    # Note: uvicorn reload mode requires an import string
    run_kwargs = dict(host=host, port=port, reload=reload, log_level="info")
    if not reload and workers > 1:
        run_kwargs["workers"] = workers

    # Module path to app instance
    uvicorn.run("fastapi_app:app", **run_kwargs)

    # Alt (no reload):
    # from fastapi_app import app
    # uvicorn.run(app, host=host, port=port, workers=workers)

