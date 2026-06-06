from __future__ import annotations

from pathlib import Path


def load_project_env() -> Path:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except Exception:
        pass
    return env_path
