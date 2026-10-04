"""Config and logging setup shared by all jobs."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml and pull secrets from .env into the environment.

    Relative paths in the config are resolved against the project root, so
    jobs behave the same whether run from cron or from a shell elsewhere.
    """
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    load_dotenv(PROJECT_ROOT / ".env")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = path.resolve().parent
    for section, key in (("storage", "db_path"), ("logging", "log_dir")):
        value = cfg.get(section, {}).get(key)
        if value and value != ":memory:" and not Path(value).is_absolute():
            cfg[section][key] = str(base / value)
    return cfg


def setup_logging(cfg: dict, job_name: str) -> None:
    log_cfg = cfg.get("logging", {})
    log_dir = Path(log_cfg.get("log_dir", PROJECT_ROOT / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(
        log_dir / f"{job_name}.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers[:] = [file_handler, console]
    root.setLevel(log_cfg.get("level", "INFO"))
    # PRAW/urllib3 debug logs can include request headers; keep them quiet.
    for noisy in ("prawcore", "praw", "urllib3", "httpx", "huggingface_hub", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
