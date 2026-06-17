"""
Centralised logging configuration.
Creates one rotating file handler per day and a console handler.
"""

import logging
import logging.handlers
from pathlib import Path

from config import LOG_DIR

_INITIALIZED = False


def setup_logging(level: int = logging.INFO) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "mlb_pipeline.log"

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file, when="midnight", utc=True, backupCount=30
    )
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "requests", "pybaseball"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
