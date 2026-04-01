from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


LOGGER_NAME = "startup_churn_classifier.api"


def configure_structured_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


def log_event(event: str, **fields: Any) -> None:
    logger = configure_structured_logging()
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "level": "INFO",
        "event": event,
        **fields,
    }
    logger.info(json.dumps(payload, sort_keys=True))
