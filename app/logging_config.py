import logging
import sys

from app.config import settings


def configure_logging() -> None:
    """
    Configures root logging so every log line is timestamped and tagged.
    Follows the same [SERVICE-TAG] convention used in answer-validator-api's
    console logs, so a shared terminal / log aggregator reads consistently
    across all seven LEDGER services.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet down noisy third-party loggers a bit
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
