import logging
import os


_CONFIGURED = False


def configure(level: str | None = None) -> None:
    global _CONFIGURED
    log_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    level_value = getattr(logging, log_level, logging.INFO)
    if not _CONFIGURED:
        logging.basicConfig(
            level=level_value,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    logging.getLogger().setLevel(level_value)


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(name)
