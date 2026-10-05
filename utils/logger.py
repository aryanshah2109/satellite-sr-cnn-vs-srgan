"""Idempotent console and rotating-file logging configuration."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.config import Config

LOGGER = logging.getLogger(__name__)
_SETUP_MARKER = "_sr_logging_configured"
_EXCEPTION_MARKER = "_sr_exception_hook_configured"


def _value(config: Config | dict[str, Any], key: str, default: Any) -> Any:
    """Read a key from a Config or dictionary."""
    if isinstance(config, Config):
        return config[key] if key in config else default

    return config.get(key, default)


def setup_logging(cfg: Config, run_name: str | None = None) -> logging.Logger:
    """Configure the root logger once with console and optional file output.

    Args:
        cfg: Resolved configuration containing logging and path settings.
        run_name: Optional name used in the log filename.

    Returns:
        Configured root logger.
    """
    root_logger = logging.getLogger()
    if getattr(root_logger, _SETUP_MARKER, False):
        return root_logger

    logging_config = cfg.logging
    formatter = logging.Formatter(
        _value(
            logging_config,
            "format",
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        ),
        _value(logging_config, "date_format", "%Y-%m-%d %H:%M:%S"),
    )
    root_logger.setLevel(getattr(logging, _value(logging_config, "level", "INFO")))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(
        getattr(logging, _value(logging_config, "console_level", "INFO"))
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if _value(logging_config, "log_to_file", True):
        logs_dir = Path(cfg.paths.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_name = run_name or "run"
        file_path = logs_dir / f"{safe_name}_{timestamp}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=_value(logging_config, "max_bytes", 10_485_760),
            backupCount=_value(logging_config, "backup_count", 5),
            encoding="utf-8",
        )
        file_handler.setLevel(
            getattr(logging, _value(logging_config, "file_level", "DEBUG"))
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if not getattr(sys, _EXCEPTION_MARKER, False):
        original_hook = sys.excepthook

        def log_uncaught_exception(
            exception_type: type[BaseException],
            exception: BaseException,
            traceback: Any,
        ) -> None:
            """Log uncaught exceptions before delegating to Python."""
            root_logger.critical(
                "Uncaught exception", exc_info=(exception_type, exception, traceback)
            )
            original_hook(exception_type, exception, traceback)

        sys.excepthook = log_uncaught_exception
        setattr(sys, _EXCEPTION_MARKER, True)

    setattr(root_logger, _SETUP_MARKER, True)
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a named module logger."""
    return logging.getLogger(name)
