"""Tests for idempotent logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from utils.config import Config
from utils.logger import setup_logging


def _logging_config(log_dir: Path) -> Config:
    """Build a minimal logger config for an isolated temporary directory."""
    return Config(
        {
            "logging": {
                "level": "DEBUG",
                "console_level": "CRITICAL",
                "file_level": "DEBUG",
                "log_to_file": True,
                "format": "%(message)s",
                "date_format": "%Y",
                "max_bytes": 100000,
                "backup_count": 1,
            },
            "paths": {"logs_dir": log_dir},
        }
    )


def test_logging_is_idempotent(tmp_path: Path) -> None:
    """Repeated setup should not add duplicate handlers."""
    root_logger = logging.getLogger()
    root_logger._sr_logging_configured = False
    before = len(root_logger.handlers)

    setup_logging(_logging_config(tmp_path), run_name="test")
    after_first = len(root_logger.handlers)
    setup_logging(_logging_config(tmp_path), run_name="test")

    assert after_first >= before
    assert len(root_logger.handlers) == after_first

    logging.getLogger("phase0-test").info("sentinel record")
    log_files = list(tmp_path.glob("test_*.log"))
    assert log_files
    assert "sentinel record" in log_files[0].read_text(encoding="utf-8")
