#!/usr/bin/env python3
"""Lightweight logging utilities for the pipeline.

Provides a JSON formatter (for machine-friendly logs), a plain formatter
fallback, and helpers to configure the root logger consistently across
modules. Use `get_logger(__name__)` in modules to obtain a configured logger.

Environment variables are not used. Defaults are `plain` format and
`INFO` level unless explicitly set by the pipeline configuration.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """Format log records as compact JSON strings.

    Includes timestamp, level, logger name, message, module/function/line,
    and any `extra={...}` fields provided by the caller.
    """
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        base: Dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        # Attach extras if any (fields added via logger.*(..., extra={...}))
        # Avoid serializing non-JSON types
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in {
                'name','msg','args','levelname','levelno','pathname','filename','module','exc_info',
                'exc_text','stack_info','lineno','funcName','created','msecs','relativeCreated','thread',
                'threadName','processName','process','asctime'
            } and not k.startswith('_')
        }
        if extras:
            base["extra"] = extras
        return json.dumps(base, ensure_ascii=False)


_configured = False


def configure_logging(fmt: Optional[str] = None, level: Optional[str] = None) -> None:
    """Configure the process-wide root logger.

    - If `fmt`/`level` are None, defaults to `plain` and `INFO`.
    - Removes previous handlers to avoid duplicate outputs when called repeatedly.
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    fmt_val = (fmt or "plain").lower()
    if fmt_val == "plain":
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    else:
        handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    # Remove existing handlers to avoid duplication
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel((level or "INFO").upper())
    global _configured
    _configured = True


def _configure_root():
    global _configured
    if _configured:
        return
    configure_logging()


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name, ensuring root is configured."""
    _configure_root()
    return logging.getLogger(name)
