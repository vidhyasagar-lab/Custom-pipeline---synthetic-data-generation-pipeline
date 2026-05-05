"""Structured logging configuration for the Synthetic Data Generation pipeline.
Provides consistent logging across all agents with timing, context, and step tracking.
"""

import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable


# Custom log format with agent/step context
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(agent)-25s | %(step)-30s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class AgentLogFilter(logging.Filter):
    """Filter that adds default agent/step fields if not present."""

    def filter(self, record):
        if not hasattr(record, "agent"):
            record.agent = "system"
        if not hasattr(record, "step"):
            record.step = "-"
        return True


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with agent-aware formatting."""
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers
    root.handlers.clear()

    # Console handler with structured format
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)
    handler.addFilter(AgentLogFilter())
    root.addHandler(handler)

    return root


def get_agent_logger(agent_name: str) -> logging.LoggerAdapter:
    """Get a logger adapter pre-configured with agent name."""
    logger = logging.getLogger(f"pipeline.{agent_name}")
    return AgentLogger(logger, agent_name)


class AgentLogger:
    """Logger wrapper that automatically includes agent context."""

    def __init__(self, logger: logging.Logger, agent_name: str):
        self._logger = logger
        self.agent_name = agent_name
        self._current_step = "-"

    def set_step(self, step: str):
        self._current_step = step

    def _log(self, level: int, msg: str, *args, **kwargs):
        extra = {"agent": self.agent_name, "step": self._current_step}
        self._logger.log(level, msg, *args, extra=extra, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    @contextmanager
    def step(self, step_name: str):
        """Context manager that logs step start/end with timing."""
        self.set_step(step_name)
        self.info(f"▶ Starting: {step_name}")
        start = time.time()
        try:
            yield
            duration = time.time() - start
            self.info(f"✓ Completed: {step_name} ({duration:.2f}s)")
        except Exception as e:
            duration = time.time() - start
            self.error(f"✗ Failed: {step_name} ({duration:.2f}s) - {e}")
            raise
        finally:
            self.set_step("-")

    @contextmanager
    def phase(self, phase_name: str):
        """Context manager for a major phase (logs with separator lines)."""
        self.info("=" * 60)
        self.info(f"PHASE: {phase_name}")
        self.info("=" * 60)
        start = time.time()
        try:
            yield
            duration = time.time() - start
            self.info(f"PHASE COMPLETE: {phase_name} ({duration:.2f}s)")
        except Exception as e:
            duration = time.time() - start
            self.error(f"PHASE FAILED: {phase_name} ({duration:.2f}s) - {e}")
            raise


def log_node(agent_name: str):
    """Decorator for LangGraph nodes that adds automatic logging."""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(state: dict, *args, **kwargs) -> dict:
            logger = get_agent_logger(agent_name)
            node_name = func.__name__
            logger.set_step(node_name)
            logger.info(f"▶ Node entered: {node_name}")
            start = time.time()
            try:
                result = await func(state, *args, **kwargs)
                duration = time.time() - start
                logger.info(f"✓ Node completed: {node_name} ({duration:.2f}s)")
                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(f"✗ Node failed: {node_name} ({duration:.2f}s) - {e}")
                raise

        return wrapper

    return decorator
