"""Async workflow abstraction.

Default backend uses FastAPI ``BackgroundTasks`` (in-process, no broker). When
``TASK_BACKEND=celery`` and Celery+broker are available, work is dispatched to
the queue instead. The API surface (``submit``) is identical either way.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from app.config import Settings

logger = logging.getLogger(__name__)


class TaskRunner:
    backend = "background"

    def submit(self, background_tasks, func: Callable, *args: Any, **kwargs: Any) -> str:
        """Schedule ``func`` to run after the response is returned."""
        if background_tasks is not None:
            background_tasks.add_task(func, *args, **kwargs)
        else:  # no request context -> run synchronously
            func(*args, **kwargs)
        return "scheduled"


class CeleryTaskRunner(TaskRunner):
    backend = "celery"

    def __init__(self):
        from celery import Celery  # type: ignore

        self.app = Celery("docupilot", broker=Settings.CELERY_BROKER_URL)

    def submit(self, background_tasks, func: Callable, *args: Any, **kwargs: Any) -> str:
        # In a real deployment tasks are registered with @app.task and called
        # via .delay(); here we fall back to background execution if the named
        # task isn't registered, keeping the demo functional.
        try:
            self.app.send_task(func.__name__, args=args, kwargs=kwargs)
            return "queued"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Celery dispatch failed (%s); running in background", exc)
            return super().submit(background_tasks, func, *args, **kwargs)


_RUNNER: Optional[TaskRunner] = None


def get_task_runner() -> TaskRunner:
    global _RUNNER
    if _RUNNER is not None:
        return _RUNNER
    if Settings.TASK_BACKEND == "celery":
        try:
            _RUNNER = CeleryTaskRunner()
            return _RUNNER
        except Exception as exc:  # noqa: BLE001
            logger.warning("Celery unavailable (%s); using background tasks", exc)
    _RUNNER = TaskRunner()
    return _RUNNER
