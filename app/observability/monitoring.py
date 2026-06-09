"""Drift + data-quality monitoring.

* Maintains a *baseline* label distribution + text-length stats captured at
  training time (``baseline.json`` under ``METRICS_DIR``).
* Tracks *live* prediction counts and input quality as the API serves traffic
  (persisted to ``live_stats.json``).
* Computes a Population Stability Index (PSI) between baseline and live label
  distributions and flags drift above ``DRIFT_THRESHOLD``.

Pure-Python (numpy only); integrates with Evidently in production if installed.

Robustness: reads tolerate a missing/empty/corrupt JSON file (falling back to
defaults), and writes are atomic (temp file + ``os.replace``) so a concurrent
reader can never observe a half-written file.
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import Settings

_EPS = 1e-6
_DEFAULT_BASELINE = {"label_counts": {}, "n": 0, "avg_length": 0}
_DEFAULT_LIVE = {"label_counts": {}, "n": 0, "length_sum": 0, "short_inputs": 0}


def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    """Read JSON, returning a fresh copy of ``default`` on any error."""
    try:
        if not path.exists():
            return json.loads(json.dumps(default))
        data = json.loads(path.read_text(encoding="utf-8") or "null")
        return data if isinstance(data, dict) else json.loads(json.dumps(default))
    except (json.JSONDecodeError, OSError):
        return json.loads(json.dumps(default))


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON atomically so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _psi(baseline: Dict[str, float], live: Dict[str, float]) -> float:
    categories = set(baseline) | set(live)
    total = 0.0
    for cat in categories:
        b = max(baseline.get(cat, 0.0), _EPS)
        l = max(live.get(cat, 0.0), _EPS)
        total += (l - b) * math.log(l / b)
    return round(total, 4)


def _to_dist(counts: Dict[str, int]) -> Dict[str, float]:
    n = sum(counts.values())
    if n == 0:
        return {}
    return {k: v / n for k, v in counts.items()}


class DriftMonitor:
    def __init__(self, metrics_dir: Optional[Path] = None, threshold: Optional[float] = None):
        self.dir = Path(metrics_dir or Settings.METRICS_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.baseline_path = self.dir / "baseline.json"
        self.live_path = self.dir / "live_stats.json"
        self.threshold = threshold if threshold is not None else Settings.DRIFT_THRESHOLD
        self._lock = threading.Lock()

    # ---- baseline (set at training time) ----------------------------------
    def write_baseline(self, labels: List[str], texts: List[str]) -> Dict[str, Any]:
        lengths = [len(t) for t in texts] or [0]
        baseline = {
            "label_counts": dict(Counter(labels)),
            "n": len(labels),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }
        with self._lock:
            _write_json_atomic(self.baseline_path, baseline)
        return baseline

    def _load_baseline(self) -> Dict[str, Any]:
        return _read_json(self.baseline_path, _DEFAULT_BASELINE)

    # ---- live tracking ----------------------------------------------------
    def _load_live(self) -> Dict[str, Any]:
        return _read_json(self.live_path, _DEFAULT_LIVE)

    def record_prediction(self, label: str, text: str) -> None:
        with self._lock:
            live = self._load_live()
            live["label_counts"][label] = live["label_counts"].get(label, 0) + 1
            live["n"] = live.get("n", 0) + 1
            live["length_sum"] = live.get("length_sum", 0) + len(text)
            if len(text.strip()) < 20:
                live["short_inputs"] = live.get("short_inputs", 0) + 1
            _write_json_atomic(self.live_path, live)

    # ---- reporting --------------------------------------------------------
    def drift_report(self) -> Dict[str, Any]:
        with self._lock:
            baseline = self._load_baseline()
            live = self._load_live()
        base_dist = _to_dist(baseline.get("label_counts", {}))
        live_dist = _to_dist(live.get("label_counts", {}))
        psi = _psi(base_dist, live_dist) if live_dist else 0.0
        n_live = live.get("n", 0)
        avg_live_len = (live.get("length_sum", 0) / n_live) if n_live else 0.0
        return {
            "psi": psi,
            "drift_detected": psi >= self.threshold,
            "threshold": self.threshold,
            "baseline_distribution": {k: round(v, 4) for k, v in base_dist.items()},
            "live_distribution": {k: round(v, 4) for k, v in live_dist.items()},
            "samples_scored": n_live,
            "data_quality": {
                "avg_input_length": round(avg_live_len, 1),
                "baseline_avg_length": round(baseline.get("avg_length", 0), 1),
                "short_input_rate": round(live.get("short_inputs", 0) / n_live, 4) if n_live else 0.0,
            },
        }


_MONITOR: Optional[DriftMonitor] = None


def get_monitor() -> DriftMonitor:
    global _MONITOR
    if _MONITOR is None:
        _MONITOR = DriftMonitor()
    return _MONITOR
