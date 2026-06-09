import tempfile
from pathlib import Path

from app.observability.monitoring import DriftMonitor


def test_no_drift_when_live_matches_baseline():
    with tempfile.TemporaryDirectory() as tmp:
        m = DriftMonitor(Path(tmp), threshold=0.2)
        labels = ["invoice", "contract", "resume", "support_ticket"] * 5
        m.write_baseline(labels, ["x" * 50] * len(labels))
        for lbl in labels:
            m.record_prediction(lbl, "a document of reasonable length here")
        report = m.drift_report()
        assert report["drift_detected"] is False
        assert report["psi"] < 0.2


def test_drift_detected_on_skew():
    with tempfile.TemporaryDirectory() as tmp:
        m = DriftMonitor(Path(tmp), threshold=0.2)
        m.write_baseline(["invoice", "contract", "resume", "support_ticket"] * 5, ["x" * 50] * 20)
        for _ in range(10):
            m.record_prediction("invoice", "short")
        report = m.drift_report()
        assert report["drift_detected"] is True


def test_corrupt_live_stats_tolerated():
    import json
    with tempfile.TemporaryDirectory() as tmp:
        m = DriftMonitor(Path(tmp), threshold=0.2)
        m.write_baseline(["invoice", "contract"], ["x" * 30, "y" * 30])
        # simulate a truncated/half-written file
        (Path(tmp) / "live_stats.json").write_text('{ "label_counts": { broken', encoding="utf-8")
        # must not raise; recording recovers to a clean state
        m.record_prediction("invoice", "a document of normal length here")
        report = m.drift_report()
        assert report["samples_scored"] == 1
