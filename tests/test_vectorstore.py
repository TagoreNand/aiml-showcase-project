import tempfile
from pathlib import Path

from app.services.vectorstore.memory import InMemoryVectorStore


def _store(tmp, mode="tfidf"):
    return InMemoryVectorStore(Path(tmp) / "corpus.json", mode=mode)


def test_multi_tenant_isolation():
    with tempfile.TemporaryDirectory() as tmp:
        s = _store(tmp)
        s.add("a", "invoice for cloud services", tenant_id="t1")
        s.add("b", "support ticket login failure", tenant_id="t2")
        assert s.count("t1") == 1 and s.count("t2") == 1 and s.count() == 2
        assert {"t1", "t2"} == set(s.tenants())
        # t1 search must not see t2 docs
        res = s.search("login failure", tenant_id="t1")
        assert all(r["document_id"] == "a" for r in res)


def test_dense_mode_fallback_runs():
    with tempfile.TemporaryDirectory() as tmp:
        s = _store(tmp, mode="dense")
        s.add("x", "deployment caused login failure for users")
        s.add("y", "quarterly revenue grew in cloud division")
        res = s.search("login problem after deploy", top_k=1)
        assert res and res[0]["document_id"] == "x"


def test_delete_and_persist():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "corpus.json"
        s = InMemoryVectorStore(path)
        s.add("d1", "hello world document")
        assert s.delete("d1") is True
        # reload from disk -> empty
        s2 = InMemoryVectorStore(path)
        assert s2.count() == 0
