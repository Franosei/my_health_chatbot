import requests

import pytest

from evaluations.datasets.download import DatasetDownloadError, download_dataset


class _FakeResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield from self._chunks


def test_download_dataset_raises_on_http_error(tmp_path, monkeypatch):
    def _raise(*a, **kw):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr("evaluations.datasets.download.requests.get", _raise)

    with pytest.raises(DatasetDownloadError):
        download_dataset("healthbench", dest_dir=tmp_path)


def test_download_dataset_writes_file(tmp_path, monkeypatch):
    fake_response = _FakeResponse([b'{"prompt_id": "a"}\n', b'{"prompt_id": "b"}\n'])
    monkeypatch.setattr(
        "evaluations.datasets.download.requests.get", lambda *a, **kw: fake_response
    )

    path = download_dataset("healthbench", dest_dir=tmp_path)

    assert path.exists()
    assert path.read_bytes() == b'{"prompt_id": "a"}\n{"prompt_id": "b"}\n'


def test_download_dataset_skips_existing_file_by_default(tmp_path, monkeypatch):
    existing = tmp_path / "healthbench.jsonl"
    existing.write_text("already here", encoding="utf-8")

    def _should_not_be_called(*a, **kw):
        raise AssertionError(
            "requests.get should not be called when a local copy already exists"
        )

    monkeypatch.setattr(
        "evaluations.datasets.download.requests.get", _should_not_be_called
    )

    path = download_dataset("healthbench", dest_dir=tmp_path)
    assert path.read_text(encoding="utf-8") == "already here"


def test_download_dataset_force_redownloads(tmp_path, monkeypatch):
    existing = tmp_path / "healthbench.jsonl"
    existing.write_text("stale", encoding="utf-8")

    fake_response = _FakeResponse([b'{"prompt_id": "fresh"}\n'])
    monkeypatch.setattr(
        "evaluations.datasets.download.requests.get", lambda *a, **kw: fake_response
    )

    path = download_dataset("healthbench", dest_dir=tmp_path, force=True)
    assert "fresh" in path.read_text(encoding="utf-8")


def test_download_dataset_raises_on_empty_response(tmp_path, monkeypatch):
    fake_response = _FakeResponse([])
    monkeypatch.setattr(
        "evaluations.datasets.download.requests.get", lambda *a, **kw: fake_response
    )

    with pytest.raises(DatasetDownloadError):
        download_dataset("healthbench", dest_dir=tmp_path)


def test_download_dataset_rejects_unknown_dataset(tmp_path):
    with pytest.raises(ValueError):
        download_dataset("not_a_real_dataset", dest_dir=tmp_path)
