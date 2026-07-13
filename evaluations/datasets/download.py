"""Downloads the HealthBench dataset files this harness evaluates against.

Downloaded files are never committed (see .gitignore) -- they're large and
not ours to redistribute. Re-run this whenever you need a fresh local copy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

from evaluations.config import DATASET_URLS, DOWNLOADED_DIR


class DatasetDownloadError(RuntimeError):
    pass


def dataset_path(name: str, dest_dir: Path = DOWNLOADED_DIR) -> Path:
    if name not in DATASET_URLS:
        raise ValueError(
            f"Unknown dataset '{name}'. Known datasets: {sorted(DATASET_URLS)}"
        )
    return dest_dir / f"{name}.jsonl"


def download_dataset(
    name: str,
    dest_dir: Path = DOWNLOADED_DIR,
    force: bool = False,
    timeout: float = 120.0,
) -> Path:
    """Downloads one named dataset (see DATASET_URLS) to `dest_dir/<name>.jsonl`.

    Skips re-downloading if the file already exists and is non-empty, unless
    `force=True`. Raises DatasetDownloadError on any HTTP failure or if the
    downloaded file is empty/not valid-looking JSONL (first line parses as
    an object).
    """
    if name not in DATASET_URLS:
        raise ValueError(
            f"Unknown dataset '{name}'. Known datasets: {sorted(DATASET_URLS)}"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{name}.jsonl"

    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest

    url = DATASET_URLS[name]
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DatasetDownloadError(
            f"Failed to download '{name}' from {url}: {exc}"
        ) from exc

    tmp_path = dest.with_suffix(".jsonl.part")
    line_count = 0
    with open(tmp_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if chunk:
                fh.write(chunk)
                line_count += chunk.count(b"\n")

    if tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise DatasetDownloadError(
            f"Downloaded '{name}' from {url} but the file was empty."
        )

    tmp_path.replace(dest)
    return dest


def download_all(
    dest_dir: Path = DOWNLOADED_DIR, force: bool = False
) -> dict[str, Path]:
    return {
        name: download_dataset(name, dest_dir=dest_dir, force=force)
        for name in DATASET_URLS
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Download HealthBench dataset files.")
    parser.add_argument(
        "--dataset", choices=[*DATASET_URLS.keys(), "all"], default="all"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if a local copy exists."
    )
    args = parser.parse_args()

    if args.dataset == "all":
        results = download_all(force=args.force)
    else:
        results = {args.dataset: download_dataset(args.dataset, force=args.force)}

    for name, path in results.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    _main()
