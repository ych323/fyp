"""Resumable HTTP range download for slow large-file endpoints."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import shutil
import time

import requests


def ranged_download(url, destination, size, workers=8, seed_file=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = 8 * 1024 * 1024
    parts = destination.parent / (destination.name + ".chunks")
    parts.mkdir(exist_ok=True)
    count = (size + chunk_size - 1) // chunk_size
    if seed_file and Path(seed_file).exists():
        # Reuse only complete chunks from this run's previous partial download.
        with open(seed_file, "rb") as source:
            for i in range(min(count, Path(seed_file).stat().st_size // chunk_size)):
                p = parts / f"{i:06d}.chunk"
                data = source.read(chunk_size)
                if not p.exists():
                    p.write_bytes(data)

    def fetch(i):
        begin = i * chunk_size
        end = min(size, begin + chunk_size) - 1
        p = parts / f"{i:06d}.chunk"
        if p.exists() and p.stat().st_size == end - begin + 1:
            return
        for attempt in range(6):
            try:
                # Unique query prevents broken intermediary caches reusing the wrong range.
                sep = "&" if "?" in url else "?"
                with requests.get(url + f"{sep}fyp_chunk={i}", headers={"Range": f"bytes={begin}-{end}"},
                                  stream=True, timeout=(30, 90)) as r:
                    if r.status_code != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {begin}-{end}/"):
                        raise RuntimeError("Server did not honor byte range")
                    with open(p, "wb") as f:
                        for block in r.iter_content(1024 * 1024):
                            f.write(block)
                if p.stat().st_size != end - begin + 1:
                    raise RuntimeError("Incomplete chunk")
                return
            except (requests.RequestException, RuntimeError):
                if attempt == 5:
                    raise RuntimeError(f"Range download failed: chunk {i}") from None
                time.sleep(min(10, 2 ** attempt))

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, i) for i in range(count)]
        for completed, f in enumerate(as_completed(futures), start=1):
            f.result()
            if completed % 8 == 0 or completed == count:
                print(f"{destination.name}: {completed}/{count} chunks, {time.perf_counter() - started:.0f}s", flush=True)
    partial = destination.with_suffix(destination.suffix + ".assembling")
    digest = hashlib.sha256()
    with open(partial, "wb") as out:
        for i in range(count):
            with open(parts / f"{i:06d}.chunk", "rb") as source:
                for block in iter(lambda: source.read(chunk_size), b""):
                    digest.update(block)
                    out.write(block)
    assert partial.stat().st_size == size
    partial.replace(destination)
    # Keep resumable chunks until the caller has validated the final asset.
    return digest.hexdigest()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("destination")
    p.add_argument("--size", required=True, type=int)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed-file")
    a = p.parse_args()
    print("SHA256", ranged_download(a.url, a.destination, a.size, a.workers, a.seed_file), flush=True)
