"""Download upstream assets from the historical pinned SHA256 manifest only."""
import argparse
import hashlib
import json
from pathlib import Path
import os

import requests

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify existing files without networking')
    parser.add_argument('--include-papers', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'data/provenance/downloads.json').read_text(encoding='utf-8'))
    count = 0
    for job in manifest['files']:
        if job['local_path'].startswith('references/') and not args.include_papers:
            continue
        path = (ROOT / job['local_path']).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Unsafe manifest path')
        url = job['url']
        if not url.startswith(('https://huggingface.co/', 'https://aclanthology.org/', 'https://www.jmlr.org/', 'https://proceedings.mlr.press/')):
            raise ValueError('Unexpected download origin')
        if path.exists():
            if path.stat().st_size != job['bytes'] or sha256(path) != job['sha256']:
                raise RuntimeError('Existing file differs; refusing overwrite: ' + job['local_path'])
            count += 1
            continue
        if args.check_only:
            raise FileNotFoundError(job['local_path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        print('Downloading', job['local_path'], flush=True)
        if job['bytes'] > 100 * 1024 * 1024:
            from ranged_download import ranged_download
            ranged_download(url, path, job['bytes'], workers=8)
        else:
            part = path.with_suffix(path.suffix + '.part')
            # An earlier partial file is deliberately not overwritten.
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with part.open('xb') as out:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            out.write(block)
            if part.stat().st_size != job['bytes'] or sha256(part) != job['sha256']:
                raise RuntimeError('Download mismatch; partial retained: ' + str(part))
            os.replace(part, path)
        if path.stat().st_size != job['bytes'] or sha256(path) != job['sha256']:
            raise RuntimeError('Downloaded asset mismatch: ' + job['local_path'])
        count += 1
    print('Verified', count, 'pinned upstream files. Fine-tuned weights are separate local outputs.')


if __name__ == '__main__':
    main()
