"""Restore original local assets without networking or overwriting differences."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def restore(archive, dry_run=False):
    archive = archive.resolve()
    manifest = json.loads((ROOT / 'assets/local_assets_manifest.json').read_text(encoding='utf-8'))
    count = 0
    for entry in manifest['files']:
        if not entry['restore_to_repository']:
            continue
        rel = Path(entry['path'])
        source = (archive / rel).resolve()
        target = (ROOT / rel).resolve()
        if not source.is_relative_to(archive) or not target.is_relative_to(ROOT):
            raise ValueError('Unsafe manifest path')
        if source.stat().st_size != entry['bytes'] or sha256(source) != entry['sha256']:
            raise RuntimeError('Archive checksum mismatch: ' + str(rel))
        if target.exists():
            if sha256(target) != entry['sha256']:
                raise FileExistsError('Refusing to overwrite different file: ' + str(target))
            continue
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha256(target) != entry['sha256']:
                raise RuntimeError('Restored checksum mismatch: ' + str(rel))
        count += 1
    print(('Would restore' if dry_run else 'Restored'), count, 'files; existing identical files reused.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=ROOT.parent / 'local_assets')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    restore(args.archive, args.dry_run)
