"""Check publishable files without adding, committing or sending anything."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIRS = ('data/raw/', 'data/processed/', 'data/tokenized/', 'models/hf_cache/', 'logs/', '.venv/', 'venv/')
LOCAL_SUFFIXES = {'.safetensors', '.pt', '.pth', '.bin', '.ckpt', '.onnx', '.h5', '.pdf', '.whl', '.zip', '.pyc', '.part', '.pem', '.key'}


def local_only(rel):
    s = rel.as_posix()
    return (s.startswith(LOCAL_DIRS) or '__pycache__' in rel.parts or '.git' in rel.parts
            or rel.suffix.lower() in LOCAL_SUFFIXES or rel.name.startswith('.env')
            or s == '.publish_check.json' or s.endswith('.assembling')
            or any(x.endswith('.chunks') for x in rel.parts))


def main():
    files = []
    issues = []
    # Also reject forbidden assets if someone force-added them to an existing index.
    if (ROOT / '.git').exists():
        tracked = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z'], capture_output=True, check=True).stdout
        for name in tracked.decode('utf-8').split('\0'):
            if name and local_only(Path(name)):
                issues.append('Forbidden tracked file: ' + name)
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or local_only(path.relative_to(ROOT)):
            continue
        rel = path.relative_to(ROOT).as_posix()
        n = path.stat().st_size
        if n > 25 * 1024 * 1024:
            issues.append('Too large for browser upload: ' + rel)
        if path.suffix in {'.py', '.ps1', '.json', '.md', '.yaml', '.txt'} and n < 2 * 1024 * 1024:
            txt = path.read_text(encoding='utf-8-sig', errors='replace')
            if re.search(r'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16})\b', txt):
                issues.append('Possible secret: ' + rel)
        files.append({'path': rel, 'bytes': n})
    report = {'status': 'failed' if issues else 'passed', 'public_file_count': len(files),
              'public_bytes': sum(x['bytes'] for x in files), 'issues': issues,
              'scope': 'Size, forbidden tracked files and selected token patterns only; not a comprehensive security audit.'}
    print(json.dumps(report, indent=2))
    if issues:
        sys.exit(1)


if __name__ == '__main__':
    main()
