"""Preserve stdout/stderr from the sequential study and verification."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
log = ROOT / "logs" / ("pilot_" + stamp + ".log")
log.parent.mkdir(exist_ok=True)
command = [sys.executable, "-u", "run.py", "all", "--config", "configs/pilot.yaml"]
print("Log: " + str(log), flush=True)
with open(log, "w", encoding="utf-8") as f:
    child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in child.stdout:
        print(line, end="", flush=True)
        f.write(line)
        f.flush()
    code = child.wait()
(ROOT / "logs" / "last_run_status.json").write_text(json.dumps({"exit_code": code, "log": str(log),
    "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
sys.exit(code)
