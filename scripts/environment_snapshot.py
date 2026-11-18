"""Record the exact installed environment without collecting secrets."""
import json
from pathlib import Path
import platform
import subprocess
import sys

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "environment"
out.mkdir(exist_ok=True)
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
(out / "requirements-lock.txt").write_text(freeze, encoding="utf-8")
info = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
        "ram_gb": psutil.virtual_memory().total / 1e9,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None}
(out / "hardware.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print(json.dumps(info, indent=2), flush=True)
