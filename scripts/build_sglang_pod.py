"""Build POD in an environment-local source tree for the SGLang environment."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "pod_attn_dev/pod_attn"
target = Path(sys.prefix) / "src/asymcache-pod-attn"
target.mkdir(parents=True, exist_ok=True)
shutil.copytree(source, target, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__", "*.so", "*.pyd"))
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation", "-v"],
               cwd=target, check=True)
