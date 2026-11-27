"""Load the compiled components used by the experiment launchers."""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vllm/csrc/c_set"))
import torch
import pod_attn
import c_set

assert hasattr(c_set, "HotPrefixEvictorCore")
assert hasattr(pod_attn, "true_fused_nseg_attn_varlen_func")
assert torch.cuda.is_available()
if sys.argv[1] == "vllm":
    import vllm
    assert Path(vllm.__file__).resolve().is_relative_to(ROOT / "vllm")
else:
    import sglang
    assert Path(sglang.__file__).resolve().is_relative_to(ROOT / "sglang-dev")
    os.environ.update(USE_ORG="1", USE_POD="1")
    importlib.import_module("sglang.srt.layers.attention.fapod_backend")
print(f"Loaded {sys.argv[1]}, pod_attn and c_set with PyTorch {torch.__version__}")
