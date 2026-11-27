"""Connect SGLang to the FA2 extension compiled in the vLLM source tree."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "vllm/vllm/vllm_flash_attn"
target = ROOT / "sglang-dev/python/sglang/vllm_flash_attn_org"
libraries = list(source.glob("_vllm_fa2_C*.so"))
if len(libraries) != 1:
    raise SystemExit(f"Build vLLM first: expected one FA2 library under {source}")
if target.is_symlink():
    target.unlink()
elif target.exists():
    raise SystemExit(f"Move the existing directory before creating the link: {target}")
target.symlink_to(source, target_is_directory=True)
print(f"{target} -> {source}")
