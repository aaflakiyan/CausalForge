"""Report local compute readiness without downloading model weights or exposing secrets.

Run: python -m vla_council.causalforge.gpu_preflight --out runs/gpu-preflight.json
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import platform
from pathlib import Path


def inspect_compute() -> dict:
    report = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "mujoco_installed": importlib.util.find_spec("mujoco") is not None,
        "cuda_available": False,
        "gpu_devices": [],
    }
    if report["torch_installed"]:
        import torch
        report["torch_version"] = str(torch.__version__)
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            report["cuda_runtime"] = str(torch.version.cuda)
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                report["gpu_devices"].append({
                    "index": index,
                    "name": props.name,
                    "total_memory_gib": round(props.total_memory / 2**30, 2),
                    "compute_capability": [props.major, props.minor],
                })
            report["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    report["status"] = "gpu_ready" if report["cuda_available"] else "cpu_only"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/gpu-preflight.json"))
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    report = inspect_compute()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.require_gpu and not report["cuda_available"]:
        raise SystemExit("CUDA GPU required but unavailable")


if __name__ == "__main__":
    main()
