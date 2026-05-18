from __future__ import annotations

import subprocess
import sys

import torch

print("Python:", sys.version.replace("\n", " "))
print("Torch:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}, memory={props.total_memory / 1024**3:.1f} GB")
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("CUDA tensor test: OK", tuple(y.shape), y.dtype)
else:
    print("CUDA tensor test: SKIPPED")

try:
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    print("\n--- nvidia-smi ---")
    print(out.stdout if out.stdout else out.stderr)
except Exception as exc:
    print("\nnvidia-smi not available:", repr(exc))
