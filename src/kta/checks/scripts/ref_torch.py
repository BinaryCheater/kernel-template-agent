"""FP64 (or FP32) reference on the GPU, run inside the worker image with PyTorch.

Self-contained on purpose: it is shipped into the container as a job input file, where
the kta package is not installed. Reads ``ref.json``; for every input set, loads the
inputs, executes the Problem's ``reference_impl`` statement with the tensors and shape
variables in scope, and writes each output as float64 to ``out/<name>.s<i>.bin``.
"""

# pyright: reportMissingImports=false
import json
import sys

import numpy as np
import torch


def main(path: str) -> int:
    spec = json.load(open(path, encoding="utf-8"))
    compute = {"f64": torch.float64, "f32": torch.float32}[spec["compute"]]
    for i, files in enumerate(spec["sets"]):
        scope = {"torch": torch, **spec["shape"]}
        for name, t in spec["inputs"].items():
            arr = np.fromfile(files[name], dtype=t["dtype"]).reshape(t["dims"])
            scope[name] = torch.from_numpy(arr.astype(np.float64)).to("cuda", compute)
        exec(spec["statement"], scope)
        for name in spec["outputs"]:
            out = scope[name].to(torch.float64).cpu().numpy()
            out.tofile(f"out/{name}.s{i}.bin")
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
