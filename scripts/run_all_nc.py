from __future__ import annotations

import subprocess
import sys


DATASETS = ["Movies", "Toys", "Grocery", "Reddit-S", "ele-fashion", "books-nc"]
MODELS = ["mlp", "gcn", "sage"]


def main() -> None:
    for dataset in DATASETS:
        for model in MODELS:
            cmd = [sys.executable, "-m", "src.main", f"dataset={dataset}", "task=nc", f"model={model}"]
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
