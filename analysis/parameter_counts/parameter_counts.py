"""Trainable parameter counts of the 15 methods from their DEG family-split weights, and their Spearman correlation with the mean rank.

Run from the release root in the report environment, with the 15 method environments under ENV_ROOT/egbench-<method>:
python analysis/parameter_counts/parameter_counts.py --env-root ENV_ROOT
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS: tuple[str, ...] = (
    "bingo", "deepcelless", "deephe", "deeplyessential", "deepsnap", "ednn_eg",
    "epgat", "esssubgraph", "esstfnet", "gcnn_sfm", "help", "mahi", "pheg",
    "proteomelm", "xgep",
)
CATEGORY: dict[str, str] = {
    "bingo": "neural", "deepcelless": "neural", "deephe": "neural",
    "deeplyessential": "neural", "epgat": "neural", "esssubgraph": "neural",
    "esstfnet": "neural", "gcnn_sfm": "neural", "proteomelm": "neural",
    "deepsnap": "snapshot_ensemble", "ednn_eg": "stacked_ensemble",
    "help": "gradient_boosting", "mahi": "gradient_boosting", "xgep": "gradient_boosting",
    "pheg": "linear_svm",
}
COHORT = "deg"
SPLIT = "family_5fold"
FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)
BUFFER_NAMES = {"running_mean", "running_var", "num_batches_tracked"}
IMPORT_PATHS: dict[str, tuple[str, ...]] = {
    "help": ("methods/help/code/vendor",),
    "ednn_eg": ("methods/ednn_eg/code/src",),
}
RESULTS = ROOT / "analysis" / "parameter_counts" / "results"
MEAN_RANK = ROOT / "analysis" / "benchmark_report" / "results" / "mean_rank.csv"


def _numel(array) -> int:
    import numpy as np

    return int(np.asarray(array).size)


def count_state_dict(path: Path) -> int:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    elif isinstance(payload, dict):
        state = payload
    else:
        state = payload.state_dict()
    total = 0
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        if not torch.is_floating_point(tensor) or name.split(".")[-1] in BUFFER_NAMES:
            continue
        total += int(tensor.numel())
    return total


def count_lightgbm(model) -> int:
    try:
        dumped = getattr(model, "booster_", model).dump_model()
    except Exception:
        return 0
    return sum(int(tree["num_leaves"]) for tree in dumped["tree_info"])


def count_xgboost(model) -> int:
    try:
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        dumps = booster.get_dump(dump_format="text")
    except Exception:
        return 0
    return sum(line.count("leaf=") for line in dumps)


def count_object(obj, depth: int = 0) -> int:
    if depth > 6:
        return 0
    module = type(obj).__module__ or ""
    name = type(obj).__name__
    if module.startswith("lightgbm"):
        return count_lightgbm(obj)
    if module.startswith("xgboost"):
        return count_xgboost(obj)
    if name in {"DecisionTreeClassifier", "DecisionTreeRegressor", "ExtraTreeClassifier"}:
        tree = getattr(obj, "tree_", None)
        return 0 if tree is None else int(tree.n_leaves)
    if name in {"SVC", "NuSVC"}:
        vectors = getattr(obj, "support_vectors_", None)
        return 0 if vectors is None else _numel(vectors)
    if name in {"LinearSVC", "LogisticRegression", "LinearRegression", "SGDClassifier", "Ridge"}:
        return _numel(obj.coef_) + _numel(obj.intercept_) if hasattr(obj, "coef_") else 0
    if name in {"KNeighborsClassifier", "KNeighborsRegressor"}:
        return 0
    if name == "_SigmoidCalibration":
        return 2
    if isinstance(obj, dict):
        return sum(count_object(value, depth + 1) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(count_object(value, depth + 1) for value in obj)
    if hasattr(obj, "__dict__"):
        return sum(count_object(value, depth + 1) for key, value in vars(obj).items() if not key.startswith("__"))
    return 0


def count_weights(path: Path) -> int:
    if path.is_dir():
        return sum(count_weights(child) for child in sorted(path.iterdir()))
    if path.suffix in {".pt", ".pth"}:
        return count_state_dict(path)
    if path.suffix == ".joblib":
        import joblib

        return count_object(joblib.load(path))
    if path.suffix in {".json", ".ubj"}:
        with path.open("rb") as handle:
            head = handle.read(512)
        if b'"learner"' not in head:
            return 0
        import xgboost

        booster = xgboost.Booster()
        booster.load_model(str(path))
        return count_xgboost(booster)
    return 0


def count_method(method: str) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    for entry in IMPORT_PATHS.get(method, ()):
        sys.path.insert(0, str(ROOT / entry))
    from egbench.data import model_directory

    root = ROOT / "methods" / method
    counts = []
    for fold in FOLDS:
        directory = model_directory(root, COHORT, split_strategy=SPLIT, outer_fold=fold)
        card = json.loads((directory / "model_card.json").read_text())
        named = directory / str(card["artifacts"]["weights"])
        counts.append(count_weights(named if named.is_dir() else named.parent))
    return {"method": method, "counts": counts}


def run_worker(python: Path, method: str) -> dict:
    result = subprocess.run([str(python), str(Path(__file__).resolve()), "--method", method],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{python} --method {method} exited with {result.returncode}\n"
                           f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def drive(env_root: Path) -> None:
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    rows = []
    for method in METHODS:
        python = env_root / f"egbench-{method}" / "bin" / "python"
        if not python.exists():
            raise FileNotFoundError(f"{python} does not exist")
        counts = run_worker(python, method)["counts"]
        rows.append({"method": method, "category": CATEGORY[method],
                     "trainable_parameters": int(np.median(counts)), "minimum": min(counts), "maximum": max(counts),
                     "fold_counts": ";".join(str(c) for c in counts)})
        print(f"[parameters] {method}: median {int(np.median(counts)):,}", flush=True)
    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "parameters.csv", index=False)
    ranks = pd.read_csv(MEAN_RANK).set_index("predictor")["mean_rank"].reindex(table["method"])
    rho = spearmanr(table["trainable_parameters"].to_numpy(), ranks.to_numpy()).statistic
    pd.DataFrame([{"methods": len(table), "spearman": float(rho)}]).to_csv(
        RESULTS / "parameters_mean_rank_spearman.csv", index=False)
    print(f"wrote parameters.csv; Spearman with mean rank {rho:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--env-root", type=Path)
    arguments = parser.parse_args()
    if arguments.method:
        print(json.dumps(count_method(arguments.method)))
        return
    if arguments.env_root is None:
        parser.error("--env-root is required")
    drive(arguments.env_root.expanduser())


if __name__ == "__main__":
    main()
