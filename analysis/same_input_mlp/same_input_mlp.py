"""Replace Mahi's and HELP's gradient boosted trees by multilayer perceptrons of four widths on the same per-fold inputs (GPU).

Run from the release root in the DeepHE environment: python analysis/same_input_mlp/same_input_mlp.py [--gpus 2,3] [--workers 8]
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import numpy as np
import pandas as pd

from egbench.contracts import SEED
from egbench.data import TRAIN_ROLE, VALIDATION_ROLE, feature_directory, prediction_cohort, read_training_labels
from egbench.metrics.discrimination import auprc

from common.stats import (
    ALL_COHORTS, ANALYSIS_DIRECTORY, FOLD, FOLDS, GENE_KEY, METHODS_DIRECTORY, SCORE, SPLIT, Progress, holm,
    interval, kind_of, paired, primary_metric, target_of, test_frame, write_table,
)

RESULTS = ANALYSIS_DIRECTORY / "same_input_mlp" / "results"
METHODS: tuple[str, ...] = ("mahi", "help")
SOURCES: tuple[str, ...] = ("deg", "depmap", "context", "blood", "mouse")
SIZES: tuple[tuple[int, int], ...] = ((16, 4), (256, 64), (1024, 256), (4096, 1024))
REGULARISATION = tuple(product((0.2, 0.5), (1e-4, 1e-2)))
LEARNING_RATE = 1e-3
BATCH = 256
MAX_EPOCHS = 200
PATIENCE = 20
THREADS = 2


def feature_matrix(method: str, cohort: str, fold: int) -> pd.DataFrame:
    directory = feature_directory(METHODS_DIRECTORY / method, cohort, split_strategy=SPLIT, outer_fold=fold)
    frame = pd.read_parquet(directory / "features.parquet")
    if method == "mahi":
        values = np.stack(frame["embedding"].to_numpy()).astype("float64")
        return pd.DataFrame(values, index=pd.Index(frame[GENE_KEY].astype("object")))
    return frame.set_index(GENE_KEY).astype("float64")


def rows_of(matrix: pd.DataFrame, keys: pd.Index, what: str) -> np.ndarray:
    values = matrix.reindex(keys).to_numpy("float64")
    if not np.isfinite(values).all():
        raise ValueError(f"{what}: a gene has no feature row or a non-finite value")
    return values


def perceptron(width: int, hidden: tuple[int, int], dropout: float):
    import torch

    layers, previous = [], width
    for size in hidden:
        layers += [torch.nn.Linear(previous, size), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        previous = size
    layers.append(torch.nn.Linear(previous, 1))
    return torch.nn.Sequential(*layers)


def parameter_count(width: int, hidden: tuple[int, int]) -> int:
    sizes = (width, *hidden, 1)
    return int(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:])))


def size_name(hidden: tuple[int, int]) -> str:
    return "x".join(str(size) for size in hidden)


def fit_one(x, y, x_validation, y_validation, config, device):
    import torch

    hidden, dropout, decay = config
    torch.manual_seed(SEED)
    model = perceptron(x.shape[1], hidden, dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=decay)
    positives = float(y.sum())
    weight = torch.tensor((len(y) - positives) / positives, dtype=torch.float32, device=device)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    features = torch.tensor(x, dtype=torch.float32, device=device)
    labels = torch.tensor(y, dtype=torch.float32, device=device)
    validation = torch.tensor(x_validation, dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(SEED)
    best, best_state, best_epoch, waited = -1.0, None, -1, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(features), generator=generator).to(device)
        for start in range(0, len(features), BATCH):
            batch = order[start:start + BATCH]
            optimiser.zero_grad()
            loss = loss_function(model(features[batch]).squeeze(1), labels[batch])
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            scores = model(validation).squeeze(1).cpu().numpy().astype("float64")
        value = float(auprc(y_validation, scores))
        if value > best:
            best, best_epoch, waited = value, epoch, 0
            best_state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, best_epoch


def fit_job(job: tuple[str, str, int, int]) -> dict:
    method, source, fold, gpu = job
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(THREADS)
    device = torch.device(f"cuda:{gpu}")
    matrix = feature_matrix(method, source, fold)
    train = read_training_labels(source, fold, TRAIN_ROLE, split_strategy=SPLIT)
    validation = read_training_labels(source, fold, VALIDATION_ROLE, split_strategy=SPLIT)
    what = f"{method} {source} fold {fold}"
    x = rows_of(matrix, pd.Index(train[GENE_KEY].astype("object")), what + " train")
    center, spread = x.mean(axis=0), x.std(axis=0)
    keep = spread > 0
    center, spread = center[keep], spread[keep]
    x = (x[:, keep] - center) / spread
    y = train["label"].to_numpy("float64")
    x_validation = (rows_of(matrix, pd.Index(validation[GENE_KEY].astype("object")), what + " validation")[:, keep]
                    - center) / spread
    y_validation = validation["label"].to_numpy("float64")
    models, choices = {}, []
    for hidden in SIZES:
        fitted = []
        for dropout, decay in REGULARISATION:
            model, value, epoch = fit_one(x, y, x_validation, y_validation, (hidden, dropout, decay), device)
            fitted.append((value, (dropout, decay), epoch, model))
        value, (dropout, decay), epoch, model = max(fitted, key=lambda item: item[0])
        models[hidden] = model
        choices.append({
            "method": method, "source_cohort": source, "fold": fold, "input_columns": int(keep.sum()),
            "hidden": size_name(hidden), "dropout": dropout, "weight_decay": decay,
            "best_epoch": epoch, "validation_auprc": value,
            "parameters": parameter_count(int(keep.sum()), hidden),
        })
    frames = {}
    for cohort in [cohort for cohort in ALL_COHORTS if prediction_cohort(cohort) == source]:
        test = test_frame(method, cohort, fold)
        z = (rows_of(matrix, pd.Index(test[GENE_KEY].astype("object")), what + f" test {cohort}")[:, keep]
             - center) / spread
        test = test.rename(columns={SCORE: "method_score"})
        for hidden, model in models.items():
            with torch.no_grad():
                scores = model(torch.tensor(z, dtype=torch.float32, device=device)).squeeze(1).cpu().numpy()
            test[f"mlp_score_{size_name(hidden)}"] = scores.astype("float64")
        test[FOLD] = fold
        frames[cohort] = test
    return {"frames": frames, "choices": choices}


def evaluate(scores: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (method, cohort), frame in scores.items():
        name, metric = primary_metric(cohort)
        kind = kind_of(cohort)
        y = frame[target_of(cohort)].to_numpy("float64")
        fold = frame[FOLD].to_numpy()
        own = frame["method_score"].to_numpy("float64")
        own_point, own_half_width = interval(y, own, metric, fold, kind)
        for hidden in SIZES:
            mlp = frame[f"mlp_score_{size_name(hidden)}"].to_numpy("float64")
            row = {"method": method, "cohort_id": cohort, "hidden": size_name(hidden), "metric": name,
                   "genes": int(len(frame)), "method_point": own_point, "method_half_width": own_half_width}
            row["mlp_point"], row["mlp_half_width"] = interval(y, mlp, metric, fold, kind)
            row["difference"], row["half_width"], row["p"] = paired(y, mlp, own, metric, fold, kind)
            rows.append(row)
    table = pd.DataFrame(rows)
    table["holm_p"] = holm(table, ["method"], "p")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    gpus = [int(g) for g in arguments.gpus.split(",")]
    jobs = [(method, source, fold, gpus[index % len(gpus)])
            for index, (method, source, fold) in enumerate(product(METHODS, SOURCES, FOLDS))]
    progress = Progress(len(jobs), "fits")
    collected: dict[tuple[str, str], list[pd.DataFrame]] = {}
    choices = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=arguments.workers, mp_context=context) as pool:
        futures = {pool.submit(fit_job, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            method = result["choices"][0]["method"]
            for cohort, frame in result["frames"].items():
                collected.setdefault((method, cohort), []).append(frame)
            choices.extend(result["choices"])
            progress.tick(" ".join(str(part) for part in futures[future][:3]))
    scores = {key: pd.concat(frames, ignore_index=True).sort_values([FOLD, GENE_KEY], ignore_index=True)
              for key, frames in collected.items()}
    size_order = {size_name(hidden): i for i, hidden in enumerate(SIZES)}
    choices = pd.DataFrame(choices).sort_values(
        ["method", "source_cohort", "fold", "hidden"],
        key=lambda s: s.map(size_order) if s.name == "hidden" else s, ignore_index=True)
    write_table(choices, RESULTS / "same_input_mlp_choices.csv")
    table = evaluate(scores)
    order = {cohort: i for i, cohort in enumerate(ALL_COHORTS)}
    table = table.sort_values(
        ["method", "cohort_id", "hidden"],
        key=lambda s: s.map(order) if s.name == "cohort_id" else (s.map(size_order) if s.name == "hidden" else s),
        ignore_index=True)
    write_table(table, RESULTS / "same_input_mlp.csv")
    print("wrote same_input_mlp.csv and same_input_mlp_choices.csv", flush=True)


if __name__ == "__main__":
    main()
