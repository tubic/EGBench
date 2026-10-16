"""Biology of the hallmark exceptions on DEG and DepMap: paralogues, conditional dependency, mouse lethality,
orthologue identity, the COPI and COPII coat subunits and the share of each group the methods find.
Run after hallmark_groups.py: python analysis/hallmark_exceptions/exception_biology.py
"""
from __future__ import annotations


import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, gaussian_kde, mannwhitneyu

from egbench.contracts import SEED
from egbench.data import PROJECT_ROOT, read_cohort_labels
from egbench.modules import symbol_to_key
from egbench.reference import deg_vote_counts
from egbench.stats import holm_adjusted

from common.depmap_matrices import DEPMAP_UNIVERSE
from common.hallmark_counts import RICH, one_per_family, share_row
from common.stats import METHODS
from hallmark_exceptions.hallmark_groups import RESULTS, exception_genes_path

TASKS: tuple[str, ...] = ("deg", "depmap")
TASK_NAMES = {"deg": "DEG", "depmap": "DepMap"}
GROUPS: tuple[str, ...] = ("poor_essential", "rich_essential", "poor_nonessential", "rich_nonessential")
EDGES = PROJECT_ROOT / "datasets" / "homology" / "filtered_edges.parquet"
COMPARA = PROJECT_ROOT / "data" / "source" / "annotations" / "Compara.115.protein_default.homologies.tsv.gz"
SPECIES = {"gallus_gallus": "chicken", "danio_rerio": "zebrafish", "rattus_norvegicus": "rat"}
DEPENDENT_LINES = 0.01
CURVE_GRID = np.logspace(-3, 0, 61)
EFFECT_GRID = np.linspace(-3.5, 0.5, 201)
IDENTITY_GRID = np.linspace(0.0, 100.0, 201)
COAT = {
    "COPI": (["COPA", "COPB1", "COPB2", "ARCN1", "COPE", "COPG1", "COPZ1"], {"COPG1": "COPG2", "COPZ1": "COPZ2"}),
    "COPII": (["SEC13", "SEC24A", "SEC24B", "SEC31A", "SEC23A", "SEC24C", "SAR1A"],
              {"SEC31A": "SEC31B", "SEC23A": "SEC23B", "SEC24C": "SEC24D", "SAR1A": "SAR1B"}),
}
ALL, FAMILY = "all_genes", "one_gene_per_family"


def fisher(a: pd.Series, b: pd.Series) -> float:
    return float(fisher_exact([[int(a.sum()), int(len(a) - a.sum())], [int(b.sum()), int(len(b) - b.sum())]])[1])


def test_row(task: str, scope: str, name: str, a: pd.Series, b: pd.Series, a_name: str, b_name: str) -> dict:
    return {"task": task, "scope": scope, "test": name, "a": a_name, "a_value": float(a.mean()), "a_n": int(len(a)),
            "b": b_name, "b_value": float(b.mean()), "b_n": int(len(b)), "statistic": "share", "p": fisher(a, b)}


def rank_row(task: str, scope: str, name: str, a: pd.Series, b: pd.Series, a_name: str, b_name: str) -> dict:
    a, b = a.dropna(), b.dropna()
    return {"task": task, "scope": scope, "test": name, "a": a_name, "a_value": float(a.median()), "a_n": int(len(a)),
            "b": b_name, "b_value": float(b.median()), "b_n": int(len(b)), "statistic": "median",
            "p": float(mannwhitneyu(a, b, alternative="two-sided").pvalue)}


def paralogue_keys() -> set[str]:
    edges = pd.read_parquet(EDGES, columns=["gene_key_a", "gene_key_b"])
    return set(edges["gene_key_a"].astype(str)) | set(edges["gene_key_b"].astype(str))


def orthologue_identity() -> pd.DataFrame:
    pieces = []
    for chunk in pd.read_csv(COMPARA, sep="\t", na_values="NULL", chunksize=500_000,
                             usecols=["gene_stable_id", "identity", "homology_type", "homology_species"]):
        kept = chunk[chunk["homology_species"].isin(SPECIES) & (chunk["homology_type"] == "ortholog_one2one")]
        pieces.append(kept[["gene_stable_id", "homology_species", "identity"]])
    table = pd.concat(pieces, ignore_index=True)
    universe = pd.read_parquet(DEPMAP_UNIVERSE, columns=["gene_key", "ensembl_gene_id"]).astype(str)
    table = table.merge(universe, left_on="gene_stable_id", right_on="ensembl_gene_id")
    wide = table.pivot_table(index="gene_key", columns="homology_species", values="identity", aggfunc="max")
    return wide.rename(columns={species: f"identity_{name}" for species, name in SPECIES.items()})


def gene_table(task: str, paralogues: set[str], identity: pd.DataFrame, mouse: pd.Series) -> pd.DataFrame:
    frame = pd.read_csv(exception_genes_path(task))
    frame["gene_key"] = frame["gene_key"].astype(str)
    frame["paralogue"] = frame["gene_key"].isin(paralogues)
    frame["mouse_lethal"] = mouse.reindex(frame["gene_key"]).to_numpy("float64")
    for column in identity.columns:
        frame[column] = identity[column].reindex(frame["gene_key"]).to_numpy()
    kept = ["gene_key", "gene_symbol", "label", "group", "count", "complex", "old", "connected", "expressed",
            "cluster_id", "paralogue", "mouse_lethal", "gene_effect_mean", "dependent_share", *identity.columns]
    return frame[kept]


def paralogue_tables(task: str, scopes: dict[str, pd.DataFrame]) -> tuple[list[dict], list[dict]]:
    shares, tests = [], []
    for scope, frame in scopes.items():
        for k in range(5):
            part = frame[frame["count"] == k]
            for flag in (False, True):
                count, size, share, half = share_row(part[part.paralogue == flag].label == 1)
                shares.append({"task": task, "scope": scope, "count": k, "paralogue": int(flag), "genes": size,
                               "essential": count, "essential_share": share, "half_width": half})
            tests.append(test_row(task, scope, f"essential_share_count_{k}", part[~part.paralogue].label == 1,
                                  part[part.paralogue].label == 1, "no_paralogue", "paralogue"))
        rich = frame[frame["count"] >= RICH]
        tests.append(test_row(task, scope, "paralogue_share_rich", rich[rich.label == 0].paralogue,
                              rich[rich.label == 1].paralogue, "rich_nonessential", "rich_essential"))
        tests.append(test_row(task, scope, "essential_share_rich", rich[~rich.paralogue].label == 1,
                              rich[rich.paralogue].label == 1, "no_paralogue", "paralogue"))
    return shares, tests


def exception_tests(task: str, scopes: dict[str, pd.DataFrame]) -> list[dict]:
    tests = []
    for scope, frame in scopes.items():
        group = {name: frame[frame.group == name] for name in GROUPS}
        measured = {name: part.dropna(subset=["dependent_share"]) for name, part in group.items()}
        tests.append(test_row(task, scope, "dependent_in_1pct_lines",
                              measured["rich_nonessential"].dependent_share >= DEPENDENT_LINES,
                              measured["poor_nonessential"].dependent_share >= DEPENDENT_LINES,
                              "rich_nonessential", "poor_nonessential"))
        tested = {name: part.dropna(subset=["mouse_lethal"]) for name, part in group.items()}
        tests.append(test_row(task, scope, "mouse_lethal", tested["rich_nonessential"].mouse_lethal == 1,
                              tested["poor_nonessential"].mouse_lethal == 1, "rich_nonessential", "poor_nonessential"))
        poor, rich = group["poor_essential"], group["rich_essential"]
        for column in ("gene_effect_mean", "dependent_share", *[f"identity_{name}" for name in SPECIES.values()]):
            tests.append(rank_row(task, scope, column, poor[column], rich[column], "poor_essential", "rich_essential"))
        if scope == ALL:
            tests.append(rank_row(task, scope, "identity_chicken_against_poor_nonessential", poor.identity_chicken,
                                  group["poor_nonessential"].identity_chicken, "poor_essential", "poor_nonessential"))
            for column in ("old", "expressed"):
                for value in (0, 1):
                    tests.append(rank_row(task, scope, f"identity_chicken_{column}_{value}",
                                          poor[poor[column] == value].identity_chicken,
                                          rich[rich[column] == value].identity_chicken,
                                          "poor_essential", "rich_essential"))
    return tests


def group_summary(task: str, frame: pd.DataFrame) -> list[dict]:
    rows = []
    for name in GROUPS:
        part = frame[frame.group == name]
        measured, tested = part.dropna(subset=["dependent_share"]), part.dropna(subset=["mouse_lethal"])
        row = {"task": task, "group": name, "genes": int(len(part)), "paralogue_share": float(part.paralogue.mean()),
               "dependent_known": int(len(measured)),
               "dependent_in_1pct_lines": float((measured.dependent_share >= DEPENDENT_LINES).mean()),
               "dependent_share_median": float(measured.dependent_share.median()),
               "mouse_tested": int(len(tested)), "mouse_lethal_share": float((tested.mouse_lethal == 1).mean()),
               "gene_effect_median": float(part.gene_effect_mean.median())}
        for species in SPECIES.values():
            values = part[f"identity_{species}"].dropna()
            row[f"identity_{species}_genes"] = int(len(values))
            for label, q in (("q25", 0.25), ("median", 0.5), ("q75", 0.75)):
                row[f"identity_{species}_{label}"] = float(values.quantile(q))
        rows.append(row)
    return rows


def curves(task: str, frame: pd.DataFrame) -> tuple[list[dict], list[dict], list[dict]]:
    dependency, effect, identity = [], [], []
    for name in ("rich_nonessential", "poor_nonessential"):
        values = frame[frame.group == name].dependent_share.dropna().to_numpy()
        dependency += [{"task": task, "group": name, "dependent_share_at_least": float(x), "gene_share": float((values >= x).mean())}
                       for x in CURVE_GRID]
    for name in ("poor_essential", "rich_essential"):
        values = frame[frame.group == name].gene_effect_mean.dropna().to_numpy()
        effect += [{"task": task, "group": name, "gene_effect_mean": float(x), "density": float(d)}
                   for x, d in zip(EFFECT_GRID, gaussian_kde(values)(EFFECT_GRID))]
    for name in GROUPS:
        values = frame[frame.group == name].identity_chicken.dropna().to_numpy()
        identity += [{"task": task, "group": name, "identity": float(x), "density": float(d)}
                     for x, d in zip(IDENTITY_GRID, gaussian_kde(values)(IDENTITY_GRID))]
    return dependency, effect, identity


def coat_table(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = symbol_to_key()
    votes = deg_vote_counts()
    votes.index = votes.index.astype(str)
    by_task = {task: frame.set_index("gene_key") for task, frame in frames.items()}
    rows = []
    for coat, (subunits, partners) in COAT.items():
        for position, name in enumerate(subunits):
            members = [(name, "subunit", "")]
            if name in partners:
                members.append((partners[name], "paralogue", name))
            for symbol, role, partner in members:
                key = keys[symbol]
                row = {"coat": coat, "position": position, "gene_symbol": symbol, "gene_key": key, "role": role,
                       "partner": partner, "deg_votes": float(votes.get(key, np.nan))}
                for task, table in by_task.items():
                    known = key in table.index
                    row[f"label_{task}"] = float(table.at[key, "label"]) if known else np.nan
                    row[f"count_{task}"] = float(table.at[key, "count"]) if known else np.nan
                rows.append(row)
    table = pd.DataFrame(rows)
    edges = pd.read_parquet(EDGES).astype({"gene_key_a": str, "gene_key_b": str})
    inside = set(table["gene_key"])
    edges = edges[edges.gene_key_a.isin(inside) & edges.gene_key_b.isin(inside)]
    symbol = dict(zip(table["gene_key"], table["gene_symbol"]))
    edges = pd.DataFrame({"gene_a": edges.gene_key_a.map(symbol), "gene_b": edges.gene_key_b.map(symbol),
                          "identity": edges["fident"].to_numpy()})
    return table, edges


def recall_tests(task: str, frame: pd.DataFrame) -> pd.DataFrame:
    methods = pd.read_csv(RESULTS / "exception_methods.csv")
    values = methods[methods.cohort_id == task].pivot(index="predictor", columns="quantity", values="point")
    sizes = {name: int((frame.group == name).sum()) for name in ("poor_essential", "rich_essential")}
    rows = []
    for method in METHODS:
        found = {name: int(round(float(values.at[method, f"recall_{name}"]) * sizes[name])) for name in sizes}
        p = fisher_exact([[found["rich_essential"], sizes["rich_essential"] - found["rich_essential"]],
                          [found["poor_essential"], sizes["poor_essential"] - found["poor_essential"]]])[1]
        rows.append({"task": task, "method": method,
                     "rich_essential_found": found["rich_essential"], "rich_essential": sizes["rich_essential"],
                     "poor_essential_found": found["poor_essential"], "poor_essential": sizes["poor_essential"],
                     "rich_share": found["rich_essential"] / sizes["rich_essential"],
                     "poor_share": found["poor_essential"] / sizes["poor_essential"], "p": float(p)})
    table = pd.DataFrame(rows)
    table["holm_p"] = holm_adjusted(table["p"].to_numpy("float64"))
    return table


def write_table(frame: pd.DataFrame, name: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / f"{name}.csv", index=False)


def main() -> None:
    generator = np.random.default_rng(SEED)
    paralogues = paralogue_keys()
    identity = orthologue_identity()
    mouse = read_cohort_labels("mouse").astype({"gene_key": str}).set_index("gene_key")["label"].astype("float64")

    shares, tests, summary, dependency, effect, density, recalls, frames = [], [], [], [], [], [], [], {}
    for task in TASKS:
        frame = gene_table(task, paralogues, identity, mouse)
        frames[task] = frame
        scopes = {ALL: frame, FAMILY: one_per_family(frame, generator)}
        part_shares, part_tests = paralogue_tables(task, scopes)
        shares += part_shares
        tests += part_tests + exception_tests(task, scopes)
        summary += group_summary(task, frame)
        curve = curves(task, frame)
        dependency += curve[0]
        effect += curve[1]
        density += curve[2]
        recalls.append(recall_tests(task, frame))
        print(f"{TASK_NAMES[task]}: {len(frame):,} genes, {len(scopes[FAMILY]):,} with one gene per family", flush=True)

    write_table(pd.DataFrame(shares), "paralogue_essential_share")
    write_table(pd.DataFrame(summary), "exception_group_summary")
    write_table(pd.DataFrame(dependency), "dependency_curve")
    write_table(pd.DataFrame(effect), "gene_effect_density")
    write_table(pd.DataFrame(density), "identity_density")
    write_table(pd.concat(recalls, ignore_index=True), "recall_tests")
    coat, coat_edges = coat_table(frames)
    write_table(coat, "coat_subunits")
    write_table(coat_edges, "coat_edges")
    table = pd.DataFrame(tests)
    table["holm_p"] = np.nan
    for _, rows in table.groupby(["task", "scope"], sort=False):
        table.loc[rows.index, "holm_p"] = holm_adjusted(rows["p"].to_numpy("float64"))
    write_table(table, "biology_tests")


if __name__ == "__main__":
    main()
