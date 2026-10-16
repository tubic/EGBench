"""Biology of the genes DepMap 26Q1 newly calls common essential: paralogues by hallmark count, GO processes,
five CORUM complexes and the running enrichment curves of the predictors.
Run after prospective_core.py: python analysis/prospective_validation/prospective_biology.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom

from egbench.contracts import SEED
from egbench.data import PROJECT_ROOT, read_cohort_labels
from egbench.modules import complex_names, complex_pairs, process_pairs, restrict_terms
from egbench.stats import benjamini_hochberg_adjusted, holm_adjusted

from common.depmap_matrices import DEPENDENT_ABOVE, DEPMAP_UNIVERSE, dependency_matrix
from common.feature_blocks import go_names
from common.hallmark_counts import RICH, one_per_family, share_row, skip_draws
from common.stats import FOLDS, percentile
from prospective_validation.prospective_core import CORE, PREDICTORS, RESULTS, assembled_paths

EDGES = PROJECT_ROOT / "datasets" / "homology" / "filtered_edges.parquet"
CLUSTERS = PROJECT_ROOT / "datasets" / "homology" / "cluster_map.parquet"
EXCEPTION_TASKS: tuple[str, ...] = ("deg", "depmap")
COMPLEXES = {"CORUM:306": "Cytosolic ribosome", "CORUM:181": "26S proteasome", "CORUM:320": "Mitochondrial ribosome",
             "CORUM:178": "Respiratory complex I", "CORUM:7305": "Cytochrome c oxidase"}
TERM_MINIMUM, TERM_MAXIMUM = 5, 200
CURVE_POINTS = 400
ALL, FAMILY = "all_genes", "one_gene_per_family"
TASK = "depmap_23q4"


def fisher(a: pd.Series, b: pd.Series) -> float:
    return float(fisher_exact([[int(a.sum()), int(len(a) - a.sum())], [int(b.sum()), int(len(b) - b.sum())]])[1])


def test_row(scope: str, name: str, a: pd.Series, b: pd.Series, a_name: str, b_name: str) -> dict:
    return {"task": TASK, "scope": scope, "test": name, "a": a_name, "a_value": float(a.mean()), "a_n": int(len(a)),
            "b": b_name, "b_value": float(b.mean()), "b_n": int(len(b)), "statistic": "share", "p": fisher(a, b)}


def paralogue_keys() -> set[str]:
    edges = pd.read_parquet(EDGES, columns=["gene_key_a", "gene_key_b"])
    return set(edges["gene_key_a"].astype(str)) | set(edges["gene_key_b"].astype(str))


def prospective_table(paralogues: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "hallmarks_depmap_23q4.csv")
    frame["gene_key"] = frame["gene_key"].astype(str)
    frame["paralogue"] = frame["gene_key"].isin(paralogues)
    clusters = pd.read_parquet(CLUSTERS, columns=["gene_key", "cluster_id"]).astype(str).set_index("gene_key")["cluster_id"]
    frame["cluster_id"] = clusters.reindex(frame["gene_key"]).to_numpy()
    frame["cluster_id"] = frame["cluster_id"].fillna(frame["gene_key"])
    fate = frame["prospective_class"].map({"foreground": "newly_essential", "background": "unchanged"})
    fate = fate.where(frame["label_23q4"] == 0, "essential_23q4").fillna("strongly_selective")
    frame["fate"] = fate
    return frame


def prospective_paralogue(scopes: dict[str, pd.DataFrame]) -> tuple[list[dict], list[dict]]:
    shares, tests = [], []
    for scope, frame in scopes.items():
        negatives = frame[frame.label_23q4 == 0].assign(new=lambda d: d.fate == "newly_essential")
        for k in range(5):
            part = negatives[negatives["count"] == k]
            for flag in (False, True):
                count, size, share, half = share_row(part[part.paralogue == flag].new)
                shares.append({"scope": scope, "count": k, "paralogue": int(flag), "genes": size,
                               "newly_essential": count, "share": share, "half_width": half})
            tests.append(test_row(scope, f"became_essential_count_{k}", part[~part.paralogue].new,
                                  part[part.paralogue].new, "no_paralogue", "paralogue"))
        rich = negatives[negatives["count"] >= RICH]
        tests.append(test_row(scope, "became_essential_rich", rich[~rich.paralogue].new,
                              rich[rich.paralogue].new, "no_paralogue", "paralogue"))
        scored = frame[frame.fate.isin(("newly_essential", "unchanged"))]
        tests.append(test_row(scope, "no_paralogue_share", ~scored[scored.fate == "newly_essential"].paralogue,
                              ~scored[scored.fate == "unchanged"].paralogue, "newly_essential", "unchanged"))
    return shares, tests


def prospective_enrichment(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame[frame.fate.isin(("newly_essential", "unchanged"))]
    inside = pd.Series((scored.fate == "newly_essential").to_numpy(), index=scored["gene_key"].to_numpy())
    terms = restrict_terms(process_pairs(), inside.index, minimum=TERM_MINIMUM, maximum=TERM_MAXIMUM)
    terms = terms.astype({"gene_key": str})
    names = go_names()
    total, drawn = int(len(inside)), int(inside.sum())
    rows = []
    for term, members in terms.groupby("term_id"):
        present = inside.reindex(members["gene_key"]).dropna().astype(bool)
        size, hits = int(len(present)), int(present.sum())
        rows.append({"term_id": term, "term_name": names.get(term, ""), "genes": size, "newly_essential": hits,
                     "fold_enrichment": (hits / size) / (drawn / total),
                     "p": float(hypergeom.sf(hits - 1, total, size, drawn))})
    table = pd.DataFrame(rows)
    table["q"] = benjamini_hochberg_adjusted(table["p"].to_numpy("float64"))
    table = table.sort_values(["q", "p", "term_id"], kind="stable").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table["background_total"], table["foreground_total"] = total, drawn
    return table


def prospective_complexes(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = dependency_matrix()
    matrix.columns = matrix.columns.astype(str)
    dependent = (matrix > DEPENDENT_ABOVE).sum(axis=0) / matrix.notna().sum(axis=0)
    fate = frame.set_index("gene_key")["fate"]
    symbols = pd.read_parquet(DEPMAP_UNIVERSE, columns=["gene_key", "gene_symbol"]).astype(str).set_index("gene_key")["gene_symbol"]
    pairs = complex_pairs().astype({"gene_key": str})
    names = complex_names()
    scored = frame[frame.fate.isin(("newly_essential", "unchanged"))]
    total, drawn = int(len(scored)), int((scored.fate == "newly_essential").sum())
    members, summary = [], []
    for term, label in COMPLEXES.items():
        keys = pairs.loc[pairs.term_id == term, "gene_key"].drop_duplicates()
        fates = fate.reindex(keys).fillna("not_in_cohort")
        for key, value in zip(keys, fates):
            members.append({"complex_id": term, "complex": label, "gene_key": key, "gene_symbol": symbols.get(key, ""),
                            "fate": value, "dependent_share_26q1": float(dependent.get(key, np.nan))})
        counts = fates.value_counts()
        new, unchanged = int(counts.get("newly_essential", 0)), int(counts.get("unchanged", 0))
        summary.append({"complex_id": term, "complex": label, "corum_name": names.get(term, ""), "subunits": int(len(keys)),
                        "essential_23q4": int(counts.get("essential_23q4", 0)), "newly_essential": new,
                        "strongly_selective": int(counts.get("strongly_selective", 0)), "unchanged": unchanged,
                        "not_in_cohort": int(counts.get("not_in_cohort", 0)),
                        "p": float(hypergeom.sf(new - 1, total, new + unchanged, drawn)) if new + unchanged else np.nan})
    return pd.DataFrame(members), pd.DataFrame(summary)


def running_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    table_path, block_path = assembled_paths()
    base = pd.read_csv(table_path)
    block = np.load(block_path)
    hits = base["label"].to_numpy("float64") == 1
    fold = base["fold"].to_numpy()
    positions = np.linspace(0.0, 1.0, CURVE_POINTS + 1)
    curves_, barcode = [], []
    for column, predictor in enumerate(PREDICTORS):
        within = np.empty(len(base))
        for f in FOLDS:
            inside = fold == f
            within[inside] = percentile(block[inside, column])
        values, inverse = np.unique(-within, return_inverse=True)
        size = np.bincount(inverse, minlength=len(values)).astype("float64")
        found = np.bincount(inverse, weights=hits.astype("float64"), minlength=len(values))
        rank = np.concatenate([[0.0], np.cumsum(size)]) / len(base)
        height = np.concatenate([[0.0], np.cumsum(found / hits.sum() - (size - found) / (~hits).sum())])
        for x, y in zip(positions, np.interp(positions, rank, height)):
            curves_.append({"predictor": predictor, "rank_share": float(x), "running_enrichment": float(y)})
        if predictor == CORE:
            middle = (rank[:-1] + rank[1:]) / 2.0
            barcode = [{"gene_key": key, "rank_share": float(middle[group])}
                       for key, group in zip(base.loc[hits, "gene_key"].astype(str), inverse[hits])]
    return pd.DataFrame(curves_), pd.DataFrame(barcode).sort_values("rank_share").reset_index(drop=True)


def write_table(frame: pd.DataFrame, name: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / f"{name}.csv", index=False)


def main() -> None:
    generator = np.random.default_rng(SEED)
    skip_draws(generator, [len(read_cohort_labels(task)) for task in EXCEPTION_TASKS])
    prospective = prospective_table(paralogue_keys())
    scopes = {ALL: prospective, FAMILY: one_per_family(prospective, generator)}
    shares, tests = prospective_paralogue(scopes)
    write_table(pd.DataFrame(shares), "prospective_paralogue_share")
    table = pd.DataFrame(tests)
    table["holm_p"] = np.nan
    for _, rows in table.groupby(["task", "scope"], sort=False):
        table.loc[rows.index, "holm_p"] = holm_adjusted(rows["p"].to_numpy("float64"))
    write_table(table, "biology_tests")
    enrichment = prospective_enrichment(prospective)
    write_table(enrichment, "prospective_go")
    members, complexes = prospective_complexes(prospective)
    write_table(members, "prospective_complex_subunits")
    write_table(complexes, "prospective_complexes")
    running, barcode = running_curves()
    write_table(running, "prospective_running")
    write_table(barcode, "prospective_barcode")


if __name__ == "__main__":
    main()
