"""Dependency strength and breadth: complexes by gene effect, hallmarks and paralogues along the share of dependent
cell lines, GO processes and CORUM complexes of pan-essential and context-specific genes, lineage genes.
Run after prospective_validation/prospective_core.py: python analysis/dependency_strength_context/context_biology.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom, mannwhitneyu, spearmanr

from egbench.contracts import SEED
from egbench.data import PROJECT_ROOT, primary_scope, scope_gene_keys
from egbench.modules import complex_names, complex_pairs, process_pairs, restrict_terms
from egbench.reference import GENE_EFFECT_COLUMN, gene_effect_means
from egbench.stats import benjamini_hochberg_adjusted

from common.depmap_matrices import DEPENDENT_ABOVE, DEPMAP_UNIVERSE, MODELS, dependency_matrix
from common.feature_blocks import go_names
from common.hallmark_counts import RICH, hallmark_table, one_per_family, share_row

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WORK = HERE / "work"
PROSPECTIVE_HALLMARKS = PROJECT_ROOT / "analysis" / "prospective_validation" / "results" / "hallmarks_depmap_23q4.csv"
EDGES = PROJECT_ROOT / "datasets" / "homology" / "filtered_edges.parquet"
CLUSTERS = PROJECT_ROOT / "datasets" / "homology" / "cluster_map.parquet"
TERM_MINIMUM, TERM_MAXIMUM = 5, 200
ALL, FAMILY = "all_genes", "one_gene_per_family"
STRENGTH_COMPLEXES = {"CORUM:306": "Cytosolic ribosome", "CORUM:8371": "Spliceosome B complex",
                      "CORUM:181": "26S proteasome", "CORUM:2685": "RNA polymerase II", "CORUM:1097": "eIF3",
                      "CORUM:230": "Mediator", "CORUM:8674": "Nuclear pore complex",
                      "CORUM:320": "Mitochondrial ribosome", "CORUM:178": "Respiratory complex I"}
COMPLEX_MINIMUM = 10
BREADTH_EDGES = (0.0, 0.01, 0.10, 0.50, 0.90, 1.0)
BREADTH_NAMES = ("0", "0-1%", "1-10%", "10-50%", "50-90%", ">90%")
CONTEXT_COMPLEX_MINIMUM = 5
COMPOSITION_COMPLEXES = {"CORUM:306": "Cytosolic ribosome", "CORUM:181": "26S proteasome",
                         "CORUM:178": "Respiratory complex I", "CORUM:8561": "TNFR1 signalling complex",
                         "CORUM:1624": "Fanconi anaemia core complex", "CORUM:1492": "LSD1-CoREST (BHC110) complex"}
LINEAGE_MINIMUM = 30
LINEAGE_SHARE = 0.5
HEATMAP_GENES = 20


def strength_genes_path() -> Path:
    return WORK / "strength_genes_depmap.csv"


def context_genes_path() -> Path:
    return WORK / "context_genes.csv"


def fisher(a: pd.Series, b: pd.Series) -> float:
    return float(fisher_exact([[int(a.sum()), int(len(a) - a.sum())], [int(b.sum()), int(len(b) - b.sum())]])[1])


def test_row(section: str, scope: str, name: str, a: pd.Series, b: pd.Series, a_name: str, b_name: str) -> dict:
    return {"section": section, "scope": scope, "test": name, "a": a_name, "a_value": float(a.mean()), "a_n": int(len(a)),
            "b": b_name, "b_value": float(b.mean()), "b_n": int(len(b)), "statistic": "share", "p": fisher(a, b)}


def rank_row(section: str, scope: str, name: str, a: pd.Series, b: pd.Series, a_name: str, b_name: str) -> dict:
    a, b = a.dropna(), b.dropna()
    return {"section": section, "scope": scope, "test": name, "a": a_name, "a_value": float(a.median()), "a_n": int(len(a)),
            "b": b_name, "b_value": float(b.median()), "b_n": int(len(b)), "statistic": "median",
            "p": float(mannwhitneyu(a, b, alternative="two-sided").pvalue)}


def trend_row(section: str, scope: str, name: str, x: pd.Series, y: pd.Series, x_name: str, y_name: str) -> dict:
    result = spearmanr(x.to_numpy("float64"), y.to_numpy("float64"))
    return {"section": section, "scope": scope, "test": name, "a": x_name, "a_value": float(result.statistic),
            "a_n": int(len(x)), "b": y_name, "b_value": np.nan, "b_n": int(len(y)), "statistic": "spearman",
            "p": float(result.pvalue)}


def with_clusters(frame: pd.DataFrame) -> pd.DataFrame:
    clusters = pd.read_parquet(CLUSTERS, columns=["gene_key", "cluster_id"]).astype(str).set_index("gene_key")["cluster_id"]
    frame = frame.copy()
    frame["cluster_id"] = clusters.reindex(frame["gene_key"]).to_numpy()
    frame["cluster_id"] = frame["cluster_id"].fillna(frame["gene_key"])
    return frame


def scopes_of(frame: pd.DataFrame, generator: np.random.Generator) -> dict[str, pd.DataFrame]:
    return {ALL: frame, FAMILY: one_per_family(frame, generator)}


def paralogue_keys() -> set[str]:
    edges = pd.read_parquet(EDGES, columns=["gene_key_a", "gene_key_b"])
    return set(edges["gene_key_a"].astype(str)) | set(edges["gene_key_b"].astype(str))


def enrichment(inside: pd.Series, group: str, names: dict[str, str], symbols: pd.Series) -> pd.DataFrame:
    terms = restrict_terms(process_pairs(), inside.index, minimum=TERM_MINIMUM, maximum=TERM_MAXIMUM)
    terms = terms.astype({"gene_key": str})
    total, drawn = int(len(inside)), int(inside.sum())
    rows = []
    for term, members in terms.groupby("term_id"):
        present = inside.reindex(members["gene_key"]).dropna().astype(bool)
        size, hits = int(len(present)), int(present.sum())
        found = sorted(symbols.reindex(present.index[present.to_numpy()]).dropna().astype(str))
        rows.append({"group": group, "term_id": term, "term_name": names.get(term, ""), "background_genes": size,
                     "foreground_genes": hits, "fold_enrichment": (hits / size) / (drawn / total),
                     "p": float(hypergeom.sf(hits - 1, total, size, drawn)), "foreground_symbols": ";".join(found)})
    table = pd.DataFrame(rows)
    table["q"] = benjamini_hochberg_adjusted(table["p"].to_numpy("float64"))
    table = table.sort_values(["q", "p", "term_id"], kind="stable").reset_index(drop=True)
    table.insert(1, "rank", np.arange(1, len(table) + 1))
    table["background_total"], table["foreground_total"] = total, drawn
    return table


def depmap_table(paralogues: set[str], effect: pd.Series, dependent: pd.Series, symbols: pd.Series) -> pd.DataFrame:
    frame, _ = hallmark_table("depmap")
    frame["gene_key"] = frame["gene_key"].astype(str)
    frame["paralogue"] = frame["gene_key"].isin(paralogues)
    frame["gene_effect_mean"] = effect.reindex(frame["gene_key"]).to_numpy()
    frame["dependent_share"] = dependent.reindex(frame["gene_key"]).to_numpy()
    frame["gene_symbol"] = symbols.reindex(frame["gene_key"]).to_numpy()
    return with_clusters(frame)


def context_table(paralogues: set[str], dependent: pd.Series, symbols: pd.Series) -> pd.DataFrame:
    frame, _ = hallmark_table("context")
    frame["gene_key"] = frame["gene_key"].astype(str)
    kept = set(map(str, scope_gene_keys("context", primary_scope("context"))))
    frame = frame[frame["gene_key"].isin(kept)].reset_index(drop=True)
    frame["paralogue"] = frame["gene_key"].isin(paralogues)
    frame["dependent_share"] = dependent.reindex(frame["gene_key"]).to_numpy()
    frame["gene_symbol"] = symbols.reindex(frame["gene_key"]).to_numpy()
    return with_clusters(frame)


def breadth_frame(paralogues: set[str], dependent: pd.Series) -> pd.DataFrame:
    frame = pd.read_csv(PROSPECTIVE_HALLMARKS).astype({"gene_key": str})
    frame["dependent_share"] = dependent.reindex(frame["gene_key"]).to_numpy()
    frame = with_clusters(frame.dropna(subset=["dependent_share"]).reset_index(drop=True))
    frame["paralogue"] = frame["gene_key"].isin(paralogues)
    frame["rich"] = frame["count"] >= RICH
    band = np.searchsorted(np.asarray(BREADTH_EDGES), frame["dependent_share"].to_numpy(), side="left")
    frame["band"] = np.asarray(BREADTH_NAMES, dtype="object")[band]
    return frame


def strength_tests(scopes: dict[str, pd.DataFrame]) -> list[dict]:
    tests = []
    for scope, frame in scopes.items():
        essential = frame[frame.label == 1].dropna(subset=["gene_effect_mean"])
        tests.append(trend_row("A", scope, "hallmark_count_against_strength", essential["count"],
                               -essential["gene_effect_mean"], "hallmark_count", "minus_gene_effect"))
        tests.append(rank_row("A", scope, "gene_effect_by_yeast_orthologue", essential.loc[essential.old == 1, "gene_effect_mean"],
                              essential.loc[essential.old == 0, "gene_effect_mean"], "yeast_orthologue", "no_yeast_orthologue"))
        tests.append(rank_row("A", scope, "gene_effect_four_against_fewer", essential.loc[essential["count"] == 4, "gene_effect_mean"],
                              essential.loc[essential["count"] < 4, "gene_effect_mean"], "four_hallmarks", "fewer"))
    return tests


def strength_complexes(effect: pd.Series, universe: pd.DataFrame, depmap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = complex_pairs().astype({"gene_key": str})
    names = complex_names()
    label = depmap.set_index("gene_key")["label"]
    count = depmap.set_index("gene_key")["count"]
    summary, members = [], []
    for term, rows in pairs.groupby("term_id"):
        keys = rows["gene_key"].drop_duplicates()
        values = effect.reindex(keys).dropna()
        if len(values) < COMPLEX_MINIMUM:
            continue
        labels = label.reindex(values.index)
        summary.append({"complex_id": term, "corum_name": names.get(term, ""), "complex": STRENGTH_COMPLEXES.get(term, ""),
                        "subunits": int(len(keys)), "measured": int(len(values)), "essential": int((labels == 1).sum()),
                        "q25": float(values.quantile(0.25)), "median": float(values.median()),
                        "q75": float(values.quantile(0.75)),
                        "rich_share": float((count.reindex(values.index).dropna() >= RICH).mean())})
        if term in STRENGTH_COMPLEXES:
            for key, value in values.items():
                state = ("common_essential" if universe.at[key, "official_common_essential"] == 1
                         else "strongly_selective" if universe.at[key, "strongly_selective"] == 1 else "neither")
                members.append({"complex_id": term, "complex": STRENGTH_COMPLEXES[term], "gene_key": key,
                                "gene_symbol": universe.at[key, "gene_symbol"], "gene_effect_mean": float(value),
                                "depmap_class": state})
    table = pd.DataFrame(summary).sort_values("median", kind="stable").reset_index(drop=True)
    return table, pd.DataFrame(members)


def breadth_tables(frame: pd.DataFrame, generator: np.random.Generator) -> tuple[pd.DataFrame, list[dict]]:
    rows, tests = [], []
    for scope, part in scopes_of(frame, generator).items():
        for name in BREADTH_NAMES:
            inside = part[part.band == name]
            row = {"scope": scope, "band": name, "genes": int(len(inside))}
            for column in ("paralogue", "rich", "old", "complex", "connected", "expressed"):
                _, _, share, half = share_row(inside[column].astype(bool))
                row[f"{column}_share"], row[f"{column}_half_width"] = share, half
            rows.append(row)
        tests.append(trend_row("B", scope, "dependent_share_against_paralogue", part["dependent_share"],
                               part["paralogue"].astype(float), "dependent_share", "paralogue"))
        tests.append(trend_row("B", scope, "dependent_share_against_hallmark_count", part["dependent_share"],
                               part["count"], "dependent_share", "hallmark_count"))
        broad, narrow = part[part.band == BREADTH_NAMES[-1]], part[part.band == BREADTH_NAMES[2]]
        tests.append(test_row("B", scope, "paralogue_share_1_10_against_over_90", narrow["paralogue"], broad["paralogue"],
                              "dependent_1_10pct", "dependent_over_90pct"))
    return pd.DataFrame(rows), tests


def class_tests(scopes_depmap: dict[str, pd.DataFrame], scopes_context: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[dict]]:
    rows, tests = [], []
    for scope in (ALL, FAMILY):
        depmap, context = scopes_depmap[scope], scopes_context[scope]
        groups = {"pan_essential": depmap[depmap.label == 1], "context_specific": context[context.label == 1],
                  "context_negative": context[context.label == 0]}
        for name, part in groups.items():
            row = {"scope": scope, "group": name, "genes": int(len(part)),
                   "dependent_share_median": float(part["dependent_share"].median())}
            for column, mask in (("paralogue", part["paralogue"]), ("rich", part["count"] >= RICH), ("old", part["old"] == 1),
                                 ("complex", part["complex"] == 1)):
                _, _, share, half = share_row(mask.astype(bool))
                row[f"{column}_share"], row[f"{column}_half_width"] = share, half
            for layer in range(1, 10):
                row[f"age_layer_{layer}"] = float((part["age_layer"] == layer).mean())
            rows.append(row)
        tests.append(test_row("B", scope, "paralogue_share_context_against_pan", groups["context_specific"]["paralogue"],
                              groups["pan_essential"]["paralogue"], "context_specific", "pan_essential"))
        tests.append(test_row("B", scope, "rich_share_context_against_pan", groups["context_specific"]["count"] >= RICH,
                              groups["pan_essential"]["count"] >= RICH, "context_specific", "pan_essential"))
        tests.append(test_row("B", scope, "yeast_share_context_against_pan", groups["context_specific"]["old"] == 1,
                              groups["pan_essential"]["old"] == 1, "context_specific", "pan_essential"))
    return pd.DataFrame(rows), tests


def context_complexes(depmap: pd.DataFrame, context: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = complex_pairs().astype({"gene_key": str})
    names = complex_names()
    label = context.set_index("gene_key")["label"]
    symbol = context.set_index("gene_key")["gene_symbol"]
    total, drawn = int(len(label)), int(label.sum())
    state = pd.Series("non_essential", index=pd.Index(pd.concat([depmap["gene_key"], context["gene_key"]]).unique()))
    state.loc[depmap.loc[depmap.label == 1, "gene_key"].to_numpy()] = "pan_essential"
    state.loc[context.loc[context.label == 1, "gene_key"].to_numpy()] = "context_specific"
    rows, composition = [], []
    for term, members in pairs.groupby("term_id"):
        keys = members["gene_key"].drop_duplicates()
        present = label.reindex(keys).dropna()
        if term in COMPOSITION_COMPLEXES:
            counts = state.reindex(keys).fillna("unlabelled").value_counts()
            composition.append({"complex_id": term, "complex": COMPOSITION_COMPLEXES[term], "corum_name": names.get(term, ""),
                                "subunits": int(len(keys)), **{name: int(counts.get(name, 0)) for name in
                                                               ("pan_essential", "context_specific", "non_essential", "unlabelled")}})
        if len(present) < CONTEXT_COMPLEX_MINIMUM:
            continue
        hits = int(present.sum())
        found = sorted(symbol.reindex(present.index[present.to_numpy() == 1]).dropna().astype(str))
        rows.append({"complex_id": term, "corum_name": names.get(term, ""), "subunits": int(len(keys)),
                     "labelled": int(len(present)), "context_specific": hits,
                     "fold_enrichment": (hits / len(present)) / (drawn / total),
                     "p": float(hypergeom.sf(hits - 1, total, len(present), drawn)), "context_symbols": ";".join(found)})
    table = pd.DataFrame(rows)
    table["q"] = benjamini_hochberg_adjusted(table["p"].to_numpy("float64"))
    table = table.sort_values(["q", "p", "complex_id"], kind="stable").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table, pd.DataFrame(composition)


def lineage_tables(matrix: pd.DataFrame, context: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = pd.read_csv(MODELS, usecols=["ModelID", "OncotreeLineage"]).set_index("ModelID")["OncotreeLineage"]
    lineage = models.reindex(matrix.index)
    if lineage.isna().any():
        raise ValueError(f"{int(lineage.isna().sum())} dependency models have no lineage")
    sizes = lineage.value_counts()
    counted = list(sizes.index[sizes >= LINEAGE_MINIMUM])
    positives = context[context.label == 1].set_index("gene_key")
    measured = [key for key in positives.index if key in matrix.columns]
    values = matrix[measured]
    dependent = (values > DEPENDENT_ABOVE).astype("float64").where(values.notna())
    share = dependent.groupby(lineage.to_numpy()).mean().T[counted]
    overall = dependent.mean(axis=0)
    genes = pd.DataFrame({"gene_key": share.index, "gene_symbol": positives["gene_symbol"].reindex(share.index).to_numpy(),
                          "dependent_share": overall.reindex(share.index).to_numpy(),
                          "highest_share": share.max(axis=1).to_numpy(), "highest_lineage": share.idxmax(axis=1).to_numpy(),
                          "dependent_lineages": (share >= LINEAGE_SHARE).sum(axis=1).to_numpy(),
                          "paralogue": positives["paralogue"].reindex(share.index).to_numpy(),
                          "count": positives["count"].reindex(share.index).to_numpy()})
    genes["selectivity"] = genes["highest_share"] - genes["dependent_share"]
    genes = genes[genes["dependent_lineages"] >= 1].sort_values(["selectivity", "gene_key"], ascending=[False, True],
                                                                kind="stable").reset_index(drop=True)
    genes.insert(0, "rank", np.arange(1, len(genes) + 1))
    drawn = genes.head(HEATMAP_GENES)
    heat = share.loc[drawn["gene_key"]].rename_axis("gene_key").reset_index().melt("gene_key", var_name="lineage",
                                                                                   value_name="dependent_share")
    heat.insert(1, "gene_symbol", heat["gene_key"].map(dict(zip(drawn["gene_key"], drawn["gene_symbol"]))))
    lineages = pd.DataFrame({"lineage": counted, "cell_lines": [int(sizes[name]) for name in counted]})
    return genes, heat, lineages


def write_table(frame: pd.DataFrame, name: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / f"{name}.csv", index=False)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(SEED)
    paralogues = paralogue_keys()
    names = go_names()
    universe = pd.read_parquet(DEPMAP_UNIVERSE, columns=["gene_key", "gene_symbol", "official_common_essential",
                                                         "strongly_selective"]).astype({"gene_key": str}).set_index("gene_key")
    symbols = universe["gene_symbol"]
    effect = gene_effect_means()[GENE_EFFECT_COLUMN]
    effect.index = effect.index.astype(str)
    matrix = dependency_matrix()
    matrix.columns = matrix.columns.astype(str)
    dependent = (matrix > DEPENDENT_ABOVE).sum(axis=0) / matrix.notna().sum(axis=0)

    depmap = depmap_table(paralogues, effect, dependent, symbols)
    context = context_table(paralogues, dependent, symbols)
    scopes_depmap, scopes_context = scopes_of(depmap, generator), scopes_of(context, generator)
    tests = strength_tests(scopes_depmap)
    complexes, subunits = strength_complexes(effect, universe, depmap)
    write_table(complexes, "strength_complexes")
    write_table(subunits, "strength_complex_subunits")
    breadth, part = breadth_tables(breadth_frame(paralogues, dependent), generator)
    tests += part
    write_table(breadth, "breadth_bands")
    classes, part = class_tests(scopes_depmap, scopes_context)
    tests += part
    write_table(classes, "class_profile")
    go = pd.concat([enrichment(depmap.set_index("gene_key")["label"] == 1, "pan_essential", names, symbols),
                    enrichment(context.set_index("gene_key")["label"] == 1, "context_specific", names, symbols)],
                   ignore_index=True)
    write_table(go, "class_go")
    enriched, composition = context_complexes(depmap, context)
    write_table(enriched, "context_complexes")
    write_table(composition, "complex_composition")
    lineage_genes, heat, lineages = lineage_tables(matrix, context)
    write_table(lineage_genes, "lineage_genes")
    write_table(heat, "lineage_heatmap")
    write_table(lineages, "lineages")
    write_table(pd.DataFrame(tests), "biology_tests")
    depmap.to_csv(strength_genes_path(), index=False)
    context.to_csv(context_genes_path(), index=False)


if __name__ == "__main__":
    main()
