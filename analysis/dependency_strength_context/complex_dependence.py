"""Fanconi anaemia core and TNFR1 signalling complexes: genome-wide co-dependency with their context-specific
subunits and dependency on their subunits across quartiles of TP53 and CFLAR knockout effect.
Run after context_biology.py: python analysis/dependency_strength_context/complex_dependence.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from scipy.stats import t as student_t

from egbench.modules import complex_pairs
from egbench.reference import MAXIMUM_MISSING_FRACTION
from egbench.stats import holm_adjusted

from common.depmap_matrices import DEPENDENT_ABOVE, DEPMAP_UNIVERSE, MODELS, dependency_matrix, gene_effect_matrix
from dependency_strength_context.context_biology import RESULTS, context_genes_path, strength_genes_path

LINEAGE_MINIMUM = 30
PARTNERS = 3
QUARTILES = ("Q1", "Q2", "Q3", "Q4")
COMPLEXES = (
    {"complex_id": "CORUM:1624", "complex": "Fanconi anaemia core complex", "state_gene": "TP53", "state_sign": 1.0,
     "state": "p53_activity", "markers": ("MDM2",)},
    {"complex_id": "CORUM:8561", "complex": "TNFR1 signalling complex", "state_gene": "CFLAR", "state_sign": -1.0,
     "state": "cflip_dependency", "markers": ("CASP8",)},
)


def gene_classes() -> pd.Series:
    depmap = pd.read_csv(strength_genes_path(), usecols=["gene_key", "label"]).astype({"gene_key": str})
    context = pd.read_csv(context_genes_path(), usecols=["gene_key", "label"]).astype({"gene_key": str})
    state = pd.Series("non_essential", index=pd.Index(pd.concat([depmap["gene_key"], context["gene_key"]]).unique()))
    state.loc[depmap.loc[depmap.label == 1, "gene_key"].to_numpy()] = "pan_essential"
    state.loc[context.loc[context.label == 1, "gene_key"].to_numpy()] = "context_specific"
    return state


def lineage_groups(index: pd.Index) -> pd.Series:
    lineage = pd.read_csv(MODELS, usecols=["ModelID", "OncotreeLineage"]).set_index("ModelID")["OncotreeLineage"].reindex(index)
    if lineage.isna().any():
        raise ValueError(f"{int(lineage.isna().sum())} cell lines have no lineage")
    sizes = lineage.value_counts()
    return lineage.where(lineage.map(sizes) >= LINEAGE_MINIMUM, "other")


def centred(values, groups: pd.Series):
    return values - values.groupby(groups.to_numpy()).transform("mean")


def pearson_with(matrix: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    values = matrix.to_numpy("float64")
    target = score.reindex(matrix.index).to_numpy("float64")
    seen = ~np.isnan(values) & ~np.isnan(target)[:, None]
    x = np.where(seen, values, 0.0)
    y = np.where(seen, target[:, None], 0.0)
    n = seen.sum(axis=0).astype("float64")
    covariance = (x * y).sum(axis=0) - x.sum(axis=0) * y.sum(axis=0) / n
    spread_x = (x * x).sum(axis=0) - x.sum(axis=0) ** 2 / n
    spread_y = (y * y).sum(axis=0) - y.sum(axis=0) ** 2 / n
    r = covariance / np.sqrt(spread_x * spread_y)
    statistic = r * np.sqrt((n - 2) / (1 - r * r))
    return pd.DataFrame({"r": r, "cell_lines": n.astype("int64"), "p": 2 * student_t.sf(np.abs(statistic), n - 2)},
                        index=matrix.columns)


def spearman_with(matrix: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    target = score.reindex(matrix.index).to_numpy("float64")
    rows = []
    for key in matrix.columns:
        values = matrix[key].to_numpy("float64")
        seen = ~np.isnan(values) & ~np.isnan(target)
        result = spearmanr(values[seen], target[seen])
        rows.append((float(result.statistic), float(result.pvalue)))
    return pd.DataFrame(rows, index=matrix.columns, columns=["spearman", "spearman_p"])


def subunit_table(spec: dict, keys: pd.Index, classes: pd.Series, symbols: pd.Series, share: pd.Series,
                  mean_effect: pd.Series) -> pd.DataFrame:
    table = pd.DataFrame({"complex_id": spec["complex_id"], "complex": spec["complex"], "gene_key": keys,
                          "gene_symbol": symbols.reindex(keys).to_numpy(),
                          "gene_class": classes.reindex(keys).fillna("unlabelled").to_numpy(),
                          "dependent_share": share.reindex(keys).to_numpy(),
                          "gene_effect_mean": mean_effect.reindex(keys).to_numpy()})
    return table.sort_values(["dependent_share", "gene_key"], ascending=[False, True], kind="stable").reset_index(drop=True)


def codependency_table(spec: dict, score: pd.Series, others: pd.DataFrame, groups: pd.Series,
                       symbols: pd.Series) -> pd.DataFrame:
    table = pearson_with(others, score)
    within = pearson_with(centred(others, groups), centred(score, groups))
    table["bonferroni_p"] = np.minimum(1.0, table["p"] * len(table))
    table["r_within_lineage"] = within["r"]
    table["p_within_lineage"] = within["p"]
    table = table.rename_axis("gene_key").reset_index()
    table.insert(1, "gene_symbol", symbols.reindex(table["gene_key"]).to_numpy())
    table = table.sort_values(["r", "gene_key"], ascending=[False, True], kind="stable").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table.insert(0, "complex", spec["complex"])
    table.insert(0, "complex_id", spec["complex_id"])
    return table


def state_table(spec: dict, state: pd.Series, genes: list[tuple[str, str]], effect: pd.DataFrame, dependent: pd.DataFrame,
                scanned: pd.DataFrame, groups: pd.Series, symbols: pd.Series, classes: pd.Series) -> pd.DataFrame:
    quartile = pd.qcut(state, 4, labels=list(QUARTILES))
    genome = spearman_with(scanned, state)["spearman"]
    block = effect[[key for key, _ in genes]]
    across = spearman_with(block, state)
    within = spearman_with(centred(block, groups), centred(state, groups))
    low, high = quartile == QUARTILES[0], quartile == QUARTILES[-1]
    rows = []
    for key, role in genes:
        flags, values = dependent[key], effect[key]
        row = {"complex_id": spec["complex_id"], "complex": spec["complex"], "state": spec["state"],
               "state_gene": spec["state_gene"], "gene_key": key, "gene_symbol": symbols.get(key, key), "role": role,
               "gene_class": classes.get(key, "unlabelled"), "dependent_share": float(flags.mean())}
        for name in QUARTILES:
            inside = quartile == name
            row[f"share_{name.lower()}"] = float(flags[inside].mean())
            row[f"effect_{name.lower()}"] = float(values[inside].median())
        table = [[int(flags[high].sum()), int(flags[high].notna().sum() - flags[high].sum())],
                 [int(flags[low].sum()), int(flags[low].notna().sum() - flags[low].sum())]]
        row["fisher_p"] = float(fisher_exact(table)[1])
        row["spearman"] = float(across.at[key, "spearman"])
        row["spearman_p"] = float(across.at[key, "spearman_p"])
        row["genome_share_below"] = float((genome < row["spearman"]).mean())
        row["spearman_within_lineage"] = float(within.at[key, "spearman"])
        row["spearman_within_lineage_p"] = float(within.at[key, "spearman_p"])
        rows.append(row)
    table = pd.DataFrame(rows)
    tested = table["role"].isin(["subunit", "partner"])
    table["fisher_holm_p"] = np.nan
    table.loc[tested, "fisher_holm_p"] = holm_adjusted(table.loc[tested, "fisher_p"].to_numpy("float64"))
    table["spearman_holm_p"] = np.nan
    table.loc[tested, "spearman_holm_p"] = holm_adjusted(table.loc[tested, "spearman_p"].to_numpy("float64"))
    return table


def main() -> None:
    effect = gene_effect_matrix()
    effect.columns = effect.columns.astype(str)
    probability = dependency_matrix()
    probability.columns = probability.columns.astype(str)
    if not effect.index.equals(probability.index):
        probability = probability.reindex(effect.index)
    dependent = (probability > DEPENDENT_ABOVE).astype("float64").where(probability.notna())
    universe = pd.read_parquet(DEPMAP_UNIVERSE, columns=["gene_key", "gene_symbol"]).astype({"gene_key": str})
    symbols = universe.set_index("gene_key")["gene_symbol"]
    by_symbol = pd.Series(symbols.index, index=symbols.to_numpy())
    classes = gene_classes()
    groups = lineage_groups(effect.index)
    share = dependent.mean(axis=0)
    mean_effect = effect.mean(axis=0)
    scanned = effect.loc[:, effect.isna().mean(axis=0) <= MAXIMUM_MISSING_FRACTION]
    pairs = complex_pairs().astype({"gene_key": str})

    scans, states = [], []
    for spec in COMPLEXES:
        members = pd.Index(pairs.loc[pairs.term_id == spec["complex_id"], "gene_key"].drop_duplicates())
        measured = members.intersection(effect.columns)
        table = subunit_table(spec, measured, classes, symbols, share, mean_effect)
        specific = table.loc[table.gene_class == "context_specific", "gene_key"].tolist()
        score = effect[specific].mean(axis=1)
        scan = codependency_table(spec, score, scanned.drop(columns=members.intersection(scanned.columns)), groups, symbols)
        scans.append(scan)
        state_key = by_symbol[spec["state_gene"]]
        partners = [key for key in scan["gene_key"] if key != state_key][:PARTNERS]
        genes = [(key, "subunit") for key in table["gene_key"]] + [(key, "partner") for key in partners]
        genes += [(by_symbol[name], "marker") for name in spec["markers"]]
        state = spec["state_sign"] * effect[state_key]
        if state.isna().any():
            raise ValueError(f"{spec['state_gene']} is not measured in every cell line")
        states.append(state_table(spec, state, genes, effect, dependent, scanned, groups, symbols, classes))
        print(f"{spec['complex']}: {len(measured)} measured subunits, {len(specific)} context-specific", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.concat(scans, ignore_index=True).to_csv(RESULTS / "codependency.csv", index=False)
    pd.concat(states, ignore_index=True).to_csv(RESULTS / "state_dependence.csv", index=False)


if __name__ == "__main__":
    main()
