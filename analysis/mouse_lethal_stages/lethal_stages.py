"""Mouse knockout lethal stages against viable knockouts: cell-level dependency, hallmarks, paralogues, gene age,
GO processes, embryo phenotypes and the AUROC of the hallmark count and the 15 methods per stage.
Run after prospective_validation/prospective_core.py: python analysis/mouse_lethal_stages/lethal_stages.py
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom, mannwhitneyu, spearmanr

from egbench.contracts import SEED
from egbench.data import PROJECT_ROOT, primary_scope, scope_gene_keys
from egbench.fusion import pooled
from egbench.modules import process_pairs, restrict_terms
from egbench.reference import GENE_EFFECT_COLUMN, deg_vote_counts, gene_effect_means
from egbench.stats import benjamini_hochberg_adjusted
from egbench.strata import quartile_bands, strata_table

from common.depmap_matrices import DEPMAP_UNIVERSE, dependency_matrix, dependent_share, tissue_variation
from common.feature_blocks import go_names
from common.hallmark_counts import (
    RICH, bootstrap, cells_of, fold_table, hallmark_table, one_per_family, share_row, skip_draws,
)
from common.stats import METHODS, fast_auroc

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WORK = HERE / "work"
ANNOTATIONS = PROJECT_ROOT / "data" / "source" / "annotations"
ASSERTIONS = ANNOTATIONS / "impc_24.0_genotype_phenotype.csv.gz"
ONTOLOGY = ANNOTATIONS / "MPheno_OBO.ontology"
UNIVERSE = PROJECT_ROOT / "datasets" / "Mouse" / "data" / "gene_universe.parquet"
CLUSTERS = PROJECT_ROOT / "datasets" / "homology" / "cluster_map.parquet"
EDGES = PROJECT_ROOT / "datasets" / "homology" / "filtered_edges.parquet"
PROSPECTIVE_HALLMARKS = PROJECT_ROOT / "analysis" / "prospective_validation" / "results" / "hallmarks_depmap_23q4.csv"
TASK = "mouse"
PRE_ROOTS: tuple[str, ...] = ("MP:0013292",)
ORGANOGENESIS_ROOTS: tuple[str, ...] = ("MP:0006207", "MP:0013293")
LATE_ROOTS: tuple[str, ...] = ("MP:0002080", "MP:0002081", "MP:0002058", "MP:0002082", "MP:0010770")
EMBRYONIC_ROOTS: tuple[str, ...] = ("MP:0008762",)
HEART_SEPTATION = "MP:0013294"
PRE = "before_organogenesis"
ORGANOGENESIS = "during_organogenesis"
LATE = "perinatal_and_after"
EMBRYONIC_UNSTAGED = "embryonic_unstaged"
NO_ASSERTION = "no_lethal_assertion"
BANDS: tuple[str, ...] = (PRE, ORGANOGENESIS, LATE)
AFTER, UNSTAGED, VIABLE = "after_e12", "unstaged", "viable"
STAGES: tuple[str, ...] = (PRE, ORGANOGENESIS, AFTER)
GROUPS: tuple[str, ...] = (PRE, ORGANOGENESIS, AFTER, UNSTAGED, VIABLE)
LATE_PROCEDURE = r"E14\.5|E15\.5|E18\.5"
EFFECT_EDGES = (-np.inf, -1.0, -0.5, -0.25, -0.1, np.inf)
EFFECT_NAMES = ("<=-1.0", "-1.0 to -0.5", "-0.5 to -0.25", "-0.25 to -0.1", ">-0.1")
EARLY_NOT_CELL = "before_organogenesis_not_cell_essential"
COUNT = "hallmark_count"
VOTES = 5
TERM_MINIMUM, TERM_MAXIMUM = 5, 200
ALL, FAMILY = "all_genes", "one_gene_per_family"
GENE_COLUMNS = ["gene_key", "gene_symbol", "stage", "label", "cell_essential", "count", "old", "paralogue",
                "age_layer", "gene_effect_mean"]


def ontology_children() -> dict[str, set[str]]:
    parents: dict[str, list[str]] = {}
    current: dict[str, object] | None = None

    def close() -> None:
        if current is not None and "id" in current and not current.get("obsolete"):
            parents[str(current["id"])] = list(current.get("is_a", []))

    for line in ONTOLOGY.read_text(errors="replace").splitlines():
        line = line.strip()
        if line == "[Term]":
            close()
            current = {}
            continue
        if line.startswith("[") and line.endswith("]"):
            close()
            current = None
            continue
        if current is None:
            continue
        if line.startswith("id: "):
            current["id"] = line[4:]
        elif line.startswith("is_a: "):
            current.setdefault("is_a", []).append(line[6:].split("!")[0].strip())
        elif line == "is_obsolete: true":
            current["obsolete"] = True
    close()
    children: dict[str, set[str]] = {}
    for term, above in parents.items():
        for parent in above:
            children.setdefault(parent, set()).add(term)
    return children


def closure(roots: tuple[str, ...], children: dict[str, set[str]]) -> frozenset[str]:
    found = set(roots)
    stack = list(roots)
    while stack:
        term = stack.pop()
        for child in children.get(term, ()):
            if child not in found:
                found.add(child)
                stack.append(child)
    return frozenset(found)


def homozygous_records(columns: list[str]) -> pd.DataFrame:
    with gzip.open(ASSERTIONS, "rt", errors="replace") as handle:
        records = pd.read_csv(handle, usecols=columns, low_memory=False)
    return records[records["zygosity"].astype("string").str.lower() == "homozygote"]


def gene_bands() -> pd.Series:
    children = ontology_children()
    sets = {
        PRE: closure(PRE_ROOTS, children),
        ORGANOGENESIS: closure(ORGANOGENESIS_ROOTS, children),
        LATE: closure(LATE_ROOTS, children),
        EMBRYONIC_UNSTAGED: closure(EMBRYONIC_ROOTS, children),
    }
    rows = homozygous_records(["marker_accession_id", "zygosity", "mp_term_id"])
    rows = rows.dropna(subset=["marker_accession_id", "mp_term_id"]).reset_index(drop=True)
    universe = pd.read_parquet(UNIVERSE, columns=["gene_key", "mgi_id"]).dropna(subset=["mgi_id"])
    mapping = pd.Series(universe["gene_key"].astype("object").to_numpy(), index=pd.Index(universe["mgi_id"].astype("object")))
    rows = rows.assign(gene=mapping.reindex(rows["marker_accession_id"].astype("object")).to_numpy())
    rows = rows.loc[pd.notna(rows["gene"])]
    flags = {name: set(pd.unique(rows.loc[rows["mp_term_id"].isin(terms), "gene"].to_numpy("object")))
             for name, terms in sets.items()}
    genes = pd.Index(pd.unique(universe["gene_key"].astype("object").to_numpy()))
    band = []
    for gene in genes:
        if gene in flags[PRE]:
            band.append(PRE)
        elif gene in flags[ORGANOGENESIS]:
            band.append(ORGANOGENESIS)
        elif gene in flags[EMBRYONIC_UNSTAGED]:
            band.append(EMBRYONIC_UNSTAGED)
        elif gene in flags[LATE]:
            band.append(LATE)
        else:
            band.append(NO_ASSERTION)
    return pd.Series(band, index=genes, name="band")


def assemble(paralogues: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = None
    for method in METHODS:
        pool = pooled(method, TASK)
        pool["gene_key"] = pool["gene_key"].astype(str)
        pool = pool.sort_values(["fold", "gene_key"], kind="stable").reset_index(drop=True)
        if base is None:
            base = pool[["gene_key", "fold", "label"]].copy()
        elif not pool["gene_key"].equals(base["gene_key"]):
            raise ValueError(f"{method}: the scored genes differ")
        base[method] = pool["y_score"].to_numpy("float64")
    base["label"] = base["label"].astype("int64")
    keys = pd.Index(base["gene_key"], dtype="object")

    hallmarks, _ = hallmark_table(TASK)
    hallmarks["gene_key"] = hallmarks["gene_key"].astype(str)
    base = base.merge(hallmarks.drop(columns=["label"]), on="gene_key", how="left", validate="one_to_one")
    if base["count"].isna().any():
        raise ValueError("an evaluated Mouse gene has no hallmark")
    base[COUNT] = base["count"].astype("float64")

    band = gene_bands()
    band.index = band.index.astype(str)
    stage = band.reindex(keys).to_numpy("object")
    base["group"] = np.where(base["label"].to_numpy() == 1, stage, VIABLE)
    if not base.loc[base.label == 1, "group"].isin(BANDS).all():
        raise ValueError("a Mouse positive falls outside the three stage bands")

    depmap = pd.read_parquet(DEPMAP_UNIVERSE, columns=["gene_key", "gene_symbol", "official_common_essential", "strongly_selective"])
    depmap["gene_key"] = depmap["gene_key"].astype(str)
    depmap = depmap.set_index("gene_key")
    base["gene_symbol"] = depmap["gene_symbol"].reindex(keys).to_numpy()
    base["depmap_common_essential"] = depmap["official_common_essential"].astype("float64").reindex(keys).to_numpy()
    votes = deg_vote_counts()
    votes.index = votes.index.astype(str)
    base["deg_votes"] = votes.reindex(keys).to_numpy()
    base["cell_essential"] = ((base["depmap_common_essential"] == 1) | (base["deg_votes"] >= VOTES)).astype("int64")
    effect = gene_effect_means()[GENE_EFFECT_COLUMN]
    effect.index = effect.index.astype(str)
    base["gene_effect_mean"] = effect.reindex(keys).to_numpy()
    variation = tissue_variation()
    variation.index = variation.index.astype(str)
    base["tissue_cv"] = variation.reindex(keys).to_numpy()
    table = strata_table(TASK)
    table["gene_key"] = table["gene_key"].astype(str)
    base["quartile"] = pd.Series(quartile_bands(table["pubmed_count"]).to_numpy(), index=table["gene_key"]).reindex(keys).to_numpy()

    records = homozygous_records(["marker_accession_id", "zygosity", "procedure_name", "mp_term_id", "mp_term_name"])
    universe = pd.read_parquet(UNIVERSE, columns=["gene_key", "mgi_id"]).dropna(subset=["mgi_id"])
    mapping = dict(zip(universe["mgi_id"].astype(str), universe["gene_key"].astype(str)))
    records = records.assign(gene_key=records["marker_accession_id"].map(mapping)).dropna(subset=["gene_key"])
    late = records["procedure_name"].astype(str).str.contains(LATE_PROCEDURE, regex=True)
    seen = set(records.loc[late | (records["mp_term_id"] == HEART_SEPTATION), "gene_key"])
    base["late_record"] = base["gene_key"].isin(seen)
    stage = base["group"].to_numpy("object").copy()
    perinatal = base["group"].to_numpy() == LATE
    stage[perinatal & base["late_record"].to_numpy()] = AFTER
    stage[perinatal & ~base["late_record"].to_numpy()] = UNSTAGED
    base["stage"] = stage
    base["paralogue"] = base["gene_key"].isin(paralogues)
    base["rich"] = base["count"] >= RICH
    late_records = records[late & records["gene_key"].isin(set(base.loc[base.stage == AFTER, "gene_key"]))]
    phenotypes = (late_records.groupby("mp_term_name")["gene_key"].nunique().sort_values(ascending=False)
                  .rename("genes").reset_index())
    phenotypes.insert(0, "rank", np.arange(1, len(phenotypes) + 1))
    phenotypes["after_e12_genes"] = int((base.stage == AFTER).sum())
    clusters = pd.read_parquet(CLUSTERS, columns=["gene_key", "cluster_id"]).astype(str).set_index("gene_key")["cluster_id"]
    base["cluster_id"] = clusters.reindex(base["gene_key"]).to_numpy()
    base["cluster_id"] = base["cluster_id"].fillna(base["gene_key"])
    return base, phenotypes


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


def stage_profile(scopes: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[dict]]:
    rows, tests = [], []
    columns = ("cell_essential", "rich", "complex", "old", "connected", "expressed", "paralogue")
    for scope, frame in scopes.items():
        parts = {name: frame[frame.stage == name] for name in GROUPS}
        for name, part in parts.items():
            row = {"scope": scope, "stage": name, "genes": int(len(part)),
                   "gene_effect_median": float(part["gene_effect_mean"].median()),
                   "tissue_cv_median": float(part["tissue_cv"].median())}
            for column in columns:
                _, _, share, half = share_row(part[column].astype(bool))
                row[f"{column}_share"], row[f"{column}_half_width"] = share, half
            for layer in range(1, 10):
                row[f"age_layer_{layer}"] = float((part["age_layer"] == layer).mean())
            rows.append(row)
        for a, b in ((PRE, ORGANOGENESIS), (ORGANOGENESIS, AFTER), (PRE, AFTER), (AFTER, VIABLE)):
            for column in ("cell_essential", "rich", "old", "paralogue"):
                tests.append(test_row("F", scope, f"{column}_share", parts[a][column].astype(bool),
                                      parts[b][column].astype(bool), a, b))
        tests.append(test_row("F", scope, "bilaterian_layer_share", parts[AFTER]["age_layer"] == 2,
                              parts[PRE]["age_layer"] == 2, AFTER, PRE))
        tests.append(rank_row("F", scope, "tissue_cv", parts[AFTER]["tissue_cv"], parts[VIABLE]["tissue_cv"], AFTER, VIABLE))
    return pd.DataFrame(rows), tests


def effect_bands(scopes: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[dict]]:
    rows, tests = [], []
    order = {name: position for position, name in enumerate(STAGES)}
    for scope, frame in scopes.items():
        band = pd.cut(frame["gene_effect_mean"], list(EFFECT_EDGES), labels=list(EFFECT_NAMES))
        table = pd.crosstab(band, frame["stage"]).reindex(index=list(EFFECT_NAMES), columns=list(GROUPS), fill_value=0)
        for name in EFFECT_NAMES:
            total = int(table.loc[name].sum())
            for stage in GROUPS:
                rows.append({"scope": scope, "gene_effect_band": name, "stage": stage, "genes": int(table.at[name, stage]),
                             "band_genes": total, "share": float(table.at[name, stage] / total)})
        staged = frame[frame.stage.isin(STAGES)]
        tests.append(trend_row("G", scope, "gene_effect_against_stage_order", staged["gene_effect_mean"],
                               staged["stage"].map(order), "gene_effect_mean", "stage_order"))
        essential = frame[frame.cell_essential == 1]
        alive, dead = essential[essential.stage == VIABLE], essential[essential.label == 1]
        tests.append(test_row("G", scope, "paralogue_share_cell_essential", alive["paralogue"], dead["paralogue"],
                              "viable", "lethal"))
        tests.append(rank_row("G", scope, "gene_effect_cell_essential", alive["gene_effect_mean"], dead["gene_effect_mean"],
                              "viable", "lethal"))
    return pd.DataFrame(rows), tests


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


def stage_enrichment(frame: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    genes = frame.set_index("gene_key")
    symbols = genes["gene_symbol"]
    tables = [enrichment(genes["stage"] == stage, stage, names, symbols) for stage in (PRE, ORGANOGENESIS, AFTER, UNSTAGED)]
    tables.append(enrichment((genes["stage"] == PRE) & (genes["cell_essential"] == 0), EARLY_NOT_CELL, names, symbols))
    return pd.concat(tables, ignore_index=True)


def stage_auroc(labels: np.ndarray, fold: np.ndarray, stage: np.ndarray, scores: np.ndarray, name: str) -> float:
    inside = (stage == name) | (labels == 0)
    values = []
    for k in range(5):
        mask = inside & (fold == k)
        y = labels[mask]
        if y.size == 0 or y.sum() in (0, y.size):
            continue
        values.append(fast_auroc(y, scores[mask]))
    return float(np.mean(values)) if values else float("nan")


def stage_methods(frame: pd.DataFrame) -> pd.DataFrame:
    predictors = (COUNT, *METHODS)
    labels = frame["label"].to_numpy("int64")
    fold = frame["fold"].to_numpy()
    stage = frame["stage"].to_numpy("object")
    block = {p: frame[p].to_numpy("float64") for p in predictors}
    lethal = (PRE, ORGANOGENESIS, AFTER, UNSTAGED)

    def statistic(take):
        y, f, s = (labels, fold, stage) if take is None else (labels[take], fold[take], stage[take])
        out = {}
        for predictor in predictors:
            scores = block[predictor] if take is None else block[predictor][take]
            for name in lethal:
                out[f"{predictor}:{name}"] = stage_auroc(y, f, s, scores, name)
            out[f"{predictor}:pre_minus_after"] = out[f"{predictor}:{PRE}"] - out[f"{predictor}:{AFTER}"]
        return out

    table, _ = bootstrap(cells_of(fold, labels), statistic, "stage AUROC")
    parts = table["statistic"].str.split(":", expand=True)
    table["predictor"], table["quantity"] = parts[0], parts[1]
    sizes = {name: int((stage == name).sum()) for name in lethal}
    table["positives"] = table["quantity"].map(sizes)
    return table


def context_draw_sizes() -> list[int]:
    depmap = pd.read_parquet(fold_table("help", "depmap"), columns=["gene_key"])["gene_key"].astype(str)
    context = pd.read_parquet(fold_table("help", "context"), columns=["gene_key"])["gene_key"].astype(str)
    kept = set(map(str, scope_gene_keys("context", primary_scope("context"))))
    matrix = dependency_matrix()
    matrix.columns = matrix.columns.astype(str)
    hallmarks = pd.read_csv(PROSPECTIVE_HALLMARKS, usecols=["gene_key"]).astype({"gene_key": str})
    breadth = int(dependent_share(matrix).reindex(hallmarks["gene_key"]).notna().sum())
    return [len(depmap), int(context.isin(kept).sum()), breadth]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(SEED)
    skip_draws(generator, context_draw_sizes())
    edges = pd.read_parquet(EDGES, columns=["gene_key_a", "gene_key_b"])
    paralogues = set(edges["gene_key_a"].astype(str)) | set(edges["gene_key_b"].astype(str))
    mouse, phenotypes = assemble(paralogues)
    mouse.to_csv(WORK / "lethal_stage_genes.csv", index=False)
    scopes = {ALL: mouse, FAMILY: one_per_family(mouse, generator)}
    profile, tests = stage_profile(scopes)
    profile.to_csv(RESULTS / "lethal_stage_profile.csv", index=False)
    bands, part = effect_bands(scopes)
    tests += part
    bands.to_csv(RESULTS / "lethal_stage_effect_bands.csv", index=False)
    stage_enrichment(mouse, go_names()).to_csv(RESULTS / "lethal_stage_go.csv", index=False)
    phenotypes.to_csv(RESULTS / "lethal_stage_phenotypes.csv", index=False)
    mouse[GENE_COLUMNS].to_csv(RESULTS / "lethal_stage_genes.csv", index=False)
    pd.DataFrame(tests).to_csv(RESULTS / "biology_tests.csv", index=False)
    stage_methods(mouse).to_csv(RESULTS / "lethal_stage_auroc.csv", index=False)


if __name__ == "__main__":
    main()
