"""Cut the 15 methods' inputs into 22 feature groups, write them to the array cache and write blocks.csv.

Run from the release root: python analysis/feature_groups/blocks.py [--workers N]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from egbench.data import feature_directory

from common.feature_blocks import (
    ALONE_COHORTS, BLOCKS, CACHE, CATEGORIES, CCCFS_TERMS, EPGAT_SOURCE, FOLDS, GENE_KEY, HELP_BIO_NAMES,
    METHOD_NAMES, METHODS_DIRECTORY, SPLIT, WORK, block_path, go_names, induced_degree, names_path, read_indexed,
    read_vector, run_pool, write_table,
)

ESM2 = WORK.parent.parent / "foundation_model_probes" / "work" / "esm2_650m_mean.parquet"
MAHI_WIDTH = 3666
MAHI_SLICES = {"esmc": (0, 1152), "deepsea": (1152, 3154), "mahi_gnn": (3154, 3666)}


def fold_dir(method: str, cohort: str, fold: int):
    return feature_directory(METHODS_DIRECTORY / method, cohort, split_strategy=SPLIT, outer_fold=fold)


def save(cohort: str, fold: int, key: str, frame: pd.DataFrame, genes: pd.Index, names: list[str]) -> dict:
    values = frame.reindex(genes).to_numpy("float64")
    if frame.index.difference(genes).size or genes.difference(frame.index).size:
        raise ValueError(f"{key} {cohort} fold {fold}: the block's genes differ from the cohort's")
    if values.shape[1] != len(names):
        raise ValueError(f"{key} {cohort} fold {fold}: {values.shape[1]} columns and {len(names)} names")
    path = block_path(cohort, fold, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(values))
    names_path(cohort, fold, key).write_text(json.dumps(names, ensure_ascii=False))
    return {"cohort_id": cohort, "fold": fold, "block": key, "columns": int(values.shape[1]),
            "missing": int(np.isnan(values).sum())}


def build_cohort(cohort: str) -> list[dict]:
    written = []
    help0 = read_indexed(fold_dir("help", cohort, 0) / "features.parquet")
    genes = pd.Index(sorted(help0.index), dtype="object", name=GENE_KEY)
    (CACHE / cohort).mkdir(parents=True, exist_ok=True)
    (CACHE / cohort / "genes.txt").write_text("\n".join(genes) + "\n")

    bio = [f"bio__{i}" for i in range(18)]
    cc = [c for c in help0.columns if c.startswith("cc__")]
    n2v = [c for c in help0.columns if c.startswith("n2v__")]
    if len(bio) + len(cc) + len(n2v) != help0.shape[1] or len(cc) != 3301 or len(n2v) != 128:
        raise ValueError(f"HELP {cohort}: unexpected block widths")
    terms = pd.read_csv(CCCFS_TERMS).set_index("column")
    written.append(save(cohort, 0, "help_bio", help0[bio], genes, list(HELP_BIO_NAMES)))
    written.append(save(cohort, 0, "cccfs", help0[cc], genes, [str(terms.loc[c, "term_name"]) for c in cc]))
    written.append(save(cohort, 0, "string_n2v", help0[n2v], genes, n2v))

    mahi = read_vector(fold_dir("mahi", cohort, 0) / "features.parquet", "embedding", MAHI_WIDTH)
    for key, (start, stop) in MAHI_SLICES.items():
        written.append(save(cohort, 0, key, mahi.iloc[:, start:stop], genes,
                            [f"mahi_{i}" for i in range(start, stop)]))

    plm = read_indexed(fold_dir("proteomelm", cohort, 0) / "features.parquet")
    written.append(save(cohort, 0, "proteomelm", plm, genes, list(plm.columns)))
    evo = read_vector(fold_dir("esstfnet", cohort, 0) / "evo2_embeddings.parquet", "embedding", 4096)
    written.append(save(cohort, 0, "evo2", evo, genes, [f"evo2_{i}" for i in range(4096)]))
    esm2 = read_indexed(ESM2).drop(columns=["cached_length", "truncated"])
    written.append(save(cohort, 0, "esm2", esm2.loc[esm2.index.intersection(genes)], genes, list(esm2.columns)))
    xgep = read_indexed(fold_dir("xgep", cohort, 0) / "features.parquet")
    written.append(save(cohort, 0, "xgep", xgep, genes, list(xgep.columns)))

    deeply = read_indexed(fold_dir("deeplyessential", cohort, 0) / "features.parquet")
    written.append(save(cohort, 0, "seq_composition", deeply, genes, list(deeply.columns)))
    deephe = read_indexed(fold_dir("deephe", cohort, 0) / "features.parquet")
    node2vec = [c for c in deephe.columns if c.startswith("node2vec_")]
    written.append(save(cohort, 0, "biogrid_n2v", deephe[node2vec], genes, node2vec))
    deepsnap0 = read_indexed(fold_dir("deepsnap", cohort, 0) / "features.parquet")
    corum = ["corum_membership_count", "corum_neighbor_complex_ratio_sum"]
    pfam = ["pfam_domain_instance_count", "pfam_unique_domain_type_count", "pfam_sidf"]
    written.append(save(cohort, 0, "corum", deepsnap0[corum], genes, corum))
    written.append(save(cohort, 0, "pfam", deepsnap0[pfam], genes, pfam))

    static = read_indexed(EPGAT_SOURCE / "generated" / f"{cohort}_static_features.parquet")
    compartments = [c for c in static.columns if c.startswith("compartment__")] + ["localization_missing"]
    orthology = [c for c in static.columns if c.startswith("ortholog__")] + ["orthology_missing"]
    if len(compartments) != 12 or len(orthology) != 195 or len(compartments) + len(orthology) != static.shape[1]:
        raise ValueError(f"EPGAT {cohort}: unexpected static columns")
    written.append(save(cohort, 0, "compartments", static[compartments], genes, compartments))
    written.append(save(cohort, 0, "orthology", static[orthology], genes, orthology))
    degree = pd.DataFrame({"ppi_induced_degree": induced_degree(genes)}, index=genes)
    written.append(save(cohort, 0, "ppi_degree", degree, genes, ["ppi_induced_degree"]))

    obo = go_names()
    for fold in FOLDS:
        pheg = read_indexed(fold_dir("pheg", cohort, fold) / "features.parquet")
        selected = json.loads((fold_dir("pheg", cohort, fold) / "selected_features.json").read_text())
        if list(pheg.columns) != selected["selected_feature_names"]:
            raise ValueError(f"Pheg {cohort} fold {fold}: column names differ from selected_features.json")
        written.append(save(cohort, fold, "zcurve", pheg, genes, list(pheg.columns)))
        fasttext = read_indexed(fold_dir("ednn_eg", cohort, fold) / "features.parquet")
        written.append(save(cohort, fold, "fasttext", fasttext, genes, list(fasttext.columns)))
        manifest = json.loads((fold_dir("esstfnet", cohort, fold) / "feature_selection_manifest.json").read_text())
        physchem = read_vector(fold_dir("esstfnet", cohort, fold) / "physicochemical_features.parquet",
                               "features", 4250)
        written.append(save(cohort, fold, "physchem", physchem, genes,
                            list(manifest["selected_raw_feature_names_ordered"])))
        pcs = read_indexed(fold_dir("esssubgraph", cohort, fold) / "features.parquet")
        written.append(save(cohort, fold, "tcga_pca", pcs, genes, list(pcs.columns)))
        deepsnap = read_indexed(fold_dir("deepsnap", cohort, fold) / "features.parquet")
        go_manifest = json.loads((fold_dir("deepsnap", cohort, fold) / "feature_manifest.json").read_text())
        go_terms = sorted(go_manifest["selected_go_terms"], key=lambda item: item["rank"])
        go_columns = [f"go_enrichment_{i:03d}" for i in range(1, 101)]
        go_labels = [f"{item['go_id']} {obo.get(item['go_id']) or ''}".strip() for item in go_terms]
        written.append(save(cohort, fold, "go_enrichment", deepsnap[go_columns], genes, go_labels))
    print(f"[blocks] {cohort} done, {len(genes):,} genes", flush=True)
    return written


def inventory(written: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block in BLOCKS:
        widths = written[written.block == block.key]
        rows.append({
            "block": block.key, "name": block.name, "category": CATEGORIES[block.category],
            "columns": int(widths["columns"].max()), "readers": "; ".join(METHOD_NAMES[m] for m in block.readers),
            "fold_dependent": block.fold_dependent, "per_column_importance": block.per_column,
            "missing_values": int(widths.missing.sum()), "source": block.source,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=len(ALONE_COHORTS))
    arguments = parser.parse_args()
    results = run_pool(build_cohort, list(ALONE_COHORTS), arguments.workers, "blocks")
    written = pd.DataFrame([row for cohort in ALONE_COHORTS for row in results[cohort]])
    write_table(inventory(written), "blocks")
    print("wrote blocks.csv", flush=True)


if __name__ == "__main__":
    main()
