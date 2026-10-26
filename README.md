# EGBench

Code and datasets of EGBench, a benchmark of 15 published predictors of human gene essentiality on six tasks under protein-family cross-validation.

## Use EGBench from an AI agent

EGBench is also a Model Context Protocol (MCP) server, so any MCP client can query the benchmark, check and score a new predictor, split your own labels into protein-family folds, train the core feature set model on them and analyse genes. Eleven tools need no downloads; four tools download public data on first use (3.6 GB, CPU). Tested on macOS (the tools without downloads) and Linux x86_64 (all tools); on Windows the server runs through WSL only.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), which runs the server in an environment of its own.

2. Start the server once in a terminal.

   ```
   uvx --from "egbench[mcp] @ git+https://github.com/tubic/EGBench" egbench-mcp
   ```

   The first start downloads the package and its 42 dependencies and imports them. It took 3 to 4 minutes with an empty uv cache and about 1 minute with the dependencies cached. When the terminal stops printing, press Ctrl+C. Later starts take a few seconds. This step is what lets a client connect: Claude Code 2.1.287 reported `Failed to connect — Request timed out` after 35 seconds when it started the server for the first time itself, and `Connected` once the server had been started in a terminal.

3. Register the server in your client with the same command.

   ```
   claude mcp add egbench -- uvx --from "egbench[mcp] @ git+https://github.com/tubic/EGBench" egbench-mcp
   claude mcp list
   ```

   For Claude Desktop, add the entry below to `claude_desktop_config.json`. A client started from the desktop may not search your shell's `PATH`; then write the full path that `which uvx` prints as the command.

   ```
   {"mcpServers": {"egbench": {"command": "uvx",
                               "args": ["--from", "egbench[mcp] @ git+https://github.com/tubic/EGBench", "egbench-mcp"],
                               "env": {"EGBENCH_DATA_ROOT": "/path/to/data"}}}}
   ```

   Downloads and the tables built from them go to `EGBENCH_DATA_ROOT`, `~/egbench-data` when it is unset.

4. Ask for something. For example:

   - Which tasks does EGBench contain? On the Mouse task, which predictor ranks first, and is it significantly better than the second?
   - Score the predictor in `agent/examples/deg_logistic.csv` on the DEG task and tell me its rank among the published predictors.
   - `agent/examples/mouse_lethality_labels.csv` lists mouse knockout calls by gene symbol (1 lethal, 0 viable). Split them into protein-family folds as a task named mouse_lethality and train the core feature set model on it. (Your own labels work the same way: a CSV with a gene column and a 0/1 label column.)
   - Describe these genes by the four hallmarks and tell me which processes and complexes they are enriched in: PSMA1, PSMB2, PSMC3, PSMD1, POLR2A.

   A relative path such as `agent/examples/deg_logistic.csv` is read from the package's copy of the repository; give any other file by its full path. The first call of a tool that downloads data reports what it needs and how long it takes, and the three DepMap files that the dependency analysis reads can only be downloaded with a browser; save them at the path the tool names. [agent/README.md](https://github.com/tubic/EGBench/blob/main/agent/README.md) lists the tools, the data they download, the commands that rerun the 15 methods and the analyses, and how the results were checked.

## Layout

```
datasets/              benchmark datasets: labels and protein-family fold assignments
  DEG/ DepMap/ Context/ GeneEffect/ Blood/ Mouse/   the six benchmark tasks
  DepMap23Q4/          DepMap 23Q4 labels for the prospective validation
  homology/            protein families (cluster_map) and homology edges (filtered_edges)
src/egbench/           library: fold contracts, data readers, metrics, evaluation and benchmark report
methods/<method>/      the 15 methods; run.py has the subcommands features, preflight, train and predict
analysis/<analysis>/   code of each manuscript analysis and its result tables (results/)
analysis/common/       helpers shared by the analyses
agent/                 MCP server that gives AI agents the benchmark as tools, a client for open-weight models, and commands that rerun the methods from public data
```

Each task directory holds `folds/` (`split_assignments`, `training_labels/fold_<k>`, `fold_genes/fold_<k>_genes`, `evaluation_truth`, `common_binary` or `common_continuous`, `cluster_projection`, all parquet, and `summary.json`) and, except GeneEffect, `data/` with the label tables.

Not distributed: external data (`data/`, below), method inputs (`methods/<method>/feature/source/`), fold features (`methods/<method>/feature/`), trained models (`methods/<method>/model/`), third-party code fetched by the commands below, and analysis caches (`analysis/<analysis>/work/`). The scripts that built the method inputs are included in each method directory (`acquire_assets.py`, `build_*.py`, `cache_*.py`, `embed_*.py`).

## Environments

| Environment | Main packages |
|---|---|
| analyses and report | Python 3.11.16, numpy 2.4.6, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.0, pyarrow 22.0.0 |
| methods (one environment each) | Python 3.11.16 (Bingo, ProteomeLM-Ess: 3.12.3), numpy 2.4.6, pandas 2.3.3, scikit-learn 1.9.0, pyarrow 22.0.0 |
| neural methods | torch 2.6.0; Bingo and EssSubgraph also torch-geometric 2.6.1; EssTFNet einops 0.8.2; eDNN-EG fasttext 0.9.3 |
| tree methods | HELP lightgbm 4.6.0 and xgboost 3.0.5; Mahi and XGEP-XGB xgboost 3.2.0 |
| ESM-2 embeddings | fair-esm 2.0.0, torch 2.6.0 |
| agent server, client and tools | Python 3.11.17, numpy 2.4.6, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.0, pyarrow 22.0.0, mcp 2.3.0; all versions in `agent/requirements.txt` |

Install the library with `pip install -e .`, or set `PYTHONPATH=$PWD/src`. All commands run from the repository root. Before a run:

```
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## External data

| Path | Source |
|---|---|
| `data/depmap_26q1/{CRISPRGeneEffect,CRISPRGeneDependency,Model}.csv` | DepMap Public 26Q1 |
| `data/hgnc/hgnc_complete_set_2026-07-07.txt` | HGNC complete set, 2026-07-07 |
| `data/source/annotations/gene2pubmed.gz` | NCBI gene2pubmed, downloaded 2026-09-02 |
| `data/source/annotations/corum_human_complexes.tsv` | CORUM 5.3 (2026-04-14), human complexes |
| `data/source/annotations/goa_human.gaf.gz` | GOA human, downloaded 2026-07-13 |
| `data/source/annotations/go-basic.obo` | Gene Ontology go-basic, downloaded 2026-09-12 |
| `data/source/annotations/Compara.115.protein_default.homologies.tsv.gz` | Ensembl Compara 115 homologies of Homo sapiens |
| `data/source/annotations/human_compartment_integrated_full.tsv` | JensenLab COMPARTMENTS, integrated channel |
| `data/source/annotations/Human_CCcfs.csv` | HELP cellular component features, Zenodo 12597679 |
| `data/source/annotations/impc_24.0_genotype_phenotype.csv.gz` | IMPC release 24.0 genotype-phenotype assertions |
| `data/source/annotations/MPheno_OBO.ontology` | Mammalian Phenotype Ontology, MGI, downloaded 2026-09-05 |
| `data/source/expression/tcga_gene_by_sample.parquet`, `tcga_expression_qa.json` | TCGA RSEM expression, Zenodo 7885656, converted by `methods/xgep/code/method_features/multiomics.py` |
| `data/pretrained/esm2_t30_150M_UR50D.pt`, `esm2_t36_3B_UR50D.pt` | ESM-2 checkpoints, dl.fbaipublicfiles.com/fair-esm/models |
| `data/pretrained/geneformer/{model.safetensors,token_dictionary_gc30M.pkl}` | Geneformer-V1-10M, huggingface.co/ctheodoris/Geneformer at 1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5 |

## Methods

Two methods import third-party code at run time. Fetch it once:

```
cd methods/esstfnet && python acquire_assets.py && cd ../..   # EssTFNet
cd methods/help && python acquire_assets.py && cd ../..       # svelearn
```

One fold of one method, with that method's environment (`<cohort>` is `deg`, `depmap`, `context`, `blood`, `mouse` or `depmap_23q4`; `<k>` is 0 to 4; GeneEffect is scored on the DepMap models):

```
M=<method>; A="--dataset <cohort> --split-strategy family_5fold --outer-fold <k>"
python methods/$M/run.py features $A
python methods/$M/run.py preflight $A
python methods/$M/run.py train $A
python -m egbench evaluate methods/$M/model/<cohort>/family_5fold/fold_<k>
```

`train` writes `methods/<method>/model/<cohort>/family_5fold/fold_<k>/` and refuses an existing directory. `predict --model-dir <fold dir> --genes <parquet with a gene_key column> --output <dir>` scores other genes with a trained fold. HELP, Mahi, Pheg and XGEP-XGB run on CPU (Pheg and eDNN-EG with one thread); the other methods need a GPU. Method identifiers: `bingo deepcelless deephe deeplyessential deepsnap ednn_eg epgat esssubgraph esstfnet gcnn_sfm help mahi pheg proteomelm xgep`.

## Benchmark report

After all methods are trained on the six cohorts and five folds:

```
python -m egbench report --workers 16
```

It writes the leaderboard, the fusion of the top five methods, the paired comparisons, the attribute-matched subsets and the false positives to `analysis/benchmark_report/results/` (about 10 minutes).

## Agent interface

[agent/README.md](https://github.com/tubic/EGBench/blob/main/agent/README.md) is the reference of the MCP server: the tools, the data they download, the commands that install the environments of the 15 methods, build their inputs from public data, train and check folds, score new genes with a trained method, rebuild the benchmark report and rerun the analyses below, and how the results were checked against the manuscript.

## Analyses

Run from the repository root with `export PYTHONPATH=$PWD/src:$PWD/analysis`, in the analysis environment unless noted, in this order. Each script writes its tables to `analysis/<analysis>/results/`.

```
# label sensitivity
python analysis/label_sensitivity/screen_agreement.py
python analysis/label_sensitivity/label_definitions.py 15

# pretrained model embeddings; the first three in the Bingo environment on a GPU
python analysis/foundation_model_probes/esm2_pool.py
python analysis/foundation_model_probes/esm2_extract.py data/pretrained/esm2_t30_150M_UR50D.pt esm2_150m 16000
python analysis/foundation_model_probes/esm2_extract.py data/pretrained/esm2_t36_3B_UR50D.pt esm2_3b 4000
python analysis/foundation_model_probes/geneformer_embeddings.py
python analysis/foundation_model_probes/embedding_probes.py --workers 12

# parameter counts; needs <env-root>/egbench-<method>/bin/python for the 15 methods
python analysis/parameter_counts/parameter_counts.py --env-root <env-root>

# same input, neural networks; in the DeepHE environment on GPUs
python analysis/same_input_mlp/same_input_mlp.py --gpus 0,1 --workers 4

# feature groups and core feature set
python analysis/feature_groups/cccfs_terms.py
python analysis/feature_groups/blocks.py --workers 5
python analysis/feature_groups/alone.py --workers 12
python analysis/feature_groups/forward.py --workers 12
python analysis/feature_groups/core_fixed.py --workers 12
python analysis/feature_groups/importance.py --workers 10

# hallmarks of essential genes
python analysis/essential_gene_hallmarks/pc_importance.py --workers 10
python analysis/essential_gene_hallmarks/feature_biology.py --workers 12
python analysis/essential_gene_hallmarks/four_properties.py --workers 12

# research attention
python analysis/research_attention/attention_trend.py --workers 5
python analysis/research_attention/excluded_genes.py
python analysis/research_attention/attention_properties.py
python analysis/research_attention/attention_followup.py
python analysis/research_attention/input_models.py --workers 12
python analysis/research_attention/network_degree.py

# exceptions to the hallmarks
python analysis/hallmark_exceptions/hallmark_groups.py
python analysis/hallmark_exceptions/exception_biology.py

# prospective validation
python analysis/prospective_validation/prospective_core.py --workers 12
python analysis/prospective_validation/prospective_biology.py

# dependency strength and context
python analysis/dependency_strength_context/strength_context.py
python analysis/dependency_strength_context/context_biology.py
python analysis/dependency_strength_context/complex_dependence.py

# mouse lethal stages
python analysis/mouse_lethal_stages/lethal_stages.py
```

The analyses read the method inputs, fold features and predictions under `methods/<method>/feature/` and `methods/<method>/model/`. Run one after another on 96 cores and 24 GB GPUs, the sequence takes about five hours; the forward selection (75 min), the neural networks (40 min), the 3B embeddings (35 min) and the probes (32 min) take longest. 
