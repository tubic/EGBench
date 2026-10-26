# Agent interface

`agent/server.py` is a Model Context Protocol (MCP) server that gives AI agents the benchmark as tools. An agent can query the benchmark, score and compare a new predictor, split its user's labels into the benchmark's protein-family folds, describe a list of genes as the manuscript does, and predict essentiality for other genes with the manuscript's core feature set model. The server installs as a package and holds the fold tables, the labels and the result tables, including the test-fold predictions of the 15 published predictors (`predictions/`, 6.7 MB). It does not hold external data: the tools that need public data download them on first use, and the manifest `external.json` records where each file comes from, its size and its licence.

The 15 published methods and the manuscript's analyses are rerun from the command line in a clone of the repository (Rerunning the methods and the analyses): the commands install the methods' environments, build their inputs from public data, train and check folds, score new genes with a trained method, rebuild the leaderboard and rerun the analyses.

## Layers

| Layer | What a reader can do | What it needs | Tools |
|---|---|---|---|
| 0 | query the tasks, the folds, the leaderboard and single genes; check, score and compare their own predictor; split their own labels into protein-family folds and find the published predictor that fits them best | the installed server; CPU; seconds per call | `list_tasks`, `get_fold`, `leaderboard`, `gene_profile`, `check_predictions`, `evaluate`, `compare`, `make_task`, `score_labels`, `layer_status`, `fetch_external` |
| 1 | describe a list of genes by the manuscript's hallmarks, gene age, literature, dependency and mouse knockouts; test it for enriched processes and complexes; score predictions within groups of genes; train the core feature set model on a task and predict other genes | public downloads of up to 3.6 GB; CPU; the first call builds the tables it reads | `annotate_genes`, `enrichment`, `evaluate_by_strata`, `core_model` |

What each layer-1 tool downloads, and what its first call builds, measured on a 96-core server:

| Tool | Downloads | Built on the first call |
|---|---|---|
| `annotate_genes`, hallmarks aspect; `evaluate_by_strata` | 2,417 MB: HGNC, GENCODE v50, Ensembl Compara 115, BioGRID 5.0.259, CORUM 5.3, gene2pubmed and the TCGA expression archives | gene reference (1.0 min), TCGA matrix (0.9 min), BioGRID edges (0.7 min), orthology (0.3 min), annotation table (0.5 min) |
| `annotate_genes`, dependency aspect | 1,045 MB: the three DepMap 26Q1 files (873 MB, by hand) and the gene reference's sources | gene reference, DepMap table (2.2 min) |
| `annotate_genes`, mouse aspect | 191 MB: IMPC 24.0, the Mammalian Phenotype Ontology, two Ensembl 115 BioMart tables and the gene reference's sources | gene reference, mouse table (under 0.1 min) |
| `enrichment` | 225 MB: GOA, Gene Ontology, CORUM and the gene reference's sources | gene reference |
| `core_model` | 2,182 MB: HELP's two feature tables, Compara, GENCODE, HGNC and the TCGA archives | gene reference, TCGA matrix, orthology, static features (0.3 min); then five models per task: 7.4 minutes for DEG on 32 cores with 9.4 GB of memory, cached for later calls, which take seconds |

`layer_status` gives the same accounting for the data root at hand, and `fetch_external` downloads what is missing.

## Tools

| Tool | Layer | Purpose |
|---|---|---|
| `list_tasks` | 0 | tasks, label definitions, numbers of genes and positives, primary metrics, and the tasks made by `make_task` |
| `get_fold` | 0 | the train, validation and test genes of one fold, with the train and validation labels, as a CSV |
| `leaderboard` | 0 | published ranking of the 15 predictors on one task and the paired tests between the top predictors |
| `gene_profile` | 0 | labels, protein family, test folds and the predictions of the 15 predictors for one gene |
| `check_predictions` | 0 | checks a table of test-fold predictions (columns `gene_key`, `y_score`, `outer_fold`) against the folds of a task |
| `evaluate` | 0 | scores such a table as the leaderboard is scored and gives the rank it would take among the 15 predictors |
| `compare` | 0 | paired bootstrap of such a table against the 15 predictors, Holm-adjusted |
| `make_task` | 0 | splits a user's binary labels into five protein-family folds with the procedure and the seed that built the benchmark's folds, and writes them as `get_fold` does; `check_predictions` and `evaluate` then work on the new task |
| `score_labels` | 0 | scores the test-fold predictions of the 15 predictors on one benchmark task against a user's labels, with paired tests against the best |
| `layer_status` | 0 | which tools can run, the downloads each layer-1 tool still needs and the tables its first call builds |
| `fetch_external` | 0 | downloads external data to the data root |
| `annotate_genes` | 1 | per gene: the four hallmarks, gene age, paralogues and literature count; the DepMap 26Q1 dependency; or the IMPC knockout call and lethal stage, each next to the genes of a task |
| `enrichment` | 1 | GO biological processes and CORUM complexes over-represented in a gene list, hypergeometric test with Benjamini-Hochberg adjustment |
| `evaluate_by_strata` | 1 | scores a table of test-fold predictions within the manuscript's groups of genes (hallmark count, literature quartile, family size, PPI degree, expression) next to the 15 predictors |
| `core_model` | 1 | trains the core feature set model on a benchmark task or a task made by `make_task`, writes its test-fold predictions for `evaluate`, and predicts other genes |

The method cards and both READMEs are served as resources, and the prompts `evaluate_new_predictor` and `explain_gene` open the two common requests of layer 0.

## Installing the server

The quick start is in the [repository README](https://github.com/tubic/EGBench#use-egbench-from-an-ai-agent). This section gives the details behind it.

| | Linux x86_64 | macOS | Windows |
|---|---|---|---|
| layers 0 and 1 | tested | layer 0 tested; layer 1 reports what it needs, its downloads and builds were not run | through WSL only: the server imports `fcntl`, which Windows lacks |

Layer 1 needs room under the data root for 3.6 GB of downloads and the tables built from them, and 9.4 GB of memory for `core_model`. `core_model` fits the five folds of a task one after another, each with the BLAS and OpenMP threads that `OMP_NUM_THREADS` allows, all cores when it is unset; on a shared machine the variable goes into the environment of the server's registration.

### As a package

The repository is the Python package `egbench`: its command `egbench-mcp` starts the server, and its extra `mcp` installs the server's packages at the versions of `requirements.txt`. uv installs it in an environment of its own and starts the server. The package can come from the GitHub repository or from a clone:

```
uvx --from "egbench[mcp] @ git+https://github.com/tubic/EGBench" egbench-mcp
uvx --from "egbench[mcp] @ file:///path/to/clone" egbench-mcp
```

Measured on macOS, the first start with an empty uv cache took 188, 255 and 356 seconds in three runs (the 42 packages are downloaded, the package is fetched and built and the packages are imported once) and 66 seconds with the packages cached; later starts take 6 to 8 seconds. uv took Python 3.13.15, with which `evaluate` gave the same values as with Python 3.11.17. uv keeps the environment it built for a path; after the clone there changes, `uv cache clean` makes the next start build the package again (with uv 0.13.1, `--refresh` and `uv cache clean egbench` did not). Downloads and the tables built from them go to the data root, `EGBENCH_DATA_ROOT` or `~/egbench-data` when it is unset. Paths passed to the tools are given in full; a relative path is read from the package's copy of the repository, where `agent/examples/` lies.

A client gives up before the first start ends, so the first start is made in a terminal before the server is registered. Claude Code 2.1.287 reported `Failed to connect — Request timed out` after 35 seconds on a first start from an empty cache, and `Connected` after the start in a terminal. Raising the limit also works: `MCP_TIMEOUT=600000 claude mcp list` connected after 4 minutes 56 seconds from an empty cache.

### From a clone

The environment is Python 3.11.17 with the packages of `requirements.txt`, installed from the repository root:

```
conda create -p <env> python=3.11.17
<env>/bin/pip install -r agent/requirements.txt
<env>/bin/python -c "import numpy, pandas, pyarrow, scipy.stats, sklearn.metrics, mcp"
<env>/bin/python agent/server.py
```

The third command imports the packages once. In a new environment on macOS the first import took 47 seconds and later ones 1 second, and a first start that slow can exceed a client's time limit for starting a server. uv 0.12.10 offers Python up to 3.11.16; an environment made with `uv venv -p 3.11.16 <env>` and `uv pip install -p <env>/bin/python -r agent/requirements.txt` gave the same values of `evaluate`, and the MCP Python SDK client started `agent/server.py` of such an environment and listed its 15 tools in 7.7 seconds. The data root is `EGBENCH_DATA_ROOT`, or `data/` in the repository when it is unset; relative paths passed to the tools are read from the repository root.

### Registering the server in a client

Steps 2 and 3 of the quick start register the package in Claude Code and Claude Desktop. Any MCP client that starts a server over standard input and output takes the same command, arguments and environment. A server run from a clone is registered with the full paths of the interpreter and of `agent/server.py`, since clients start servers from a working directory of their own.

The clients used here: Claude Code 2.1.287 (registration and connection checked with `claude mcp list`; no tool was called from inside it), the MCP Python SDK client, and `client.py` with Qwen3.8-27B (14 of the 15 tools in the 11 requests; `fetch_external` was not called). The Claude Desktop entry was not opened in the Claude Desktop application; its command, arguments and environment were run as a subprocess. Claude Code sets its time limits for starting a server and for one tool call through `MCP_TIMEOUT` and `MCP_TOOL_TIMEOUT`, in milliseconds. The first call of `core_model` on a task takes 3 to 8 minutes, and Claude Code's default limit for one call was not measured here.

### Steps the agent passes on to the user

Once the server runs, a tool that lacks something says what it lacks, and the server's instructions ask the agent to pass on the steps that only the user can take:

- saving the three DepMap 26Q1 files, which only a browser can download, at the path `fetch_external` gives;
- allowing for the minutes that the first call of a layer-1 tool and of `core_model` take, which the agent reports before the call; `python agent/core_model.py train all --workers 5` trains the core feature set model on every task ahead of use.

The downloads of `fetch_external` run in the server and stop when the client stops it. Before the server runs, none of this reaches an agent, so the installation, the registration and the time limits above are the user's.

## External data

`python agent/external.py status` lists the items of `external.json` and whether each is under the data root; `python agent/external.py fetch <id> ...` downloads items, eight at a time. The `fetch_external` tool does the same for an agent.

- **Unversioned sources.** NCBI rebuilds gene2pubmed every day, CORUM's address serves its latest release, which was 5.3 (2026-04-14) for the manuscript, and the JensenLab COMPARTMENTS file was replaced on 2026-10-05. The tools use the current files, so the literature counts and the complexes differ slightly from the manuscript's, and EPGAT's localisation features differ (Inputs of the methods). The other sources are fetched from versioned addresses: Gene Ontology release archives, Ensembl and GENCODE releases, the HGNC monthly archive, IMPC release 24.0, BioGRID 5.0.259, the Mammalian Phenotype Ontology release of 2026-08-25 and Zenodo records.
- **Downloads by hand.** The DepMap portal admits only browsers, so the three DepMap 26Q1 files are downloaded by hand from the page the tool names and saved at the path it gives; a second call confirms they are in place. Only the dependency aspect of `annotate_genes` reads them.

Four tables that the methods received from the previous benchmark are rebuilt from public sources: the gene reference (GENCODE v50 and HGNC), the TCGA expression matrix (33 archives of Zenodo record 7885656), the BioGRID physical interactions and the orthologues in 194 species (Ensembl Compara 115).

Licences: CORUM CC BY-NC 4.0; HELP's Zenodo record 12597679 GPL-3.0 or later; BioGRID MIT; HGNC CC0 1.0; DepMap, the Gene Ontology, IMPC, the Mammalian Phenotype Ontology and the TCGA record CC BY 4.0; NCBI, Ensembl and GENCODE place no restrictions on their data. `external.json` gives the licence of every item.

## Rerunning the methods and the analyses

These commands run in a clone of the repository, with the environment of `requirements.txt`, on Linux x86_64 with conda, or uv, and an NVIDIA GPU for 11 of the methods. On macOS only the environments `cpu` and `node2vec` install; Windows needs WSL.

```
python agent/environments.py install <environment | method>
python agent/inputs.py build <method>
python agent/runs.py train <method> <task> <fold>
python agent/jobs.py status <job>
python agent/runs.py check <method> <task> <fold>
python agent/runs.py genes <method> <task> <genes>
python agent/runs.py report
python agent/runs.py analysis <analysis | all>
```

`environments.py install` creates one of seven environments from its lock file, and `inputs.py build` downloads and builds every input file of a method. `runs.py train` trains one fold of a method on a task as a background job, on the GPU with the most free memory, and `jobs.py` lists, follows and cancels the jobs. `runs.py check` scores a trained fold as the leaderboard does and compares its test predictions gene by gene with the published fold; `runs.py genes` scores genes with the five trained folds of a method and averages them; `runs.py report` reruns `egbench report` on the trained folds and `runs.py analysis` the scripts of one analysis, or of all in the order of the repository README, each compared table by table with the published tables. Each command of `runs.py` prints first what it still needs, with sizes, minutes and GPU memory, and the command to run next; `environments.py status` and `inputs.py status` list the environments and the method inputs in place.

| Step | What it needs |
|---|---|
| environments and inputs | one of seven environments (0.7 to 7.2 GB, about 30 GB for all); the method's downloads (0.2 to 13.3 GB, 26.5 GB for all 15) and the builds that follow them (1 to 85 minutes), with 19 GB of GPU memory for BiNGO's builds and 22 GB for EssTFNet's |
| one fold | a GPU with 1 to 14 GB for 11 methods; 0.4 to 75 minutes, eDNN-EG about six hours |
| scoring genes with a method | the five folds of the method trained on the task |
| report and analyses | the 375 folds of the 15 methods on five tasks for the report, the 450 with DepMap 23Q4 for the analyses; the analyses' downloads and pretrained weights; a GPU for two analyses |

### Environments of the methods

The manuscript ran every method in an environment of its own (the Environments section of the repository README). The commands use seven, each locked in `environments/<name>.txt` and described in `environments.json`:

| Environment | Python | Packages | Serves | Install on the server | Size |
|---|---|---|---|---|---|
| `torch` | 3.11.16 | 62 | the nine other neural methods; `same_input_mlp` | 9.7 min | 6.4 GB |
| `bingo` | 3.12.3 | 58 | BiNGO; the ESM-2 scripts of `foundation_model_probes` | 11.3 min | 6.4 GB |
| `proteomelm` | 3.12.3 | 86 | ProteomeLM-Ess | 10.6 min | 6.7 GB |
| `cpu` | 3.11.16 | 20 | HELP, Pheg, XGEP-XGB, `egbench report` and the other analyses | 2.2 min | 0.8 GB |
| `mahi` | 3.11.16 | 28 | Mahi | 2.5 min | 1.5 GB |
| `node2vec` | 3.11.15 | 12 | the node2vec embedding of DeepHE and DeEPsnap | 1.9 min | 0.7 GB |
| `evo2` | 3.12.3 | 53 | the Evo 2 embeddings of EssTFNet | 26.5 min | 7.2 GB |

BiNGO and ProteomeLM-Ess keep environments of their own because fair-esm and esm 3 both install a module named `esm`. Mahi and XGEP-XGB need different builds of XGBoost 3.2.0: their original environments installed it from PyPI for Mahi and from conda-forge for XGEP-XGB, and the two builds draw different row and column subsamples, so `mahi` holds the PyPI wheel and `cpu` the CPU build that matches conda-forge's. HELP's original environment had XGBoost 3.0.5, which its third-party code imports but never calls. The `flash-attn` of `evo2` is the 2.7.4.post1 wheel; the 2.8 wheels reference the new C++ ABI of libtorch and fail to import with PyTorch 2.6.0 from PyPI. `egbench` is not installed in any of them: the jobs set `PYTHONPATH=<repository>/src`. `environments.py install` creates the Python version with conda (`CONDA_EXE`, or `conda` on the path) and installs the lock file with pip; without conda it stops with the commands for uv or venv, and on a system other than Linux it stops for the five environments whose lock files pin CUDA packages. Environments go to `EGBENCH_ENV_ROOT`, or `envs/` under the data root.

### Inputs of the methods

`methods.json` lists the 113 files the 15 methods read under `methods/<method>/feature/source/`. Each is a download or a build of `external.json`, or a file of the repository; it also gives the third-party code a method fetches, and the GPU memory, threads and minutes of a fold. `inputs.py build <method>` fetches the downloads the method lacks, runs the builds in order, each in the environment it names and on a GPU when it needs one, fetches the third-party code of HELP and EssTFNet from GitHub, and links the files into the method's directory, so the data stay under the data root. BiNGO refuses a graph directory reached through a link, so its graph cache is moved into the method's directory and linked back. A build that several methods share runs once.

What each method needs from an empty data root, measured on the server (the downloads of different methods overlap):

| Method | Downloads (MB) | Builds (min) | GPU for the builds (GB) |
|---|---|---|---|
| BiNGO | 2,941.9 | 70.6 | 19 |
| DeepCellEss | 337.3 | 2.6 | |
| DeepHE | 523.9 | 15.1 | |
| DeeplyEssential | 337.3 | 2.9 | |
| DeEPsnap | 578.5 | 15.1 | |
| eDNN-EG | 337.3 | 5.8 | |
| EPGAT | 2,428.1 | 3.0 | |
| EssSubgraph | 2,018.4 | 2.6 | |
| EssTFNet | 13,343.9 | 85.1 | 22 |
| GCNN-SFM | 337.3 | 5.8 | |
| HELP | 441.1 | 1.4 | |
| Mahi | 4,932.4 | 1.0 | |
| Pheg | 337.3 | 3.8 | |
| ProteomeLM-Ess | 3,320.8 | 9.9 | 8 |
| XGEP-XGB | 237.5 | 1.1 | |

The builds reproduce the inputs the methods were trained on (Checks), with three exceptions:

- **The Evo 2 embeddings of EssTFNet.** Built from the public Evo 2 7B base checkpoint with evo2 0.6.0, they differ from the manuscript's in 81,918 of 81,920 values on 20 sampled genes. The environment that computed the manuscript's embeddings is not recorded, and the cause was not found. EssTFNet trained on rebuilt embeddings is a different model from the published one; its published test-fold predictions stay in `predictions/`.
- **COMPARTMENTS.** The JensenLab file has no versioned address and was replaced on 2026-10-05, so the manuscript's version cannot be downloaded. The current file, which `inputs.py build` uses, changes EPGAT's localisation features for 14,860 of the 19,158 genes.
- **Licences.** Mahi's embeddings are under Princeton's research-only licence, which does not allow redistribution; the XGEP repository states no licence. Downloads from the Hugging Face Hub go to the mirror named by `HF_ENDPOINT` when it is set; where huggingface.co cannot be reached, the error of a download says so.

### Background jobs

A job is a directory `<data root>/agent/jobs/<id>/` holding `job.json` and `log.txt`. It runs in a session of its own, so it survives the command and the terminal that started it, and `jobs.py status` reads its state from disk. A job that needs a GPU is placed on the card with the most free memory, counting the memory of the jobs started on that card in the last ten minutes, which may not be allocated yet. Jobs run with eight threads for numpy, BLAS and OpenMP, or with the threads that reproduce a method's folds (two for DeepCellEss, one for eDNN-EG and Pheg); the analyses with the cores left free divided among their worker processes. `runs.py report` and `runs.py analysis` copy the published tables to `<data root>/agent/published/` before their first run, and compare each table the rerun writes with its copy, to a relative tolerance of 1e-6.

## Checks

Each check compares what the tools and the commands compute from public data with what the manuscript computed and writes its comparisons to `results/`.

- `python agent/check_reproduction.py --workers 16` rescores the 15 predictors through `evaluate` and `compare` on the six leaderboard tasks (`reproduction.csv`): the 90 primary metrics and the 18 paired tests agree to the precision the tools report.
- `python agent/labels.py check` rebuilds the folds of the six binary tasks from their labels with `make_task` (`labels_check.csv`): every gene takes the role it has in the published folds. `evaluate` on a task made by `make_task` and `score_labels` on DEG's own labels give the leaderboard's values.
- `python agent/sources.py check`, `python agent/annotation.py check` and `python agent/core_model.py check` compare the rebuilt tables, the annotation and the core feature set model with the methods' inputs and the analyses' caches, which are not distributed, so their arguments point to a copy of the manuscript's working directory (`sources_check.csv`, `annotation_check.csv`, `core_model_check.csv`). The gene reference, the TCGA matrix, the orthologues and the static features of the core feature set are identical to the methods' inputs; the BioGRID edges have the same 935,203 gene pairs. Recomputed from the agent's tables, `four_properties.csv`, `go_enrichment.csv` and the other hallmark tables agree within 1e-15. The core feature set model reproduces the manuscript's test-fold predictions bit for bit on its 30 folds, and the six rows of `core_fixed.csv` exactly.
- `environments_check.csv`: DEG fold 0 of every method in its environment of the seven, with the original inputs. The 13 methods trained in full give the published test predictions bit for bit (Pheg within 2.2e-16). BiNGO and eDNN-EG, whose training takes over 30 minutes, were started, stopped once training ran, and scored with their published weights (differences 1.2e-7 and 0).
- `sequences_check.csv`, `method_inputs_check.csv` and `embeddings_check.csv` (`python agent/<module>.py check --methods <copy of methods/>` in the `cpu` environment; `method_inputs.py check --compartments <file>` in `torch`) compare every built input with the manuscript's. The FASTA files, Pheg's matrix, EPGAT's static table, XGEP's matrix, the tables of HELP and Mahi, the node2vec embedding and the ESM-C and ProteomeLM features are equal byte for byte or value for value; the 89 sequence features differ on 36 genes by at most 1.1e-16 (the codon adaptation index through the C library); BiNGO's graph cache agrees on 100 sampled genes, edges exactly and embeddings within 1e-4; the Evo 2 embeddings differ.
- `layer2_check.csv`: the functions behind `inputs.py build` and `runs.py train`, `check` and `genes`, in a fresh copy whose data root held only the downloads. `inputs.py build` placed 104 files of 14 methods, 89 byte for byte equal to the manuscript's and 15 within the tolerances above; BiNGO's graph cache was started and stopped. `runs.py train` and `check` on DEG fold 0 gave the published test predictions bit for bit for 13 methods (Pheg within 2.2e-16); eDNN-EG was started and stopped. `runs.py genes` with HELP and DeepHE on DEG gave each labelled gene its published score in its test fold (DeepHE within one float32 step, 6e-8).
- `layer4_check.csv`: the functions behind `runs.py report` and `runs.py analysis`, with the trained folds. The 14 report tables and 94 analysis tables equal the published ones exactly. The ESM-2 3B embedding, the embedding probes, the same-input networks and the forward selection, over 30 minutes each, were started and stopped, so their five tables were not compared.

## Open-weight models

`client.py` drives the server with a model behind an OpenAI-compatible endpoint, such as vLLM, and logs the time and tokens of every model and tool call. Run it from a clone, in the environment of `requirements.txt`:

```
export EGBENCH_LLM_API_KEY=<key>
python agent/client.py --server-command "python agent/server.py" --base-url <endpoint> --model <name> \
  --prompt evaluate_new_predictor --arg task=deg --arg predictions_path=agent/examples/deg_logistic.csv --log run.json
```

With `--task "<request>"` in place of `--prompt`, the model gets a request in free text; `--python-dir <directory>` also gives it a function that runs Python code in that directory, with which it can train a model of its own from the folds that `get_fold` writes.

`python agent/run_scenarios.py --base-url <endpoint> --model <name>` runs the eleven requests of `scenarios.json` once each and writes their logs and a summary table to `results/`; each request lists the facts its answer must contain. The requests cover four uses: predicting a user's data with the published predictors and the core feature set model, training the core feature set model on a user's own labels, training a model that the user describes in natural language on the benchmark folds, and analysing genes further. For the two requests marked `"python": true` the client also offers a function that runs Python code in `<data root>/agent/work/python/<id>`, as a coding agent has when it trains a model of its own; the model writes the code, the tools check and score the table it saves. `--recount` counts the facts again in existing logs, for a list of facts corrected after a run. The results in the repository were obtained with the BF16 weights of Qwen3.8-27B served by vLLM 0.31.0, with the sampling the model's developers recommend for its thinking mode.

## Files

| File | Content |
|---|---|
| `server.py`, `launch.py` | the MCP server: tools, method cards and READMEs as resources, two prompts; the command `egbench-mcp` of the package, which `hatch_build.py` in the repository root fills with the files the server reads |
| `benchmark.py` | tasks, folds, genes, the published predictions and submission tables, shared by the tools |
| `external.py`, `external.json` | the data root, the acquisition manifest, downloads and the tables built from them |
| `labels.py` | `make_task`, `score_labels` and the scoring of tasks made by `make_task` |
| `sources.py` | the gene reference and the TCGA expression matrix built from public downloads |
| `annotation.py` | the BioGRID edges, the annotation tables, `annotate_genes`, `enrichment` and `evaluate_by_strata` |
| `core_model.py` | the orthologues, the static features, the per-fold TCGA components and `core_model` |
| `predictions/` | test-fold predictions of the 15 methods, one table per cohort, built by `build_assets.py predictions` |
| `examples/` | a logistic-regression predictor on DEG and a copy of it with errors (`build_assets.py example`), and the knockout calls of the Mouse task as a user's own labels (`build_assets.py labels`) |
| `client.py`, `run_scenarios.py`, `scenarios.json` | the open-weight model client and the scenario test |
| `requirements.txt` | the CPU environment of the server and of the commands |
| `environments.py`, `environments.json`, `environments/` | the seven environments of the methods and the analyses, and their lock files |
| `inputs.py`, `methods.json` | the input files of each method, the item that makes each, and the resources of a fold |
| `sequences.py`, `pheg_features.py` | the sequence sources, the FASTA files of the methods, the 89 sequence features and Pheg's Z-curve features |
| `method_inputs.py` | EPGAT's static table, XGEP's matrix, DeEPsnap's annotations, the tables of HELP and Mahi, the node2vec embedding |
| `embeddings.py` | BiNGO's graph cache, the ESM-C and ProteomeLM features, EssTFNet's candidate features and Evo 2 embeddings |
| `runs.py`, `jobs.py`, `analyses.json` | training, checking and predicting with a method, rebuilding the report and rerunning the analyses as background jobs, and what each analysis needs |
| `candidates/` | the candidate genes of each task, which the method code reads from `data/candidates/` |
| `results/` | the records of the checks and of the scenario test |
