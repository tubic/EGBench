"""Gene features tied to essentiality, the nine gene age layers and binomial share intervals.

Imported by essential_gene_hallmarks and research_attention; it runs nothing by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from egbench.data import PROJECT_ROOT, read_cohort_labels

from common.feature_blocks import HELP_BIO_NAMES, block_array, block_names, cohort_genes
from common.stats import METHODS_DIRECTORY

HALLMARKS = PROJECT_ROOT / "analysis" / "essential_gene_hallmarks"
WORK = HALLMARKS / "work"
RESULTS = HALLMARKS / "results"
EXPRESSION = PROJECT_ROOT / "data" / "source" / "expression" / "tcga_gene_by_sample.parquet"
STATIC = METHODS_DIRECTORY / "epgat" / "feature" / "source" / "generated" / "epgat_static.parquet"
TASKS: tuple[str, ...] = ("deg", "depmap")
TASK_NAMES = {"deg": "DEG", "depmap": "DepMap"}
CONTINUOUS = "continuous"
BINARY = "binary"
SPECIES_PREFIX = "ortholog__"


def feature_values_path(task: str):
    return WORK / f"feature_values_{task}.csv"


def gene_properties_path(task: str):
    return WORK / f"gene_properties_{task}.csv"


@dataclass(frozen=True)
class Feature:
    key: str
    name: str
    group: str
    kind: str
    source: str


FEATURES: tuple[Feature, ...] = (
    Feature("orth_count", "HELP orth_count (number of orthologues)", "orthology", CONTINUOUS, "HELP bio__14"),
    Feature("yeast_ortholog", "Has a Saccharomyces cerevisiae orthologue", "orthology", BINARY,
            "EPGAT ortholog__saccharomyces_cerevisiae"),
    Feature("fly_ortholog", "Has a Drosophila melanogaster orthologue", "orthology", BINARY,
            "EPGAT ortholog__drosophila_melanogaster"),
    Feature("hagfish_ortholog", "Has an Eptatretus burgeri orthologue", "orthology", BINARY,
            "EPGAT ortholog__eptatretus_burgeri"),
    Feature("ortholog_species", "Number of species with an orthologue (of 194)", "orthology", CONTINUOUS,
            "sum of the 194 EPGAT ortholog__ columns"),
    Feature("corum_count", "Number of CORUM complexes", "complex", CONTINUOUS, "DeEPsnap corum_membership_count"),
    Feature("corum_any", "Member of at least one CORUM complex", "complex", BINARY,
            "DeEPsnap corum_membership_count at least 1"),
    Feature("cc_protein_complex", "CCcfs protein-containing complex confidence", "complex", CONTINUOUS,
            "HELP CCcfs column protein-containing complex"),
    Feature("biogrid", "HELP BIOGRID (interaction records)", "interaction and pathway", CONTINUOUS, "HELP bio__9"),
    Feature("ppi_degree", "BioGRID physical interaction degree", "interaction and pathway", CONTINUOUS,
            "EPGAT ppi_induced_degree on the whole network"),
    Feature("reactome", "HELP REACTOME (pathway records)", "interaction and pathway", CONTINUOUS, "HELP bio__11"),
    Feature("gtex_median", "HELP Gtex_median_expr (GTEx median expression)", "expression", CONTINUOUS,
            "HELP bio__4"),
    Feature("tcga_median", "TCGA median expression", "expression", CONTINUOUS,
            "median over TCGA samples of log2(x + 1)"),
)
FEATURE_BY_KEY = {feature.key: feature for feature in FEATURES}
CONTINUOUS_KEYS: tuple[str, ...] = tuple(f.key for f in FEATURES if f.kind == CONTINUOUS)


def _named_column(cohort: str, block: str, name: str) -> np.ndarray:
    names = block_names(cohort, 0, block)
    positions = [i for i, n in enumerate(names) if n == name]
    if len(positions) != 1:
        raise ValueError(f"{block} of {cohort}: {len(positions)} columns are named {name!r}")
    return np.asarray(block_array(cohort, 0, block)[:, positions[0]], dtype="float64")


def expression_median(genes: pd.Index) -> np.ndarray:
    table = pq.read_table(EXPRESSION)
    keys = pd.Index([str(v) for v in table.column("gene_key").to_pylist()])
    samples = [name for name in table.column_names if name != "gene_key"]
    rows = keys.get_indexer(genes)
    if (rows < 0).any():
        raise ValueError(f"{int((rows < 0).sum())} genes have no TCGA expression row")
    values = np.column_stack([table.column(name).to_numpy() for name in samples])[rows].astype("float64")
    if np.isnan(values).any() or (values < 0).any():
        raise ValueError("the TCGA matrix holds a missing or negative value")
    return np.median(np.log2(values + 1.0), axis=1)


def feature_frame(cohort: str) -> pd.DataFrame:
    genes = cohort_genes(cohort)
    labels = read_cohort_labels(cohort).set_index("gene_key")["label"]
    labels.index = labels.index.astype(str)
    frame = pd.DataFrame(index=genes)
    frame["label"] = labels.reindex(genes).to_numpy("float64")
    if frame["label"].isna().any():
        raise ValueError(f"{cohort}: a cached gene has no label")
    bio = np.asarray(block_array(cohort, 0, "help_bio"), dtype="float64")
    frame["orth_count"] = bio[:, HELP_BIO_NAMES.index("orth_count")]
    frame["yeast_ortholog"] = _named_column(cohort, "orthology", "ortholog__saccharomyces_cerevisiae")
    frame["fly_ortholog"] = _named_column(cohort, "orthology", "ortholog__drosophila_melanogaster")
    frame["hagfish_ortholog"] = _named_column(cohort, "orthology", "ortholog__eptatretus_burgeri")
    orthology = np.asarray(block_array(cohort, 0, "orthology"), dtype="float64")
    species = [i for i, n in enumerate(block_names(cohort, 0, "orthology")) if n.startswith(SPECIES_PREFIX)]
    if len(species) != 194:
        raise ValueError(f"{cohort}: {len(species)} species columns")
    frame["ortholog_species"] = orthology[:, species].sum(axis=1)
    corum = np.asarray(block_array(cohort, 0, "corum"), dtype="float64")
    frame["corum_count"] = corum[:, 0]
    frame["corum_any"] = (corum[:, 0] >= 1).astype("float64")
    frame["cc_protein_complex"] = _named_column(cohort, "cccfs", "protein-containing complex")
    frame["biogrid"] = bio[:, HELP_BIO_NAMES.index("BIOGRID")]
    frame["ppi_degree"] = np.asarray(block_array(cohort, 0, "ppi_degree"), dtype="float64")[:, 0]
    frame["reactome"] = bio[:, HELP_BIO_NAMES.index("REACTOME")]
    frame["gtex_median"] = bio[:, HELP_BIO_NAMES.index("Gtex_median_expr")]
    frame["tcga_median"] = expression_median(genes)
    for feature in FEATURES:
        if feature.kind == BINARY and not np.isin(frame[feature.key], (0.0, 1.0)).all():
            raise ValueError(f"{cohort}: {feature.key} is not binary")
    frame.index.name = "gene_key"
    return frame


@dataclass(frozen=True)
class Layer:
    index: int
    name: str
    members: str
    species: tuple[str, ...]


LAYERS: tuple[Layer, ...] = (
    Layer(1, "Opisthokonta", "Saccharomyces cerevisiae", ("saccharomyces_cerevisiae",)),
    Layer(2, "Bilateria", "Drosophila melanogaster and Caenorhabditis elegans",
          ("drosophila_melanogaster", "caenorhabditis_elegans")),
    Layer(3, "Olfactores", "two ascidians", ("ciona_intestinalis", "ciona_savignyi")),
    Layer(4, "Vertebrata", "hagfish and lamprey", ("eptatretus_burgeri", "petromyzon_marinus")),
    Layer(5, "Gnathostomata", "elephant shark", ("callorhinchus_milii",)),
    Layer(6, "Bony fishes", "ray-finned fishes and coelacanth", (
        "acanthochromis_polyacanthus", "amphilophus_citrinellus", "amphiprion_ocellaris", "amphiprion_percula",
        "anabas_testudineus", "astatotilapia_calliptera", "astyanax_mexicanus", "betta_splendens",
        "carassius_auratus", "clupea_harengus", "cottoperca_gobio", "cyclopterus_lumpus", "cynoglossus_semilaevis",
        "cyprinodon_variegatus", "cyprinus_carpio_carpio", "danio_rerio", "denticeps_clupeoides",
        "dicentrarchus_labrax", "electrophorus_electricus", "erpetoichthys_calabaricus", "esox_lucius",
        "fundulus_heteroclitus", "gadus_morhua", "gasterosteus_aculeatus", "haplochromis_burtoni",
        "hippocampus_comes", "hucho_hucho", "ictalurus_punctatus", "kryptolebias_marmoratus", "labrus_bergylta",
        "larimichthys_crocea", "lates_calcarifer", "latimeria_chalumnae", "lepisosteus_oculatus",
        "mastacembelus_armatus", "maylandia_zebra", "myripristis_murdjan", "neolamprologus_brichardi",
        "nothobranchius_furzeri", "oncorhynchus_kisutch", "oncorhynchus_mykiss", "oncorhynchus_tshawytscha",
        "oreochromis_niloticus", "oryzias_javanicus", "oryzias_latipes", "oryzias_melastigma", "oryzias_sinensis",
        "paramormyrops_kingsleyae", "poecilia_formosa", "poecilia_latipinna", "poecilia_reticulata",
        "pundamilia_nyererei", "pygocentrus_nattereri", "salmo_salar", "salmo_trutta", "sander_lucioperca",
        "scleropages_formosus", "scophthalmus_maximus", "seriola_dumerili", "seriola_lalandi_dorsalis",
        "sparus_aurata", "stegastes_partitus", "takifugu_rubripes", "tetraodon_nigroviridis",
        "xiphophorus_couchianus", "xiphophorus_maculatus",
    )),
    Layer(7, "Tetrapods", "amphibians, reptiles and birds", (
        "anas_platyrhynchos_platyrhynchos", "anolis_carolinensis", "anser_brachyrhynchus",
        "aquila_chrysaetos_chrysaetos", "chelonoidis_abingdonii", "chrysemys_picta_bellii", "coturnix_japonica",
        "crocodylus_porosus", "ficedula_albicollis", "gallus_gallus", "geospiza_fortis", "gopherus_evgoodei",
        "laticauda_laticaudata", "leptobrachium_leishanense", "meleagris_gallopavo", "naja_naja",
        "notechis_scutatus", "parus_major", "pelodiscus_sinensis", "podarcis_muralis", "pseudonaja_textilis",
        "salvator_merianae", "serinus_canaria", "sphenodon_punctatus", "strigops_habroptila",
        "struthio_camelus_australis", "taeniopygia_guttata", "terrapene_carolina_triunguis", "xenopus_tropicalis",
    )),
    Layer(8, "Platypus and marsupials", "platypus and marsupials", (
        "ornithorhynchus_anatinus", "monodelphis_domestica", "notamacropus_eugenii", "phascolarctos_cinereus",
        "sarcophilus_harrisii", "vombatus_ursinus",
    )),
    Layer(9, "Eutheria", "placental mammals", (
        "ailuropoda_melanoleuca", "aotus_nancymaae", "balaenoptera_musculus", "bison_bison_bison", "bos_grunniens",
        "bos_indicus_hybrid", "bos_mutus", "bos_taurus", "callithrix_jacchus", "camelus_dromedarius",
        "canis_lupus_dingo", "canis_lupus_familiaris", "capra_hircus", "carlito_syrichta", "catagonus_wagneri",
        "cavia_porcellus", "cebus_imitator", "cercocebus_atys", "cervus_hanglu_yarkandensis", "chinchilla_lanigera",
        "chlorocebus_sabaeus", "choloepus_hoffmanni", "dasypus_novemcinctus", "delphinapterus_leucas",
        "dipodomys_ordii", "echinops_telfairi", "equus_asinus", "erinaceus_europaeus", "felis_catus",
        "gorilla_gorilla", "heterocephalus_glaber_female", "ictidomys_tridecemlineatus", "jaculus_jaculus",
        "loxodonta_africana", "macaca_fascicularis", "macaca_mulatta", "macaca_nemestrina", "mandrillus_leucophaeus",
        "marmota_marmota_marmota", "mesocricetus_auratus", "microcebus_murinus", "microtus_ochrogaster",
        "monodon_monoceros", "moschus_moschiferus", "mus_caroli", "mus_pahari", "mus_spicilegus", "mus_spretus",
        "mustela_putorius_furo", "myotis_lucifugus", "nannospalax_galili", "neovison_vison", "nomascus_leucogenys",
        "ochotona_princeps", "octodon_degus", "oryctolagus_cuniculus", "otolemur_garnettii", "pan_paniscus",
        "pan_troglodytes", "panthera_leo", "panthera_pardus", "panthera_tigris_altaica", "papio_anubis",
        "peromyscus_maniculatus_bairdii", "phocoena_sinus", "physeter_catodon", "pongo_abelii", "procavia_capensis",
        "prolemur_simus", "propithecus_coquereli", "pteropus_vampyrus", "rattus_norvegicus",
        "rhinolophus_ferrumequinum", "rhinopithecus_bieti", "rhinopithecus_roxellana",
        "saimiri_boliviensis_boliviensis", "sciurus_vulgaris", "sorex_araneus", "tupaia_belangeri",
        "tursiops_truncatus", "urocitellus_parryii", "ursus_americanus", "ursus_maritimus", "vicugna_pacos",
        "vulpes_vulpes",
    )),
)
LAYER_BY_INDEX = {layer.index: layer for layer in LAYERS}
YOUNGEST = 9


def species_layers(columns: list[str]) -> pd.DataFrame:
    species = [c[len(SPECIES_PREFIX):] for c in columns if c.startswith(SPECIES_PREFIX)]
    if len(species) != 194 or len(set(species)) != 194:
        raise ValueError(f"the orthology matrix holds {len(species)} species columns, 194 are expected")
    listed = [name for layer in LAYERS for name in layer.species]
    missing, extra = sorted(set(species) - set(listed)), sorted(set(listed) - set(species))
    if missing or extra or len(listed) != len(set(listed)):
        raise ValueError(f"species without a layer: {missing}; layers name species not in the matrix: {extra}")
    layer_of = {name: layer.index for layer in LAYERS for name in layer.species}
    rows = [{"species": name, "column": SPECIES_PREFIX + name, "layer": layer_of[name],
             "layer_name": LAYER_BY_INDEX[layer_of[name]].name} for name in species]
    return pd.DataFrame(rows).sort_values(["layer", "species"], kind="stable").reset_index(drop=True)


def gene_age(genes: pd.Index) -> tuple[pd.Series, pd.DataFrame]:
    static = pd.read_parquet(STATIC)
    static["gene_key"] = static["gene_key"].astype(str)
    static = static.set_index("gene_key")
    rows = static.index.get_indexer(genes)
    if (rows < 0).any():
        raise ValueError(f"{int((rows < 0).sum())} genes have no row in {STATIC.name}")
    table = species_layers(list(static.columns))
    matrix = static.iloc[rows][list(table["column"])].astype("float64")
    if not np.isin(matrix.to_numpy(), (0.0, 1.0)).all():
        raise ValueError("an orthology column holds a value other than 0 and 1")
    age = pd.Series(YOUNGEST, index=genes, dtype="int64", name="age_layer")
    for layer in sorted(LAYERS, key=lambda item: -item.index):
        if layer.index == YOUNGEST:
            continue
        columns = [SPECIES_PREFIX + name for name in layer.species]
        present = matrix[columns].to_numpy().max(axis=1) == 1.0
        age[present] = layer.index
    matrix.index = genes
    return age, matrix


def _share_and_variance(labels: np.ndarray, member: np.ndarray) -> tuple[float, float]:
    inside = labels[member.astype(bool)]
    if inside.size == 0:
        raise ValueError("the group has no genes")
    share = float(inside.mean())
    return share, share * (1.0 - share) / inside.size


def share_interval(labels: np.ndarray, member: np.ndarray) -> tuple[float, float]:
    share, variance = _share_and_variance(labels, member)
    return share, float(1.96 * np.sqrt(variance))


def difference_interval(labels: np.ndarray, group_a: np.ndarray, group_b: np.ndarray) -> tuple[float, float]:
    if (group_a & group_b).any():
        raise ValueError("the two groups share genes")
    share_a, variance_a = _share_and_variance(labels, group_a)
    share_b, variance_b = _share_and_variance(labels, group_b)
    return share_a - share_b, float(1.96 * np.sqrt(variance_a + variance_b))


def write_table(frame: pd.DataFrame, name: str, directory=RESULTS) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / f"{name}.csv", index=False)
