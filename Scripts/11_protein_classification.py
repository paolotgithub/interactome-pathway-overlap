"""
11_protein_classification.py
-----------------------------
Classifies promiscuous proteins into a five-class taxonomy using UMAP
embedding followed by HDBSCAN clustering.

Pipeline:
  1. Build a 7-feature matrix per protein (see below)
  2. Log-transform skewed features, RobustScaler normalisation
  3. UMAP to 10D (for clustering) and 2D (for visualisation)
  4. HDBSCAN on the 10D embedding (min_cluster_size=80)
  5. Greedy taxonomy label assignment from cluster centroids

Feature space (7 features):
  global_degree       total degree in BioGRID (log-transformed)
  n_pathways          Reactome pathways (n>=10) the protein belongs to (log)
  shannon_H_norm      normalised Shannon diversity of pathway membership
                      weighted by inverse pathway size, range [0,1]
  mean_ratio          mean(local_degree / global_degree) across all pathways
  bridge_score        number of distant cross-PPI pathway pairs (log)
  betweenness         approximate betweenness centrality (k=500 pivots)
  cross_ppi_count     PPI partners outside the protein's pathway neighbourhood (log)

Taxonomy labels (greedy assignment from cluster centroids):
  structural_hub        highest betweenness × global_degree
                        — true network bottlenecks: NUDT21, CUL3, PRKN, MYC
  pathway_bridge        highest bridge_score / global_degree
                        — span functionally distant pathways: ubiquitin, proteasome,
                          ribosomal proteins, and mid-range multi-pathway regulators
  specialist            highest mean_ratio
                        — most interactions confined within pathways: ZNF proteins,
                          complex subunits
  functional_connector  highest n_pathways / global_degree
                        — many pathways, very few PPI: olfactory receptors, ZNF genes
  background            low on all features

Inputs:
    --ppi           Biogr_Uniprot_clean.txt
    --features      protein_features.tsv          (from scripts 08/09/10)
    --matrix        protein_pathway_matrix_uniprot.npz
    --rows          protein_pathway_matrix_uniprot_rows.txt
    --cols          protein_pathway_matrix_uniprot_cols.tsv
    --similarity    pathway_ppi_similarity.tsv     (from script 10)
Outputs:
    --output        protein_classification.tsv
"""

import argparse
import csv
from collections import defaultdict

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import umap
import hdbscan


FEAT_COLS = ["global_degree", "n_pathways", "shannon_H_norm", "mean_ratio",
             "bridge_score", "betweenness", "cross_ppi_count"]
LOG_COLS  = {"global_degree", "n_pathways", "bridge_score", "cross_ppi_count"}
DISTANT_RELS = {"distant", "same_depth", "sibling"}

UMAP_CLUSTER_DIM = 10
UMAP_VIS_DIM     = 2
HDBSCAN_MCS      = 80
HDBSCAN_MS       = 5
BETWEENNESS_K    = 500


# ── Feature computation helpers ──────────────────────────────────────────────

def compute_betweenness(ppi_path: str, k: int = BETWEENNESS_K) -> dict:
    print(f"  Building PPI graph and computing betweenness (k={k}) ...")
    G = nx.Graph()
    with open(ppi_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] != parts[1]:
                G.add_edge(parts[0], parts[1])
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return nx.betweenness_centrality(G, k=k, normalized=True, seed=42)


def compute_cross_ppi_count(
    ppi_path: str,
    matrix_path: str,
    rows_path: str,
    cols_path: str,
) -> dict:
    """Count PPI partners outside a protein's collective pathway neighbourhood."""
    M = sp.load_npz(matrix_path).astype(np.int32)
    proteins = open(rows_path).read().splitlines()
    pathways = []
    with open(cols_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pathways.append(row["pathway_id"])
    M_csr = M.tocsr()

    # Pathway member sets
    pathway_members: dict = {}
    for j, pid in enumerate(pathways):
        pathway_members[pid] = set(
            proteins[i] for i in M_csr.getcol(j).nonzero()[0]
        )

    # Per-protein pathway neighbourhood
    pw_neighbourhood: dict = {}
    for i, acc in enumerate(proteins):
        cols = M_csr.getrow(i).nonzero()[1]
        nbhd: set = set()
        for j in cols:
            nbhd.update(pathway_members[pathways[j]])
        nbhd.discard(acc)
        pw_neighbourhood[acc] = nbhd

    # PPI adjacency
    adj: dict = defaultdict(set)
    with open(ppi_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] != parts[1]:
                adj[parts[0]].add(parts[1])
                adj[parts[1]].add(parts[0])

    return {
        acc: len(adj.get(acc, set()) - pw_neighbourhood.get(acc, set()))
        for acc in proteins
    }


def compute_bridge_scores(
    similarity_path: str,
    matrix_path: str,
    rows_path: str,
    cols_path: str,
) -> dict:
    M = sp.load_npz(matrix_path).astype(np.int32)
    proteins = open(rows_path).read().splitlines()
    pathways = []
    with open(cols_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pathways.append(row["pathway_id"])
    M_csr = M.tocsr()

    protein_pathways: dict = defaultdict(list)
    for i, acc in enumerate(proteins):
        cols = M_csr.getrow(i).nonzero()[1]
        protein_pathways[acc] = [pathways[j] for j in cols]

    cross_ppi_pairs: set = set()
    with open(similarity_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["cross_ppi"]) > 0 and row["relation"] in DISTANT_RELS:
                pa, pb = row["pathway_a"], row["pathway_b"]
                cross_ppi_pairs.add((pa, pb))
                cross_ppi_pairs.add((pb, pa))

    return {
        acc: sum(
            1 for a in range(len(pw))
            for b in range(a + 1, len(pw))
            if (pw[a], pw[b]) in cross_ppi_pairs
        )
        for acc, pw in protein_pathways.items()
    }


# ── Taxonomy assignment ───────────────────────────────────────────────────────

def assign_taxonomy(centroids: np.ndarray) -> dict:
    """Greedy label assignment. Returns dict[cluster_id -> label]."""
    gd  = centroids[:, 0]
    np_ = centroids[:, 1]
    mr  = centroids[:, 3]
    bs  = centroids[:, 4]
    bc  = centroids[:, 5]

    assigned: dict = {}
    remaining = list(range(len(centroids)))

    def pick(score_fn, label):
        c = remaining[int(np.argmax([score_fn(i) for i in remaining]))]
        assigned[label] = c
        remaining.remove(c)

    pick(lambda i: bc[i] * gd[i],        "structural_hub")
    pick(lambda i: bs[i] / (gd[i] + 1),  "pathway_bridge")
    pick(lambda i: mr[i],                 "specialist")
    pick(lambda i: np_[i] / (gd[i] + 1), "functional_connector")
    for c in remaining:
        assigned["background"] = c

    return {v: k for k, v in assigned.items()}


# ── Main ─────────────────────────────────────────────────────────────────────

def run(
    ppi_path: str,
    features_path: str,
    matrix_path: str,
    rows_path: str,
    cols_path: str,
    similarity_path: str,
    out_path: str,
) -> None:
    # Load features
    print("Loading features ...")
    records = []
    with open(features_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            for k in ("n_pathways", "global_degree", "max_local_degree",
                      "bridge_score", "cross_ppi_count"):
                row[k] = int(row[k])
            for k in ("shannon_H", "shannon_H_norm", "mean_local_degree",
                      "mean_ratio", "betweenness"):
                row[k] = float(row[k])
            row["is_paralog"] = row["is_paralog"] == "True"
            records.append(row)
    print(f"  {len(records):,} proteins")

    # Compute missing features if needed
    if "betweenness" not in records[0] or all(r["betweenness"] == 0 for r in records):
        print("Computing betweenness centrality ...")
        bc = compute_betweenness(ppi_path)
        for r in records:
            r["betweenness"] = bc.get(r["uniprot_id"], 0.0)

    if "cross_ppi_count" not in records[0] or all(r["cross_ppi_count"] == 0 for r in records):
        print("Computing cross-pathway PPI counts ...")
        cp = compute_cross_ppi_count(ppi_path, matrix_path, rows_path, cols_path)
        for r in records:
            r["cross_ppi_count"] = cp.get(r["uniprot_id"], 0)

    if "bridge_score" not in records[0] or all(r["bridge_score"] == 0 for r in records):
        print("Computing bridge scores ...")
        bs = compute_bridge_scores(similarity_path, matrix_path, rows_path, cols_path)
        for r in records:
            r["bridge_score"] = bs.get(r["uniprot_id"], 0)

    # Feature matrix
    X_raw = np.array([[r[c] for c in FEAT_COLS] for r in records], dtype=float)
    X_log = X_raw.copy()
    for j, col in enumerate(FEAT_COLS):
        if col in LOG_COLS:
            X_log[:, j] = np.log1p(X_raw[:, j])
    X_scaled = RobustScaler().fit_transform(X_log)

    # UMAP 10D for clustering
    print(f"UMAP {UMAP_CLUSTER_DIM}D (clustering) ...")
    emb_nd = umap.UMAP(
        n_components=UMAP_CLUSTER_DIM, n_neighbors=30, min_dist=0.0,
        metric="euclidean", random_state=42,
    ).fit_transform(X_scaled)

    # UMAP 2D for visualisation
    print("UMAP 2D (visualisation) ...")
    emb_2d = umap.UMAP(
        n_components=UMAP_VIS_DIM, n_neighbors=30, min_dist=0.1,
        metric="euclidean", random_state=42,
    ).fit_transform(X_scaled)

    # HDBSCAN, with k-means fallback if density clustering collapses.
    # Exact betweenness can compress the UMAP manifold enough that HDBSCAN
    # identifies too few clusters for the five-class taxonomy. In that case,
    # k-means is applied to the 10D UMAP embedding (still UMAP-first).
    print(f"HDBSCAN (min_cluster_size={HDBSCAN_MCS}) ...")
    cl = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MCS, min_samples=HDBSCAN_MS,
        cluster_selection_method="eom",
    )
    labels = cl.fit_predict(emb_nd)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  {n_clusters} clusters, {(labels==-1).sum()} noise points")
    print(f"  Sizes: {sorted([(labels==c).sum() for c in range(n_clusters)], reverse=True)}")

    if n_clusters < 5:
        print("  HDBSCAN found fewer than 5 clusters; using k-means fallback on UMAP 10D embedding ...")
        km = KMeans(n_clusters=5, random_state=42, n_init=30)
        labels = km.fit_predict(emb_nd)
        n_clusters = 5
        print(f"  k-means sizes: {sorted([(labels==c).sum() for c in range(n_clusters)], reverse=True)}")

    # Taxonomy
    centroids = np.array([X_raw[labels == c].mean(axis=0) for c in range(n_clusters)])
    label_map = assign_taxonomy(centroids)
    label_map[-1] = "unclassified"
    print(f"  Assignments: { {v: k for k, v in label_map.items() if k != -1} }")

    # Annotate and write
    for i, r in enumerate(records):
        r["cluster"]  = int(labels[i])
        r["taxonomy"] = label_map.get(int(labels[i]), f"cluster_{labels[i]}")
        r["umap_x"]   = round(float(emb_2d[i, 0]), 4)
        r["umap_y"]   = round(float(emb_2d[i, 1]), 4)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Written: {out_path}")

    # Summary
    print("\n=== Per-class summary ===")
    for tax in ["structural_hub", "pathway_bridge", "functional_connector",
                "specialist", "background", "unclassified"]:
        sub = [r for r in records if r["taxonomy"] == tax]
        if not sub:
            continue
        gds = [r["global_degree"] for r in sub]
        nps = [r["n_pathways"] for r in sub]
        mrs = [r["mean_ratio"] for r in sub]
        bss = [r["bridge_score"] for r in sub]
        bcs = [r["betweenness"] for r in sub]
        print(f"\n  {tax} (n={len(sub)})")
        print(f"    global_degree: mean={np.mean(gds):.1f}  max={max(gds)}")
        print(f"    n_pathways:    mean={np.mean(nps):.1f}  max={max(nps)}")
        print(f"    mean_ratio:    mean={np.mean(mrs):.3f}")
        print(f"    bridge_score:  mean={np.mean(bss):.1f}  max={max(bss)}")
        print(f"    betweenness:   mean={np.mean(bcs):.5f}  max={max(bcs):.5f}")
        top = sorted(sub, key=lambda r: -r["global_degree"])[:8]
        print(f"    Top proteins:  {', '.join(r['gene_name'] for r in top)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppi",        default="../Biogr_Uniprot_clean.txt")
    parser.add_argument("--features",   default="../protein_features.tsv")
    parser.add_argument("--matrix",     default="../protein_pathway_matrix_uniprot.npz")
    parser.add_argument("--rows",       default="../protein_pathway_matrix_uniprot_rows.txt")
    parser.add_argument("--cols",       default="../protein_pathway_matrix_uniprot_cols.tsv")
    parser.add_argument("--similarity", default="../pathway_ppi_similarity.tsv")
    parser.add_argument("--output",     default="../protein_classification.tsv")
    args = parser.parse_args()

    run(args.ppi, args.features, args.matrix, args.rows, args.cols,
        args.similarity, args.output)


if __name__ == "__main__":
    main()
