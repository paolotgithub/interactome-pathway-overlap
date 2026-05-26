"""
05_compute_pathway_overlap.py
------------------------------
Computes pairwise pathway overlap for all C(n_pathways, 2) pairs:
  - Raw Jaccard index
  - Weighted Jaccard (hub proteins down-weighted by 1/log(1+k_p))
  - Hypergeometric significance (N = BioGRID proteins)
  - Reactome hierarchy relationship annotation (direct_parent_child,
    ancestor_descendant, sibling, same_depth, distant)

Inputs:
    --matrix    protein_pathway_matrix_uniprot.npz
    --cols      protein_pathway_matrix_uniprot_cols.tsv
    --ppi       Biogr_Uniprot_clean.txt   (to count BioGRID proteins for N)
Outputs:
    --output    pathway_overlap_corrected.tsv
    (writes only pairs with J > 0 or FDR < 0.05)

Output columns:
    pathway_a, pathway_b, name_a, name_b,
    size_a, size_b, intersection, union,
    jaccard, weighted_jaccard,
    pval_hypergeom, padj_bh,
    relation, is_sibling,
    depth_a, depth_b,
    redundant_group_a, redundant_group_b
"""

import argparse
import csv
from collections import defaultdict, deque

import numpy as np
import requests
import scipy.sparse as sp
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

REACTOME_HIERARCHY_URL = (
    "https://reactome.org/download/current/ReactomePathwaysRelation.txt"
)


# ── Hierarchy helpers ────────────────────────────────────────────────────────

def load_hierarchy(url: str) -> tuple[dict, dict]:
    """Download Reactome hierarchy and return (parent_of, children_of) dicts."""
    print(f"Downloading hierarchy from {url} ...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    parent_of: dict = defaultdict(set)
    children_of: dict = defaultdict(set)
    for line in r.text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) == 2:
            p, c = parts
            if p.startswith("R-HSA-") and c.startswith("R-HSA-"):
                parent_of[c].add(p)
                children_of[p].add(c)
    return parent_of, children_of


def compute_depths(parent_of: dict, children_of: dict) -> dict:
    """BFS from roots to assign depth to each node."""
    all_nodes = set(parent_of) | set(children_of)
    roots = all_nodes - set(parent_of)
    depth: dict = {}
    queue: deque = deque((n, 0) for n in roots)
    while queue:
        node, d = queue.popleft()
        if node in depth:
            continue
        depth[node] = d
        for child in children_of.get(node, []):
            queue.append((child, d + 1))
    return depth


def get_ancestors(node: str, parent_of: dict) -> set:
    """Return all ancestors of node (including itself)."""
    anc: set = set()
    q: deque = deque([node])
    while q:
        n = q.popleft()
        if n in anc:
            continue
        anc.add(n)
        q.extend(parent_of.get(n, []))
    return anc


def build_sibling_pairs(pathways: list, children_of: dict) -> set:
    """Return set of (min_idx, max_idx) index pairs that are siblings."""
    our_set = set(pathways)
    pathway_idx = {p: i for i, p in enumerate(pathways)}
    sibling_pairs: set = set()
    for parent in children_of:
        kids = [c for c in children_of[parent] if c in our_set]
        for a in range(len(kids)):
            for b in range(a + 1, len(kids)):
                i = pathway_idx[kids[a]]
                j = pathway_idx[kids[b]]
                sibling_pairs.add((min(i, j), max(i, j)))
    return sibling_pairs


# ── Redundancy collapse (union-find) ────────────────────────────────────────

def build_redundancy_groups(
    pathways: list,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    jaccard: np.ndarray,
    is_hier: np.ndarray,
    depth_arr: np.ndarray,
) -> dict:
    """Union-find collapse of J=1.0 non-hierarchical pairs.

    Returns dict[pathway_id -> canonical_representative_id].
    """
    uf = {p: p for p in pathways}

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    def unite(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        # Keep the representative at lower depth (more general)
        dx = depth_arr[pathways.index(rx)] if rx in pathways else 99
        dy = depth_arr[pathways.index(ry)] if ry in pathways else 99
        if dx <= dy:
            uf[ry] = rx
        else:
            uf[rx] = ry

    pathway_idx = {p: i for i, p in enumerate(pathways)}
    depth_by_id = {p: depth_arr[i] for i, p in enumerate(pathways)}

    for k in range(len(idx_i)):
        if jaccard[k] < 1.0 or is_hier[k]:
            continue
        unite(pathways[idx_i[k]], pathways[idx_j[k]])

    return {p: find(p) for p in pathways}


# ── Main ─────────────────────────────────────────────────────────────────────

def compute_overlap(
    matrix_path: str,
    cols_path: str,
    ppi_path: str,
    out_path: str,
) -> None:
    # Load matrix
    M = sp.load_npz(matrix_path).astype(np.int32)
    pathways, pnames = [], {}
    with open(cols_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pathways.append(row["pathway_id"])
            pnames[row["pathway_id"]] = row["pathway_name"]
    n_pathways = len(pathways)
    pathway_idx = {p: i for i, p in enumerate(pathways)}

    # BioGRID protein count for hypergeometric background
    bg_proteins: set = set()
    with open(ppi_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                bg_proteins.add(parts[0]); bg_proteins.add(parts[1])
    N_bg = len(bg_proteins)
    print(f"BioGRID proteins (N): {N_bg:,}")

    # Intersection matrix (int32 to avoid overflow)
    print("Computing M^T M ...")
    MtM = (M.T @ M).toarray()
    sizes = np.diag(MtM).copy()

    # Weighted Jaccard
    print("Computing weighted Jaccard ...")
    k_p = np.array(M.sum(axis=1)).flatten().astype(float)
    weights = 1.0 / np.log1p(k_p)
    W = sp.diags(weights) @ M.astype(np.float64)
    WtW = (W.T @ W).toarray()
    w_sizes = np.diag(WtW).copy()

    # Upper triangle indices
    idx_i, idx_j = np.triu_indices(n_pathways, k=1)
    inter = MtM[idx_i, idx_j]
    si = sizes[idx_i]; sj = sizes[idx_j]
    union = (si + sj - inter).astype(float)
    jaccard = np.where(union > 0, inter.astype(float) / union, 0.0)

    w_inter = WtW[idx_i, idx_j]
    w_si = w_sizes[idx_i]; w_sj = w_sizes[idx_j]
    w_union = w_si + w_sj - w_inter
    w_jaccard = np.where(w_union > 0, w_inter / w_union, 0.0)

    # Hypergeometric p-values
    print("Computing hypergeometric p-values ...")
    mask_pos = inter > 0
    pvals = np.ones(len(inter))
    pvals[mask_pos] = stats.hypergeom.sf(
        inter[mask_pos] - 1, N_bg, si[mask_pos], sj[mask_pos]
    )
    min_pos = np.finfo(float).tiny
    _, padj, _, _ = multipletests(
        np.where(pvals == 0.0, min_pos, pvals), method="fdr_bh"
    )
    padj = np.where(pvals == 0.0, 0.0, padj)
    print(f"  Significant pairs (FDR < 0.05): {(padj < 0.05).sum():,}")

    # Hierarchy annotation
    print("Annotating hierarchy relationships ...")
    parent_of, children_of = load_hierarchy(REACTOME_HIERARCHY_URL)
    depth_dict = compute_depths(parent_of, children_of)
    depth_arr = np.array([depth_dict.get(p, 0) for p in pathways])

    our_set = set(pathways)
    ancestor_sets = {p: get_ancestors(p, parent_of) for p in pathways}

    direct_pc_set: set = set()
    for child in pathways:
        for parent in parent_of.get(child, []):
            if parent in our_set:
                ci, pi = pathway_idx[child], pathway_idx[parent]
                direct_pc_set.add((min(ci, pi), max(ci, pi)))

    sibling_pairs = build_sibling_pairs(pathways, children_of)

    is_direct_pc = np.zeros(len(inter), dtype=bool)
    is_anc_desc  = np.zeros(len(inter), dtype=bool)
    for k in range(len(inter)):
        pair = (min(idx_i[k], idx_j[k]), max(idx_i[k], idx_j[k]))
        if pair in direct_pc_set:
            is_direct_pc[k] = True
            is_anc_desc[k]  = True
        else:
            pi, pj = pathways[idx_i[k]], pathways[idx_j[k]]
            if pi in ancestor_sets[pj] or pj in ancestor_sets[pi]:
                is_anc_desc[k] = True

    is_sibling = np.array([
        (min(idx_i[k], idx_j[k]), max(idx_i[k], idx_j[k])) in sibling_pairs
        for k in range(len(inter))
    ])

    d_i = depth_arr[idx_i]; d_j = depth_arr[idx_j]
    relation = np.full(len(inter), "distant", dtype=object)
    relation[is_anc_desc & ~is_direct_pc] = "ancestor_descendant"
    relation[is_direct_pc] = "direct_parent_child"
    relation[(relation == "distant") & is_sibling] = "sibling"
    relation[(relation == "distant") & (d_i == d_j) & ~is_sibling] = "same_depth"

    # Redundancy groups
    is_hier = is_anc_desc.copy()
    canonical = build_redundancy_groups(
        pathways, idx_i, idx_j, jaccard, is_hier, depth_arr
    )

    # Write output
    print("Writing output ...")
    write_mask = (jaccard > 0) | (padj < 0.05)
    with open(out_path, "w") as f:
        f.write(
            "pathway_a\tpathway_b\tname_a\tname_b\t"
            "size_a\tsize_b\tintersection\tunion\t"
            "jaccard\tweighted_jaccard\t"
            "pval_hypergeom\tpadj_bh\t"
            "relation\tis_sibling\t"
            "depth_a\tdepth_b\t"
            "redundant_group_a\tredundant_group_b\n"
        )
        for k in np.where(write_mask)[0]:
            i, j = int(idx_i[k]), int(idx_j[k])
            pa, pb = pathways[i], pathways[j]
            f.write(
                f"{pa}\t{pb}\t{pnames[pa]}\t{pnames[pb]}\t"
                f"{si[k]}\t{sj[k]}\t{inter[k]}\t{int(union[k])}\t"
                f"{jaccard[k]:.6f}\t{w_jaccard[k]:.6f}\t"
                f"{pvals[k]:.4e}\t{padj[k]:.4e}\t"
                f"{relation[k]}\t{is_sibling[k]}\t"
                f"{d_i[k]}\t{d_j[k]}\t"
                f"{canonical[pa]}\t{canonical[pb]}\n"
            )
    print(f"Written {write_mask.sum():,} rows → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="../protein_pathway_matrix_uniprot.npz")
    parser.add_argument("--cols",   default="../protein_pathway_matrix_uniprot_cols.tsv")
    parser.add_argument("--ppi",    default="../Biogr_Uniprot_clean.txt")
    parser.add_argument("--output", default="../pathway_overlap_corrected.tsv")
    args = parser.parse_args()

    compute_overlap(args.matrix, args.cols, args.ppi, args.output)


if __name__ == "__main__":
    main()
