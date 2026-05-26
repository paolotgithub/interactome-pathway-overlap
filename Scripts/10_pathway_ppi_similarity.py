"""
10_pathway_ppi_similarity.py
-----------------------------
Builds a pathway-pathway similarity matrix based on cross-pathway PPI,
distinct from protein membership overlap (Jaccard).

For each pathway pair (A, B):

  cross_ppi       number of PPI edges between a protein in A and a protein in B
                  (computed as C = M^T * A_ppi * M, off-diagonal entries)
  cross_density   cross_ppi / (|A| * |B|)
                  fraction of all possible cross-pathway pairs that have a PPI
  pval_cross      hypergeometric p-value: is cross_ppi above random expectation?
                  (background: N = BioGRID proteins, K = total BioGRID edges,
                   n = |A|*|B| possible cross-pairs)
  padj_cross      BH FDR-corrected p-value
  pure_crosstalk  True if cross_ppi > 0 AND jaccard == 0 AND not hierarchical
                  — two pathways with no shared proteins but PPI between them

Three output files:
  pathway_ppi_similarity.tsv        all pairs with cross_ppi > 0 or FDR < 0.05
  pathway_crosstalk_network.tsv     pure crosstalk (J=0, FDR<0.05, density>=0.01)
  pathway_combined_similarity.tsv   union: J>=0.10 OR cross_density>=0.05,
                                    FDR<0.05, no hierarchical pairs

Inputs:
    --ppi       Biogr_Uniprot_clean.txt
    --matrix    protein_pathway_matrix_uniprot.npz
    --rows      protein_pathway_matrix_uniprot_rows.txt
    --cols      protein_pathway_matrix_uniprot_cols.tsv
    --overlap   pathway_overlap_corrected.tsv  (for Jaccard and relation labels)
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests


HIER_EX        = {"direct_parent_child", "ancestor_descendant"}
DENSITY_MIN_CT = 0.01   # minimum cross_density for pure crosstalk network
J_MIN_COMBINED = 0.10   # minimum Jaccard for combined network
D_MIN_COMBINED = 0.05   # minimum cross_density for combined network


def load_ppi_adjacency(
    ppi_path: str, protein_idx: dict, n_proteins: int
) -> tuple[sp.csr_matrix, dict, int]:
    """Build sparse symmetric adjacency matrix and global degree dict."""
    rows_a, cols_a = [], []
    global_degree: dict = defaultdict(int)
    ppi_set: set = set()

    with open(ppi_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            if a == b:
                continue
            edge = (min(a, b), max(a, b))
            if edge in ppi_set:
                continue
            ppi_set.add(edge)
            global_degree[a] += 1
            global_degree[b] += 1
            if a in protein_idx and b in protein_idx:
                i, j = protein_idx[a], protein_idx[b]
                rows_a += [i, j]
                cols_a += [j, i]

    data = np.ones(len(rows_a), dtype=np.int32)
    A = sp.csr_matrix(
        (data, (rows_a, cols_a)),
        shape=(n_proteins, n_proteins),
        dtype=np.int32,
    )
    n_bg = len(set(global_degree.keys()))
    n_edges = len(ppi_set)
    return A, global_degree, n_bg, n_edges


def compute_cross_ppi_matrix(
    M: sp.csr_matrix, A: sp.csr_matrix
) -> np.ndarray:
    """Compute C = M^T * A * M (pathway x pathway cross-PPI count matrix)."""
    MtA = M.T.astype(np.float32) @ A.astype(np.float32)
    C = (MtA @ M.astype(np.float32)).toarray().astype(np.int32)
    return C


def run(
    ppi_path: str,
    matrix_path: str,
    rows_path: str,
    cols_path: str,
    overlap_path: str,
    out_full: str,
    out_crosstalk: str,
    out_combined: str,
) -> None:
    # ── Load matrix ───────────────────────────────────────────────────────────
    print("Loading membership matrix ...")
    M = sp.load_npz(matrix_path).astype(np.int32)
    proteins = open(rows_path).read().splitlines()
    pathways, pnames = [], {}
    with open(cols_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pathways.append(row["pathway_id"])
            pnames[row["pathway_id"]] = row["pathway_name"]

    n_proteins, n_pathways = M.shape
    protein_idx = {p: i for i, p in enumerate(proteins)}
    pathway_sizes = np.array(M.sum(axis=0)).flatten().astype(np.int32)
    print(f"  {n_proteins:,} proteins × {n_pathways:,} pathways")

    # ── Build adjacency matrix ────────────────────────────────────────────────
    print("Building protein adjacency matrix ...")
    A, _, n_bg, n_edges = load_ppi_adjacency(ppi_path, protein_idx, n_proteins)
    print(f"  A nnz: {A.nnz:,}  BioGRID proteins: {n_bg:,}  edges: {n_edges:,}")

    # ── Cross-pathway PPI matrix ──────────────────────────────────────────────
    print("Computing C = M^T * A * M ...")
    C = compute_cross_ppi_matrix(M, A)

    # ── Upper triangle ────────────────────────────────────────────────────────
    idx_j, idx_k = np.triu_indices(n_pathways, k=1)
    cross_ppi   = C[idx_j, idx_k]
    sj          = pathway_sizes[idx_j].astype(float)
    sk          = pathway_sizes[idx_k].astype(float)
    max_cross   = sj * sk
    cross_density = np.where(max_cross > 0, cross_ppi / max_cross, 0.0)

    print(f"  Pairs with cross_ppi > 0: {(cross_ppi > 0).sum():,}")
    print(f"  Cross density mean (>0):  {cross_density[cross_density>0].mean():.5f}")

    # ── Hypergeometric significance ───────────────────────────────────────────
    print("Computing hypergeometric p-values ...")
    N_pop = n_bg * (n_bg - 1) // 2
    mask_pos = cross_ppi > 0
    pvals = np.ones(len(cross_ppi))
    pvals[mask_pos] = stats.hypergeom.sf(
        cross_ppi[mask_pos] - 1, N_pop, n_edges,
        max_cross.astype(int)[mask_pos],
    )
    min_pos = np.finfo(float).tiny
    _, padj, _, _ = multipletests(
        np.where(pvals == 0.0, min_pos, pvals), method="fdr_bh"
    )
    padj = np.where(pvals == 0.0, 0.0, padj)
    print(f"  Significant pairs (FDR < 0.05): {(padj < 0.05).sum():,}")

    # ── Load Jaccard and relation from existing overlap file ──────────────────
    print("Loading Jaccard and hierarchy relations ...")
    jaccard_map: dict = {}
    relation_map: dict = {}
    with open(overlap_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row["pathway_a"], row["pathway_b"])
            rkey = (row["pathway_b"], row["pathway_a"])
            jaccard_map[key] = jaccard_map[rkey] = float(row["jaccard"])
            relation_map[key] = relation_map[rkey] = row["relation"]

    # ── Write full similarity file ────────────────────────────────────────────
    print(f"Writing {out_full} ...")
    write_mask = (cross_ppi > 0) | (padj < 0.05)
    n_written = 0
    with open(out_full, "w") as f:
        f.write(
            "pathway_a\tpathway_b\tname_a\tname_b\t"
            "size_a\tsize_b\t"
            "jaccard\t"
            "cross_ppi\tcross_density\t"
            "pval_cross\tpadj_cross\t"
            "relation\tpure_crosstalk\n"
        )
        for k in np.where(write_mask)[0]:
            j, l = int(idx_j[k]), int(idx_k[k])
            pa, pb = pathways[j], pathways[l]
            jac = jaccard_map.get((pa, pb), 0.0)
            rel = relation_map.get((pa, pb), "distant")
            pure = (
                cross_ppi[k] > 0
                and jac == 0.0
                and rel not in HIER_EX
            )
            f.write(
                f"{pa}\t{pb}\t{pnames[pa]}\t{pnames[pb]}\t"
                f"{int(sj[k])}\t{int(sk[k])}\t"
                f"{jac:.6f}\t"
                f"{cross_ppi[k]}\t{cross_density[k]:.8f}\t"
                f"{pvals[k]:.4e}\t{padj[k]:.4e}\t"
                f"{rel}\t{pure}\n"
            )
            n_written += 1
    print(f"  Written {n_written:,} rows")

    # ── Write filtered networks ───────────────────────────────────────────────
    all_rows: list = []
    with open(out_full) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row["cross_density"] = float(row["cross_density"])
            row["cross_ppi"]     = int(row["cross_ppi"])
            row["jaccard"]       = float(row["jaccard"])
            row["padj_cross"]    = float(row["padj_cross"])
            all_rows.append(row)

    # Pure crosstalk
    ct_rows = [
        r for r in all_rows
        if r["pure_crosstalk"] == "True"
        and r["padj_cross"] < 0.05
        and r["cross_density"] >= DENSITY_MIN_CT
    ]
    ct_rows.sort(key=lambda r: -r["cross_density"])
    with open(out_crosstalk, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(ct_rows)
    print(f"Written {out_crosstalk}  ({len(ct_rows):,} pairs)")

    # Combined similarity
    comb_rows = [
        r for r in all_rows
        if r["relation"] not in HIER_EX
        and r["padj_cross"] < 0.05
        and (r["jaccard"] >= J_MIN_COMBINED or r["cross_density"] >= D_MIN_COMBINED)
    ]
    comb_rows.sort(key=lambda r: -r["cross_density"])
    with open(out_combined, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(comb_rows)
    print(f"Written {out_combined}  ({len(comb_rows):,} pairs)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"Total pairs:                    {len(cross_ppi):,}")
    print(f"Pairs with cross_ppi > 0:       {(cross_ppi>0).sum():,}")
    print(f"Significant (FDR < 0.05):       {(padj<0.05).sum():,}")
    print(f"Pure crosstalk (FDR<0.05):      {len(ct_rows):,}")
    print(f"Combined network:               {len(comb_rows):,}")

    overlap_only  = sum(1 for r in comb_rows
                        if r["jaccard"] >= J_MIN_COMBINED
                        and r["cross_density"] < D_MIN_COMBINED)
    ct_only       = sum(1 for r in comb_rows
                        if r["jaccard"] < J_MIN_COMBINED
                        and r["cross_density"] >= D_MIN_COMBINED)
    both          = sum(1 for r in comb_rows
                        if r["jaccard"] >= J_MIN_COMBINED
                        and r["cross_density"] >= D_MIN_COMBINED)
    print(f"  Overlap only (J>={J_MIN_COMBINED}):     {overlap_only:,}")
    print(f"  Crosstalk only (d>={D_MIN_COMBINED}):   {ct_only:,}")
    print(f"  Both:                         {both:,}")

    print("\nTop 10 pure crosstalk pairs by cross_density:")
    for r in ct_rows[:10]:
        print(f"  d={r['cross_density']:.4f}  ppi={r['cross_ppi']:5d}  "
              f"({r['size_a']}x{r['size_b']})  "
              f"{r['name_a'][:35]} | {r['name_b'][:35]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppi",          default="../Biogr_Uniprot_clean.txt")
    parser.add_argument("--matrix",       default="../protein_pathway_matrix_uniprot.npz")
    parser.add_argument("--rows",         default="../protein_pathway_matrix_uniprot_rows.txt")
    parser.add_argument("--cols",         default="../protein_pathway_matrix_uniprot_cols.tsv")
    parser.add_argument("--overlap",      default="../pathway_overlap_corrected.tsv")
    parser.add_argument("--out-full",     default="../pathway_ppi_similarity.tsv")
    parser.add_argument("--out-crosstalk",default="../pathway_crosstalk_network.tsv")
    parser.add_argument("--out-combined", default="../pathway_combined_similarity.tsv")
    args = parser.parse_args()

    run(
        args.ppi, args.matrix, args.rows, args.cols, args.overlap,
        args.out_full, args.out_crosstalk, args.out_combined,
    )


if __name__ == "__main__":
    main()
