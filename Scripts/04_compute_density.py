"""
04_compute_density.py
---------------------
Computes intra-pathway PPI density for each pathway and applies a
size-corrected significance test (hypergeometric vs. random graph null).

Inputs:
    --matrix    protein_pathway_matrix_uniprot.npz
    --cols      protein_pathway_matrix_uniprot_cols.tsv
    --rows      protein_pathway_matrix_uniprot_rows.txt
    --ppi       Biogr_Uniprot_clean.txt  (two-column UniProt edge list)
Outputs:
    --output    pathway_ppi_density_corrected.tsv

Columns in output:
    pathway_id, pathway_name, n_proteins, n_possible_pairs,
    n_intra_ppi, density,
    pval_density    hypergeometric p-value vs random graph null
    padj_density    BH-adjusted p-value
    frac_missing_biogrid   fraction of pathway proteins absent from BioGRID
    low_coverage_flag      True if frac_missing > 0.5
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests


def load_ppi_set(path: str) -> tuple[set, set]:
    """Return (ppi_set of canonical pairs, set of all proteins in BioGRID)."""
    ppi_set: set = set()
    proteins: set = set()
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            proteins.add(a); proteins.add(b)
            if a != b:
                ppi_set.add((min(a, b), max(a, b)))
    return ppi_set, proteins


def compute_density(
    matrix_path: str,
    cols_path: str,
    rows_path: str,
    ppi_path: str,
    out_path: str,
) -> None:
    # Load matrix
    M = sp.load_npz(matrix_path).astype(np.int32)
    proteins = open(rows_path).read().splitlines()
    pathways, pnames = [], {}
    with open(cols_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pathways.append(row["pathway_id"])
            pnames[row["pathway_id"]] = row["pathway_name"]

    # Load PPI
    print("Loading PPI network ...")
    ppi_set, biogrid_proteins = load_ppi_set(ppi_path)
    n_edges = len(ppi_set)
    n_bg = len(biogrid_proteins)
    p_bg = n_edges / (n_bg * (n_bg - 1) / 2)
    print(f"  Edges: {n_edges:,}  Proteins: {n_bg:,}  p_bg: {p_bg:.6f}")

    # Protein index
    protein_idx = {p: i for i, p in enumerate(proteins)}

    # Per-pathway protein sets (from matrix)
    pathway_protein_sets: dict = {}
    for j, pid in enumerate(pathways):
        row_indices = M[:, j].nonzero()[0]
        pathway_protein_sets[pid] = [proteins[i] for i in row_indices]

    # Coverage: fraction of pathway proteins absent from BioGRID
    coverage: dict = {}
    for pid, prots in pathway_protein_sets.items():
        n_miss = sum(1 for p in prots if p not in biogrid_proteins)
        coverage[pid] = n_miss / len(prots) if prots else 0.0

    # Compute density per pathway
    print("Computing intra-pathway PPI density ...")
    results = []
    for pid in pathways:
        prots = pathway_protein_sets[pid]
        n = len(prots)
        n_poss = n * (n - 1) // 2

        if n_poss == 0:
            results.append({
                "pathway_id": pid, "pathway_name": pnames[pid],
                "n_proteins": n, "n_possible_pairs": 0,
                "n_intra_ppi": 0, "density": float("nan"),
                "pval_density": float("nan"),
                "frac_missing_biogrid": round(coverage[pid], 4),
                "low_coverage_flag": coverage[pid] > 0.5,
            })
            continue

        n_intra = sum(
            1 for i in range(len(prots))
            for j in range(i + 1, len(prots))
            if (min(prots[i], prots[j]), max(prots[i], prots[j])) in ppi_set
        )
        density = n_intra / n_poss

        # Hypergeometric p-value vs random graph null
        # Population: C(n_bg, 2) possible edges
        # K: observed edges in BioGRID
        # n: C(n_pathway, 2) possible pairs
        # k: observed intra-pathway edges
        N_pop = n_bg * (n_bg - 1) // 2
        pval = (
            stats.hypergeom.sf(n_intra - 1, N_pop, n_edges, n_poss)
            if n_intra > 0 else 1.0
        )

        results.append({
            "pathway_id": pid, "pathway_name": pnames[pid],
            "n_proteins": n, "n_possible_pairs": n_poss,
            "n_intra_ppi": n_intra, "density": density,
            "pval_density": pval,
            "frac_missing_biogrid": round(coverage[pid], 4),
            "low_coverage_flag": coverage[pid] > 0.5,
        })

    # BH FDR correction
    min_pos = np.finfo(float).tiny
    pv_arr = np.array([
        r["pval_density"] if (
            r["pval_density"] == r["pval_density"] and r["pval_density"] > 0
        ) else 1.0
        for r in results
    ])
    _, padj, _, _ = multipletests(
        np.where(pv_arr == 0.0, min_pos, pv_arr), method="fdr_bh"
    )
    for i, r in enumerate(results):
        r["padj_density"] = float(padj[i])

    sig = sum(1 for r in results if r["padj_density"] < 0.05)
    print(f"  Significant pathways (FDR < 0.05): {sig} / {len(results)}")

    # Write output
    results.sort(key=lambda x: x["padj_density"])
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Written → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="../protein_pathway_matrix_uniprot.npz")
    parser.add_argument("--cols",   default="../protein_pathway_matrix_uniprot_cols.tsv")
    parser.add_argument("--rows",   default="../protein_pathway_matrix_uniprot_rows.txt")
    parser.add_argument("--ppi",    default="../Biogr_Uniprot_clean.txt")
    parser.add_argument("--output", default="../pathway_ppi_density_corrected.tsv")
    args = parser.parse_args()

    compute_density(args.matrix, args.cols, args.rows, args.ppi, args.output)


if __name__ == "__main__":
    main()
