"""
08_protein_promiscuity.py
--------------------------
For each protein in the membership matrix M, computes:

  n_pathways      raw count of pathways the protein belongs to
  shannon_H       Shannon diversity index over pathway memberships,
                  weighted by inverse pathway size:
                      p_ij = (1/size_j) / Σ_k (1/size_k)  for k in pathways of protein i
                      H(i) = -Σ_j p_ij * log(p_ij)
  shannon_H_norm  H(i) / log(n_pathways)  — normalised to [0,1]
                  1.0 = maximally diverse (all pathways equally weighted)
                  0.0 = all weight on a single pathway

Proteins with high H_norm and moderate-to-high n_pathways are genuinely
promiscuous (span functionally distant pathways). Proteins with high n but
low H_norm belong predominantly to closely related (often parent-child)
pathways.

Ig/TCR variable-region gene families are flagged as paralogous families
(is_paralog_family = True) since their identical pathway membership is an
annotation artefact rather than biological promiscuity.

Inputs:
    --matrix    protein_pathway_matrix_uniprot.npz
    --rows      protein_pathway_matrix_uniprot_rows.txt
    --cols      protein_pathway_matrix_uniprot_cols.tsv
    --mapping   reactome_protein_pathway_human.tsv  (for gene names)
Outputs:
    --out-protein   protein_promiscuity.tsv       (one row per protein)
    --out-expanded  protein_pathway_list.tsv      (one row per protein-pathway pair)
"""

import argparse
import csv
import re
from collections import defaultdict

import numpy as np
import scipy.sparse as sp


# Regex for Ig/TCR variable-region gene families
_PARALOG_RE = re.compile(
    r"^(V\d|IGH[VDJ]|IGL[VJ]|IGK[VJ]|TRA[VJ]|TRB[VDJ]|TRG[VJ]|TRD[VDJ]|"
    r"IGHG|IGHA|IGHM|IGHD|IGHE|IGLC|IGKC)",
    re.IGNORECASE,
)


def is_paralog_family(gene: str) -> bool:
    return bool(_PARALOG_RE.match(gene))


def load_gene_names(mapping_path: str) -> dict:
    """Return dict[uniprot_id -> gene_name] from Reactome mapping TSV."""
    gene_names: dict = {}
    with open(mapping_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            gene_names[row["uniprot_id"]] = row["gene_name"]
    return gene_names


def compute_promiscuity(
    matrix_path: str,
    rows_path: str,
    cols_path: str,
    mapping_path: str,
    out_protein: str,
    out_expanded: str,
) -> None:
    # ── Load matrix and indices ──────────────────────────────────────────────
    M = sp.load_npz(matrix_path).astype(np.int32)
    proteins = open(rows_path).read().splitlines()
    pathways, pnames = [], {}
    with open(cols_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pathways.append(row["pathway_id"])
            pnames[row["pathway_id"]] = row["pathway_name"]

    n_proteins, n_pathways = M.shape
    print(f"Matrix: {n_proteins:,} proteins × {n_pathways:,} pathways")

    gene_names = load_gene_names(mapping_path)

    # ── Pathway sizes and inverse-size weights ───────────────────────────────
    pathway_sizes = np.array(M.sum(axis=0)).flatten().astype(float)
    inv_sizes = 1.0 / pathway_sizes   # w_j = 1 / size_j

    # ── Raw pathway counts ───────────────────────────────────────────────────
    raw_counts = np.array(M.sum(axis=1)).flatten()

    # ── Weighted membership matrix W[i,j] = M[i,j] / size_j ────────────────
    W = M.astype(np.float64).multiply(inv_sizes[np.newaxis, :])
    W_csr = W.tocsr()
    M_csr = M.tocsr()

    # ── Shannon diversity H(i) ───────────────────────────────────────────────
    print("Computing Shannon diversity H(i) ...")
    H = np.zeros(n_proteins)
    for i in range(n_proteins):
        data = W_csr.getrow(i).data
        if len(data) == 0:
            continue
        total = data.sum()
        if total == 0:
            continue
        p = data / total
        p = p[p > 0]
        H[i] = -np.sum(p * np.log(p))

    # Normalised H: H(i) / log(n_pathways_i), 0 for proteins in 0 or 1 pathway
    H_norm = np.where(raw_counts > 1, H / np.log(raw_counts), 0.0)

    print(f"H range:      {H.min():.4f} – {H.max():.4f}")
    print(f"H_norm range: {H_norm[raw_counts>1].min():.4f} – "
          f"{H_norm[raw_counts>1].max():.4f}")
    print(f"H_norm mean:  {H_norm[raw_counts>1].mean():.4f}  "
          f"median: {np.median(H_norm[raw_counts>1]):.4f}")

    # ── Per-protein pathway list ─────────────────────────────────────────────
    print("Building per-protein pathway list ...")
    protein_pathways: dict = defaultdict(list)
    for i, acc in enumerate(proteins):
        cols = M_csr.getrow(i).nonzero()[1]
        for j in cols:
            protein_pathways[acc].append((pathways[j], pnames[pathways[j]],
                                          int(pathway_sizes[j])))

    # ── Write protein-level table ────────────────────────────────────────────
    print(f"Writing {out_protein} ...")
    out_rows = []
    for i, acc in enumerate(proteins):
        if raw_counts[i] == 0:
            continue
        gene = gene_names.get(acc, acc)
        pw_sorted = sorted(protein_pathways[acc], key=lambda x: x[0])
        out_rows.append({
            "uniprot_id":        acc,
            "gene_name":         gene,
            "n_pathways":        int(raw_counts[i]),
            "shannon_H":         round(float(H[i]), 6),
            "shannon_H_norm":    round(float(H_norm[i]), 6),
            "is_paralog_family": is_paralog_family(gene),
            "pathway_ids":       "|".join(pid for pid, _, _ in pw_sorted),
            "pathway_names":     "|".join(pn for _, pn, _ in pw_sorted),
        })

    # Sort: H descending, then n_pathways descending
    out_rows.sort(key=lambda r: (-r["shannon_H"], -r["n_pathways"]))

    with open(out_protein, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=out_rows[0].keys())
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"  Written {len(out_rows):,} proteins")

    # ── Write expanded (protein × pathway) table ─────────────────────────────
    print(f"Writing {out_expanded} ...")
    n_written = 0
    with open(out_expanded, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "uniprot_id", "gene_name", "n_pathways",
            "shannon_H", "shannon_H_norm", "is_paralog_family",
            "pathway_id", "pathway_name", "pathway_size",
        ])
        for i, acc in enumerate(proteins):
            if raw_counts[i] == 0:
                continue
            gene = gene_names.get(acc, acc)
            paralog = is_paralog_family(gene)
            for pid, pname, psize in sorted(protein_pathways[acc]):
                writer.writerow([
                    acc, gene, int(raw_counts[i]),
                    round(float(H[i]), 6), round(float(H_norm[i]), 6),
                    paralog, pid, pname, psize,
                ])
                n_written += 1
    print(f"  Written {n_written:,} rows")

    # ── Print summary ────────────────────────────────────────────────────────
    non_ig = [r for r in out_rows if not r["is_paralog_family"]]
    print(f"\n=== Summary (non-paralog proteins: {len(non_ig):,}) ===")
    for label, lo, hi in [("1", 1, 1), ("2-9", 2, 9), ("10-49", 10, 49), (">=50", 50, 9999)]:
        n = sum(1 for r in non_ig if lo <= r["n_pathways"] <= hi)
        print(f"  n_pathways {label:5s}: {n:,}")

    print("\n  Top 10 by H (non-paralog):")
    for r in non_ig[:10]:
        print(f"    H={r['shannon_H']:.3f}  H_norm={r['shannon_H_norm']:.3f}  "
              f"n={r['n_pathways']:4d}  {r['gene_name']:<12}  {r['uniprot_id']}")

    high_n = [r for r in non_ig if r["n_pathways"] >= 50]
    low_H_norm = sorted(high_n, key=lambda x: x["shannon_H_norm"])
    print("\n  Top 10 high-n (>=50) but low H_norm (redundant pathway membership):")
    for r in low_H_norm[:10]:
        print(f"    H_norm={r['shannon_H_norm']:.3f}  n={r['n_pathways']:4d}  "
              f"{r['gene_name']:<12}  {r['uniprot_id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix",       default="../protein_pathway_matrix_uniprot.npz")
    parser.add_argument("--rows",         default="../protein_pathway_matrix_uniprot_rows.txt")
    parser.add_argument("--cols",         default="../protein_pathway_matrix_uniprot_cols.tsv")
    parser.add_argument("--mapping",      default="../reactome_protein_pathway_human.tsv")
    parser.add_argument("--out-protein",  default="../protein_promiscuity.tsv")
    parser.add_argument("--out-expanded", default="../protein_pathway_list.tsv")
    args = parser.parse_args()

    compute_promiscuity(
        args.matrix, args.rows, args.cols, args.mapping,
        args.out_protein, args.out_expanded,
    )


if __name__ == "__main__":
    main()
