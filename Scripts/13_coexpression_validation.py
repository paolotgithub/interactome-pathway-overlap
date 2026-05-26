"""
13_coexpression_validation.py
------------------------------
Validates the PPI-pathway pipeline using GTEx v10 tissue-level co-expression.

Hypothesis: protein pairs within the same Reactome pathway that have a
BioGRID PPI show higher Spearman co-expression (across 68 GTEx tissues)
than pathway-matched pairs without a PPI.

Method:
  - For each pathway, enumerate all intra-pathway protein pairs
  - Split into PPI pairs (BioGRID edge present) and no-PPI pairs
  - Compute Spearman ρ using pre-ranked expression vectors (= Pearson on ranks)
    via vectorised matrix multiplication: C = E @ E.T / (T-1)
  - Test PPI > no-PPI with one-sided Mann-Whitney U (rbc = 2U/(n1*n2) - 1)
  - Apply BH FDR correction across pathways
  - Repeat excluding top-1% global-degree hub proteins (sensitivity analysis)

Inputs:
    --gtex      GTEx_v10_gene_median_tpm.gct.gz  (downloaded automatically if absent)
    --ppi       Biogr_Uniprot_clean.txt
    --matrix    protein_pathway_matrix_uniprot.npz
    --rows      protein_pathway_matrix_uniprot_rows.txt
    --cols      protein_pathway_matrix_uniprot_cols.tsv
    --mapping   reactome_protein_pathway_human.tsv
Outputs:
    --out-pairs   coexpression_pairs.tsv
    --out-stats   coexpression_pathway_stats.tsv
"""

import argparse
import csv
import gzip
import random
from collections import defaultdict

import numpy as np
import requests
import scipy.sparse as sp
from scipy.stats import mannwhitneyu, rankdata
from sklearn.preprocessing import scale
from statsmodels.stats.multitest import multipletests


GTEX_URL = (
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/"
    "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_median_tpm.gct.gz"
)
MIN_TISSUES_EXPRESSED = 10   # min tissues with TPM > 0.1 to include a gene
MAX_NOPPI_PER_PATHWAY = 500  # cap no-PPI pairs per pathway (random sample)
HUB_PERCENTILE        = 1    # top-N% by global degree = hub


def download_gtex(path: str) -> None:
    print(f"Downloading GTEx v10 from {GTEX_URL} ...")
    r = requests.get(GTEX_URL, timeout=180)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    print(f"  Saved: {path}  ({len(r.content)/1e6:.1f} MB)")


def load_gtex(path: str) -> tuple[dict, list]:
    """Load GTEx tissue-median TPM. Returns (gene_expr dict, tissue_names list)."""
    gene_expr: dict = {}
    tissues: list = []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if i < 2:
                continue
            if i == 2:
                cols = line.split("\t")
                tissues = cols[2:]
                continue
            parts = line.split("\t")
            sym = parts[1]
            tpm = np.array(parts[2:], dtype=np.float32)
            if (tpm > 0.1).sum() < MIN_TISSUES_EXPRESSED:
                continue
            log_tpm = np.log2(tpm + 1)
            if sym not in gene_expr or np.median(log_tpm) > np.median(gene_expr[sym]):
                gene_expr[sym] = log_tpm
    return gene_expr, tissues


def prerank(gene_expr: dict) -> dict:
    """Convert log-TPM to normalised ranks (Spearman = Pearson on ranks)."""
    normed: dict = {}
    for g, v in gene_expr.items():
        r = rankdata(v).astype(np.float32)
        r -= r.mean()
        s = r.std()
        normed[g] = r / s if s > 0 else r
    return normed


def mw_rbc(a: list, b: list) -> tuple:
    """One-sided Mann-Whitney U (a > b). Returns (median_a, median_b, pval, rbc)."""
    stat, pval = mannwhitneyu(a, b, alternative="greater")
    rbc = (2 * stat) / (len(a) * len(b)) - 1
    return float(np.median(a)), float(np.median(b)), pval, rbc


def run(
    gtex_path: str,
    ppi_path: str,
    matrix_path: str,
    rows_path: str,
    cols_path: str,
    mapping_path: str,
    out_pairs: str,
    out_stats: str,
) -> None:
    import os
    if not os.path.exists(gtex_path):
        download_gtex(gtex_path)

    # Load expression
    print("Loading GTEx expression ...")
    gene_expr, tissues = load_gtex(gtex_path)
    normed = prerank(gene_expr)
    n_tissues = len(tissues)
    print(f"  Genes: {len(gene_expr):,}  Tissues: {n_tissues}")

    # Load gene names
    gene_of: dict = {}
    with open(mapping_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            gene_of[row["uniprot_id"]] = row["gene_name"]

    # Load PPI
    ppi_genes: set = set()
    global_degree: dict = defaultdict(int)
    with open(ppi_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            ga, gb = gene_of.get(a, ""), gene_of.get(b, "")
            if ga and gb and ga != gb:
                ppi_genes.add((min(ga, gb), max(ga, gb)))
                global_degree[ga] += 1
                global_degree[gb] += 1

    # Hub proteins (top-1% by global degree)
    deg_vals = sorted(global_degree.values(), reverse=True)
    hub_thresh = deg_vals[int(len(deg_vals) * HUB_PERCENTILE / 100)]
    hub_genes = {g for g, d in global_degree.items() if d >= hub_thresh}
    print(f"  PPI edges: {len(ppi_genes):,}  Hub threshold: {hub_thresh}  "
          f"Hubs: {len(hub_genes)}")

    # Load matrix
    M = sp.load_npz(matrix_path).astype(np.int32)
    proteins = open(rows_path).read().splitlines()
    pathways, pnames = [], {}
    with open(cols_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pathways.append(row["pathway_id"])
            pnames[row["pathway_id"]] = row["pathway_name"]
    M_csr = M.tocsr()

    # Per-pathway vectorised Spearman
    print("Computing per-pathway Spearman correlations ...")
    random.seed(42)

    all_ppi, all_noppi = [], []
    all_ppi_nh, all_noppi_nh = [], []
    pw_data: dict = {}
    pair_rows: list = []

    for j, pid in enumerate(pathways):
        members_idx = M_csr.getcol(j).nonzero()[0]
        genes = [gene_of.get(proteins[i], "") for i in members_idx]
        genes = [g for g in genes if g and g in normed]
        n = len(genes)
        if n < 5:
            continue

        E = np.stack([normed[g] for g in genes])          # (n, T)
        C = (E @ E.T) / (n_tissues - 1)                   # (n, n) Spearman ρ

        ppi_rho, noppi_pool = [], []
        for a in range(n):
            for b in range(a + 1, n):
                ga, gb = genes[a], genes[b]
                rho = float(C[a, b])
                has_ppi = (min(ga, gb), max(ga, gb)) in ppi_genes
                is_hub  = ga in hub_genes or gb in hub_genes
                if has_ppi:
                    ppi_rho.append((rho, is_hub, ga, gb))
                else:
                    noppi_pool.append((rho, is_hub, ga, gb))

        if len(noppi_pool) > MAX_NOPPI_PER_PATHWAY:
            noppi_pool = random.sample(noppi_pool, MAX_NOPPI_PER_PATHWAY)

        pw_ppi, pw_noppi = [], []
        for rho, is_hub, ga, gb in ppi_rho:
            pw_ppi.append(rho)
            all_ppi.append(rho)
            if not is_hub:
                all_ppi_nh.append(rho)
            pair_rows.append({"pathway_id": pid, "gene_a": ga, "gene_b": gb,
                               "has_ppi": True, "is_hub_pair": is_hub,
                               "spearman_rho": round(rho, 6)})
        for rho, is_hub, ga, gb in noppi_pool:
            pw_noppi.append(rho)
            all_noppi.append(rho)
            if not is_hub:
                all_noppi_nh.append(rho)
            pair_rows.append({"pathway_id": pid, "gene_a": ga, "gene_b": gb,
                               "has_ppi": False, "is_hub_pair": is_hub,
                               "spearman_rho": round(rho, 6)})

        pw_data[pid] = {"ppi": pw_ppi, "noppi": pw_noppi}

        if j % 300 == 0:
            print(f"  {j}/{len(pathways)} ...", flush=True)

    print(f"  PPI pairs: {len(all_ppi):,}  no-PPI pairs: {len(all_noppi):,}")

    # Global tests
    m_p, m_n, pv, rb = mw_rbc(all_ppi, all_noppi)
    m_p_nh, m_n_nh, pv_nh, rb_nh = mw_rbc(all_ppi_nh, all_noppi_nh)
    print(f"\nGlobal (all):    PPI={m_p:.4f}  no-PPI={m_n:.4f}  "
          f"Δ={m_p-m_n:+.4f}  rbc={rb:.4f}  p={pv:.3e}")
    print(f"Global (no-hub): PPI={m_p_nh:.4f}  no-PPI={m_n_nh:.4f}  "
          f"Δ={m_p_nh-m_n_nh:+.4f}  rbc={rb_nh:.4f}  p={pv_nh:.3e}")

    # Per-pathway tests
    pw_results = []
    for pid, d in pw_data.items():
        if len(d["ppi"]) < 3 or len(d["noppi"]) < 3:
            continue
        m_p2, m_n2, pv2, rb2 = mw_rbc(d["ppi"], d["noppi"])
        pw_results.append({
            "pathway_id":   pid,
            "pathway_name": pnames.get(pid, ""),
            "n_ppi":        len(d["ppi"]),
            "n_noppi":      len(d["noppi"]),
            "median_ppi":   round(m_p2, 4),
            "median_noppi": round(m_n2, 4),
            "delta_median": round(m_p2 - m_n2, 4),
            "pval":         pv2,
            "rbc":          round(rb2, 4),
        })

    _, padj, _, _ = multipletests([r["pval"] for r in pw_results], method="fdr_bh")
    for i, r in enumerate(pw_results):
        r["padj_bh"] = padj[i]
    pw_results.sort(key=lambda x: -x["rbc"])

    sig = sum(1 for r in pw_results if r["padj_bh"] < 0.05 and r["rbc"] > 0)
    print(f"Per-pathway: {sig}/{len(pw_results)} significant (FDR<0.05, rbc>0)")

    # Write outputs
    with open(out_pairs, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=pair_rows[0].keys())
        writer.writeheader()
        writer.writerows(pair_rows)
    print(f"Written: {out_pairs}  ({len(pair_rows):,} rows)")

    with open(out_stats, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=pw_results[0].keys())
        writer.writeheader()
        writer.writerows(pw_results)
    print(f"Written: {out_stats}  ({len(pw_results)} pathways)")

    print("\nTop 10 pathways by rbc:")
    for r in pw_results[:10]:
        print(f"  rbc={r['rbc']:.4f}  padj={r['padj_bh']:.2e}  "
              f"Δmedian={r['delta_median']:+.4f}  "
              f"n_ppi={r['n_ppi']:4d}  {r['pathway_name'][:55]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtex",      default="../GTEx_v10_gene_median_tpm.gct.gz")
    parser.add_argument("--ppi",       default="../Biogr_Uniprot_clean.txt")
    parser.add_argument("--matrix",    default="../protein_pathway_matrix_uniprot.npz")
    parser.add_argument("--rows",      default="../protein_pathway_matrix_uniprot_rows.txt")
    parser.add_argument("--cols",      default="../protein_pathway_matrix_uniprot_cols.tsv")
    parser.add_argument("--mapping",   default="../reactome_protein_pathway_human.tsv")
    parser.add_argument("--out-pairs", default="../coexpression_pairs.tsv")
    parser.add_argument("--out-stats", default="../coexpression_pathway_stats.tsv")
    args = parser.parse_args()

    run(args.gtex, args.ppi, args.matrix, args.rows, args.cols,
        args.mapping, args.out_pairs, args.out_stats)


if __name__ == "__main__":
    main()
