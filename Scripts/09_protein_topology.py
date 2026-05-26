"""
09_protein_topology.py
-----------------------
Topological characterisation of proteins within pathways.

For each (protein, pathway) pair, computes:

  local_degree    number of PPI partners of the protein that are also
                  annotated to the same pathway (degree in the pathway
                  PPI subgraph)
  global_degree   total degree of the protein in the full BioGRID network
  ratio           local_degree / global_degree
                  → 1.0 : all partners are within this pathway (specialist)
                  → 0.0 : no partners within this pathway (generalist hub)

Inter-pathway connector candidates are defined as proteins with:
  global_degree >= GLOBAL_DEG_MIN  (highly connected globally)
  ratio         <= RATIO_MAX       (most partners outside this pathway)
  local_degree  >= 1               (at least one intra-pathway link)

Inputs:
    --ppi       Biogr_Uniprot_clean.txt
    --matrix    protein_pathway_matrix_uniprot.npz
    --rows      protein_pathway_matrix_uniprot_rows.txt
    --cols      protein_pathway_matrix_uniprot_cols.tsv
    --mapping   reactome_protein_pathway_human.tsv
Outputs:
    --out-topology    protein_pathway_topology.tsv   (all records)
    --out-connectors  interpathway_connectors.tsv    (connector candidates)
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
import scipy.sparse as sp


GLOBAL_DEG_MIN = 50
RATIO_MAX      = 0.10


def load_ppi(path: str) -> tuple[set, dict]:
    """Load PPI edge list. Returns (ppi_set, global_degree dict)."""
    ppi_set: set = set()
    global_degree: dict = defaultdict(int)
    with open(path) as f:
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
    return ppi_set, global_degree


def build_adjacency(ppi_set: set) -> dict:
    """Build adjacency list from edge set."""
    adj: dict = defaultdict(set)
    for a, b in ppi_set:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def compute_topology(
    ppi_path: str,
    matrix_path: str,
    rows_path: str,
    cols_path: str,
    mapping_path: str,
    out_topology: str,
    out_connectors: str,
    global_deg_min: int = GLOBAL_DEG_MIN,
    ratio_max: float = RATIO_MAX,
) -> None:
    # ── Load PPI ─────────────────────────────────────────────────────────────
    print("Loading PPI network ...")
    ppi_set, global_degree = load_ppi(ppi_path)
    adj = build_adjacency(ppi_set)
    print(f"  Edges: {len(ppi_set):,}  Proteins: {len(global_degree):,}")
    print(f"  Global degree range: {min(global_degree.values())}–"
          f"{max(global_degree.values())}")

    # ── Load matrix ───────────────────────────────────────────────────────────
    print("Loading membership matrix ...")
    M = sp.load_npz(matrix_path).astype(np.int32)
    proteins = open(rows_path).read().splitlines()
    pathways, pnames = [], {}
    with open(cols_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pathways.append(row["pathway_id"])
            pnames[row["pathway_id"]] = row["pathway_name"]
    M_csr = M.tocsr()

    # Gene names
    gene_names: dict = {}
    with open(mapping_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            gene_names[row["uniprot_id"]] = row["gene_name"]

    # ── Per-pathway protein sets ──────────────────────────────────────────────
    print("Building per-pathway protein sets ...")
    pathway_protein_sets: dict = {}
    for j, pid in enumerate(pathways):
        members = [proteins[i] for i in M_csr.getcol(j).nonzero()[0]]
        pathway_protein_sets[pid] = set(members)

    # ── Local degree per (protein, pathway) ───────────────────────────────────
    print("Computing local degree per (protein, pathway) ...")
    n_pathways = len(pathways)
    all_rows: list = []

    for idx_p, pid in enumerate(pathways):
        if idx_p % 300 == 0:
            print(f"  {idx_p}/{n_pathways} ...", flush=True)
        members = pathway_protein_sets[pid]
        psize   = len(members)
        pname   = pnames[pid]

        for acc in members:
            local_deg  = len(adj[acc] & members) if acc in adj else 0
            global_deg = global_degree.get(acc, 0)
            ratio      = (local_deg / global_deg
                          if global_deg > 0 else float("nan"))
            all_rows.append({
                "pathway_id":    pid,
                "pathway_name":  pname,
                "pathway_size":  psize,
                "uniprot_id":    acc,
                "gene_name":     gene_names.get(acc, acc),
                "local_degree":  local_deg,
                "global_degree": global_deg,
                "ratio":         round(ratio, 6) if ratio == ratio else float("nan"),
            })

    print(f"  Total records: {len(all_rows):,}")

    # ── Write full topology table ─────────────────────────────────────────────
    all_rows.sort(key=lambda r: (r["pathway_id"], r["uniprot_id"]))
    with open(out_topology, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Written: {out_topology}")

    # ── Inter-pathway connector candidates ────────────────────────────────────
    connectors = [
        r for r in all_rows
        if r["global_degree"] >= global_deg_min
        and r["ratio"] == r["ratio"]          # not nan
        and r["ratio"] <= ratio_max
        and r["local_degree"] >= 1
    ]
    connectors.sort(key=lambda r: (r["ratio"], -r["global_degree"]))
    with open(out_connectors, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(connectors)
    print(f"Written: {out_connectors}  ({len(connectors):,} records)")

    # ── Summary ───────────────────────────────────────────────────────────────
    valid = [r for r in all_rows
             if r["ratio"] == r["ratio"] and r["global_degree"] > 0]
    ratios = [r["ratio"] for r in valid]
    print(f"\n=== Summary ===")
    print(f"Records with ratio defined: {len(valid):,}")
    print(f"Ratio mean:   {np.mean(ratios):.4f}")
    print(f"Ratio median: {np.median(ratios):.4f}")
    print(f"Ratio = 0 (no local PPI):  {sum(1 for r in ratios if r == 0):,}")
    print(f"Ratio <= 0.10:             {sum(1 for r in ratios if r <= 0.10):,}")
    print(f"Ratio >= 0.50:             {sum(1 for r in ratios if r >= 0.50):,}")
    print(f"Ratio = 1.0 (all local):   {sum(1 for r in ratios if r == 1.0):,}")

    # Best connector per unique protein
    best: dict = {}
    for r in connectors:
        acc = r["uniprot_id"]
        if acc not in best or r["ratio"] < best[acc]["ratio"]:
            best[acc] = r
    top = sorted(best.values(), key=lambda r: (r["ratio"], -r["global_degree"]))
    print(f"\nUnique connector proteins: {len(top):,}")
    print("\nTop 15 inter-pathway connectors (lowest ratio, highest global degree):")
    for r in top[:15]:
        print(f"  ratio={r['ratio']:.4f}  glob={r['global_degree']:5d}  "
              f"loc={r['local_degree']:3d}  {r['gene_name']:<12}  "
              f"{r['pathway_name'][:45]}")

    # Top specialists
    specialists = [r for r in valid
                   if r["ratio"] >= 0.5 and r["local_degree"] >= 5]
    specialists.sort(key=lambda r: (-r["ratio"], -r["local_degree"]))
    print("\nTop 10 pathway specialists (ratio >= 0.5, local_degree >= 5):")
    for r in specialists[:10]:
        print(f"  ratio={r['ratio']:.4f}  glob={r['global_degree']:4d}  "
              f"loc={r['local_degree']:3d}  {r['gene_name']:<12}  "
              f"{r['pathway_name'][:45]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppi",           default="../Biogr_Uniprot_clean.txt")
    parser.add_argument("--matrix",        default="../protein_pathway_matrix_uniprot.npz")
    parser.add_argument("--rows",          default="../protein_pathway_matrix_uniprot_rows.txt")
    parser.add_argument("--cols",          default="../protein_pathway_matrix_uniprot_cols.tsv")
    parser.add_argument("--mapping",       default="../reactome_protein_pathway_human.tsv")
    parser.add_argument("--out-topology",  default="../protein_pathway_topology.tsv")
    parser.add_argument("--out-connectors",default="../interpathway_connectors.tsv")
    parser.add_argument("--global-deg-min",type=int,   default=GLOBAL_DEG_MIN)
    parser.add_argument("--ratio-max",     type=float, default=RATIO_MAX)
    args = parser.parse_args()

    compute_topology(
        args.ppi, args.matrix, args.rows, args.cols, args.mapping,
        args.out_topology, args.out_connectors,
        global_deg_min=args.global_deg_min,
        ratio_max=args.ratio_max,
    )


if __name__ == "__main__":
    main()
