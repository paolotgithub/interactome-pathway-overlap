"""
06_filter_networks.py
---------------------
Applies threshold filters to the corrected overlap file to produce two
analysis-ready pathway networks:

  1. Functional network   : weighted_jaccard >= 0.10, FDR < 0.05,
                            excluding direct_parent_child and ancestor_descendant
  2. Strict co-function   : weighted_jaccard >= 0.25, FDR < 0.05,
                            same exclusions

Also computes connected components for each network and prints a summary.

Inputs:
    --overlap   pathway_overlap_corrected.tsv  (from script 05)
Outputs:
    --fn-out    pathway_overlap_functional_network_corrected.tsv
    --scp-out   pathway_overlap_strict_cofunctional_corrected.tsv
"""

import argparse
import csv
from collections import defaultdict, deque


HIER_EXCLUDE = {"direct_parent_child", "ancestor_descendant"}


def load_overlap(path: str) -> list:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["jaccard"]          = float(row["jaccard"])
            row["weighted_jaccard"] = float(row["weighted_jaccard"])
            row["padj_bh"]          = float(row["padj_bh"])
            row["size_a"]           = int(row["size_a"])
            row["size_b"]           = int(row["size_b"])
            row["depth_a"]          = int(row["depth_a"])
            row["depth_b"]          = int(row["depth_b"])
            rows.append(row)
    return rows


def apply_filter(
    rows: list,
    wj_min: float,
    fdr_max: float,
    exclude_relations: set,
) -> list:
    return [
        r for r in rows
        if r["weighted_jaccard"] >= wj_min
        and r["padj_bh"] < fdr_max
        and r["relation"] not in exclude_relations
    ]


def write_tsv(path: str, rows: list) -> int:
    if not rows:
        return 0
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def connected_components(rows: list) -> list:
    """Return list of sets (connected components) sorted by size descending."""
    adj: dict = defaultdict(set)
    for r in rows:
        adj[r["pathway_a"]].add(r["pathway_b"])
        adj[r["pathway_b"]].add(r["pathway_a"])

    visited: set = set()
    components: list = []
    for node in adj:
        if node in visited:
            continue
        comp: set = set()
        q: deque = deque([node])
        while q:
            n = q.popleft()
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            q.extend(adj[n] - visited)
        components.append(comp)

    components.sort(key=lambda x: -len(x))
    return components


def print_summary(label: str, rows: list, pnames: dict) -> None:
    import numpy as np
    wjs = [r["weighted_jaccard"] for r in rows]
    components = connected_components(rows)

    # Degree
    degree: dict = defaultdict(int)
    for r in rows:
        degree[r["pathway_a"]] += 1
        degree[r["pathway_b"]] += 1
    degs = list(degree.values())

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"  Edges:      {len(rows):,}")
    print(f"  Nodes:      {len(degree):,}")
    print(f"  wJ mean:    {np.mean(wjs):.3f}  median: {np.median(wjs):.3f}")
    print(f"  Degree mean:{np.mean(degs):.1f}  max: {max(degs)}")
    print(f"  Components: {len(components)}")
    print(f"  Sizes (top 10): {[len(c) for c in components[:10]]}")

    # Relation breakdown
    rel_counts: dict = defaultdict(int)
    for r in rows:
        rel_counts[r["relation"]] += 1
    print("  Relations:")
    for rel, cnt in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"    {rel}: {cnt:,}")

    # Top 5 hubs
    top_hubs = sorted(degree.items(), key=lambda x: -x[1])[:5]
    print("  Top 5 hubs by degree:")
    for pid, deg in top_hubs:
        print(f"    deg={deg:4d}  {pnames.get(pid, pid)[:60]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap", default="../pathway_overlap_corrected.tsv")
    parser.add_argument("--fn-out",  default="../pathway_overlap_functional_network_corrected.tsv")
    parser.add_argument("--scp-out", default="../pathway_overlap_strict_cofunctional_corrected.tsv")
    parser.add_argument("--cols",    default="../protein_pathway_matrix_uniprot_cols.tsv")
    args = parser.parse_args()

    # Load pathway names for summary
    pnames: dict = {}
    with open(args.cols) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pnames[row["pathway_id"]] = row["pathway_name"]

    print(f"Loading {args.overlap} ...")
    rows = load_overlap(args.overlap)
    print(f"  Total rows: {len(rows):,}")

    # Functional network
    fn_rows = apply_filter(rows, wj_min=0.10, fdr_max=0.05,
                            exclude_relations=HIER_EXCLUDE)
    fn_rows.sort(key=lambda r: -r["weighted_jaccard"])
    n_fn = write_tsv(args.fn_out, fn_rows)
    print(f"\nFunctional network:  {n_fn:,} pairs → {args.fn_out}")
    print_summary("Functional network (wJ >= 0.10, FDR < 0.05)", fn_rows, pnames)

    # Strict co-function
    scp_rows = apply_filter(rows, wj_min=0.25, fdr_max=0.05,
                             exclude_relations=HIER_EXCLUDE)
    scp_rows.sort(key=lambda r: -r["weighted_jaccard"])
    n_scp = write_tsv(args.scp_out, scp_rows)
    print(f"\nStrict co-function:  {n_scp:,} pairs → {args.scp_out}")
    print_summary("Strict co-function (wJ >= 0.25, FDR < 0.05)", scp_rows, pnames)


if __name__ == "__main__":
    main()
