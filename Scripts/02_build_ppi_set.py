"""
02_build_ppi_set.py
-------------------
Loads a two-column UniProt accession PPI file, deduplicates edges,
removes self-loops, and writes a canonical (sorted) edge list.

Inputs:
    --input   path to raw PPI file (two UniProt accession columns, space/tab separated)
Outputs:
    --output  path to cleaned edge list TSV (default: ../ppi_clean.tsv)

Also prints basic network statistics.
"""

import argparse
import csv
from pathlib import Path


def load_ppi(path: str) -> set:
    """Load PPI file and return a set of canonical (min, max) accession pairs."""
    ppi_set: set = set()
    n_raw = 0
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            n_raw += 1
            if a == b:
                continue  # skip self-loops
            ppi_set.add((min(a, b), max(a, b)))

    print(f"Raw lines:              {n_raw:,}")
    print(f"Self-loops removed:     {n_raw - len(ppi_set) - (n_raw - len(ppi_set)):,}")
    print(f"Unique undirected edges:{len(ppi_set):,}")

    proteins = set()
    for a, b in ppi_set:
        proteins.add(a)
        proteins.add(b)
    print(f"Unique proteins:        {len(proteins):,}")
    return ppi_set


def write_ppi(ppi_set: set, out_path: str) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["protein_a", "protein_b"])
        for a, b in sorted(ppi_set):
            writer.writerow([a, b])
    print(f"Written {len(ppi_set):,} edges → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="../Biogr_Uniprot_clean.txt",
        help="Input PPI file (two UniProt accession columns)",
    )
    parser.add_argument(
        "--output",
        default="../ppi_clean.tsv",
        help="Output cleaned edge list",
    )
    args = parser.parse_args()

    print(f"Loading PPI from {args.input} ...")
    ppi_set = load_ppi(args.input)
    write_ppi(ppi_set, args.output)


if __name__ == "__main__":
    main()
