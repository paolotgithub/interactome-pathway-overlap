"""
03_build_membership_matrix.py
------------------------------
Builds a binary protein × pathway membership matrix M from the Reactome
protein-pathway mapping, restricted to pathways with n >= MIN_PATHWAY_SIZE proteins.

Inputs:
    --mapping   reactome_protein_pathway_human.tsv  (from script 01)
Outputs:
    --matrix    protein_pathway_matrix_uniprot.npz   (scipy CSR, int32)
    --rows      protein_pathway_matrix_uniprot_rows.txt  (one UniProt accession per line)
    --cols      protein_pathway_matrix_uniprot_cols.tsv  (pathway_id, pathway_name)

M[i, j] = 1  iff protein i belongs to pathway j.
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
import scipy.sparse as sp


MIN_PATHWAY_SIZE = 10


def load_mapping(path: str, min_size: int) -> tuple:
    """Load protein-pathway mapping and filter to pathways with >= min_size proteins.

    Returns
    -------
    proteins    : sorted list of UniProt accessions
    pathways    : sorted list of pathway IDs
    pnames      : dict[pathway_id -> pathway_name]
    pp_map      : dict[pathway_id -> set of UniProt accessions]
    """
    pp_map: dict = defaultdict(set)
    pnames: dict = {}

    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row["pathway_id"]
            acc = row["uniprot_id"]
            pp_map[pid].add(acc)
            pnames[pid] = row["pathway_name"]

    # Filter by size
    pp_map = {pid: prots for pid, prots in pp_map.items()
               if len(prots) >= min_size}
    pnames = {pid: pnames[pid] for pid in pp_map}

    protein_set: set = set()
    for prots in pp_map.values():
        protein_set.update(prots)

    proteins = sorted(protein_set)
    pathways = sorted(pp_map.keys())

    print(f"Pathways (n >= {min_size}): {len(pathways):,}")
    print(f"Unique proteins:           {len(proteins):,}")
    print(f"Non-zero entries:          {sum(len(v) for v in pp_map.values()):,}")
    return proteins, pathways, pnames, pp_map


def build_matrix(proteins, pathways, pp_map) -> sp.csr_matrix:
    protein_idx = {p: i for i, p in enumerate(proteins)}
    pathway_idx = {p: i for i, p in enumerate(pathways)}

    rows, cols = [], []
    for pid, prots in pp_map.items():
        j = pathway_idx[pid]
        for acc in prots:
            rows.append(protein_idx[acc])
            cols.append(j)

    data = np.ones(len(rows), dtype=np.int32)
    M = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(proteins), len(pathways)),
        dtype=np.int32,
    )
    sparsity = M.nnz / (M.shape[0] * M.shape[1]) * 100
    print(f"Matrix shape: {M.shape[0]:,} × {M.shape[1]:,}  "
          f"nnz={M.nnz:,}  sparsity={sparsity:.2f}%")
    return M


def write_outputs(M, proteins, pathways, pnames, matrix_path, rows_path, cols_path):
    sp.save_npz(matrix_path, M)
    print(f"Matrix written → {matrix_path}")

    with open(rows_path, "w") as f:
        f.write("\n".join(proteins))
    print(f"Row index written → {rows_path}  ({len(proteins):,} proteins)")

    with open(cols_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["pathway_id", "pathway_name"])
        for pid in pathways:
            writer.writerow([pid, pnames[pid]])
    print(f"Column index written → {cols_path}  ({len(pathways):,} pathways)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", default="../reactome_protein_pathway_human.tsv")
    parser.add_argument("--matrix",  default="../protein_pathway_matrix_uniprot.npz")
    parser.add_argument("--rows",    default="../protein_pathway_matrix_uniprot_rows.txt")
    parser.add_argument("--cols",    default="../protein_pathway_matrix_uniprot_cols.tsv")
    parser.add_argument("--min-size", type=int, default=MIN_PATHWAY_SIZE)
    args = parser.parse_args()

    proteins, pathways, pnames, pp_map = load_mapping(args.mapping, args.min_size)
    M = build_matrix(proteins, pathways, pp_map)
    write_outputs(M, proteins, pathways, pnames,
                  args.matrix, args.rows, args.cols)


if __name__ == "__main__":
    main()
