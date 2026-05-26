"""
07_interpret_density.py
------------------------
Produces a structured interpretation of intra-pathway PPI density results:
  - Density distribution by pathway size bin
  - High-density pathways (d > 0.5, n >= 10) grouped by functional cluster
  - Low-density and zero-density pathways with coverage context
  - Size-corrected significant pathways with low raw density (hidden signal)

Inputs:
    --density   pathway_ppi_density_corrected.tsv  (from script 04)
Outputs:
    --output    pathway_density_interpretation.tsv
                (per-pathway table with size_bin, functional_cluster columns added)
    Prints a human-readable summary to stdout.
"""

import argparse
import csv
from collections import defaultdict

import numpy as np


# ── Functional cluster tagger ────────────────────────────────────────────────

def tag_cluster(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ["baf", "swi/snf", "chromatin remodel"]):
        return "BAF/SWI-SNF"
    if any(x in n for x in ["proteasom", "ubiquitin", "scf", "cop1", "fbxl",
                              "btrc", "cul1", "emi1", "cyclin d", "nfe2",
                              "pak-2", "vpu", "aurka", "aurkb", "cdh1",
                              "apc/c", "skp"]):
        return "UPS/proteasome"
    if any(x in n for x in ["ribos", "translat", "elongat", "termination",
                              "nmd", "43s", "60s", "40s", "eif", "pelo",
                              "hbs1", "mrna quality"]):
        return "Translation/ribosome"
    if "rna pol iii" in n:
        return "RNA Pol III"
    if any(x in n for x in ["cct", "tric", "actin fold", "chaperonin",
                              "prefoldin", "tubulin fold"]):
        return "Chaperonin/CCT"
    if any(x in n for x in ["casp", "flip", "apoptosis", "bax", "bak",
                              "disc", "procaspase", "necropt", "ripk"]):
        return "Apoptosis/caspase"
    if any(x in n for x in ["dna rep", "lagging", "flap", "okazaki",
                              "pcna", "replication fork", "unwinding",
                              "cohesin", "pre-rc", "pre-replicat"]):
        return "DNA replication/repair"
    if any(x in n for x in ["fgfr", "egfr", "erbb", "mapk", "ras ",
                              "pi3k", "akt", "mtor", "raf"]):
        return "RTK/MAPK/PI3K"
    if any(x in n for x in ["antigen", "mhc", "complement", "antibod",
                              "fcgr", "phagocyt"]):
        return "Immune/antigen"
    if any(x in n for x in ["auf1", "hnrnp", "mrna stab", "mrna decay",
                              "deadenyl", "ksrp", "tristetraprolin"]):
        return "mRNA stability/decay"
    if any(x in n for x in ["rna pol ii", "pol ii"]):
        return "RNA Pol II"
    if any(x in n for x in ["gpcr", "g protein", "g alpha", "g beta",
                              "adenylate", "camp", "pka", "plc"]):
        return "GPCR/cAMP/Ca2+"
    if any(x in n for x in ["wnt", "beta-catenin", "ctnnb", "axin", "apc"]):
        return "WNT/beta-catenin"
    if any(x in n for x in ["notch"]):
        return "NOTCH"
    if any(x in n for x in ["mitochondr"]):
        return "Mitochondrial"
    return "Other"


def size_bin(n: int) -> str:
    if n <= 20:  return "10-20"
    if n <= 50:  return "21-50"
    if n <= 100: return "51-100"
    return ">100"


# ── Main ─────────────────────────────────────────────────────────────────────

def interpret(density_path: str, out_path: str) -> None:
    rows = []
    with open(density_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row["n_proteins"]          = int(row["n_proteins"])
            row["n_intra_ppi"]         = int(row["n_intra_ppi"])
            row["n_possible_pairs"]    = int(row["n_possible_pairs"])
            row["density"]             = (float(row["density"])
                                          if row["density"] != "nan" else float("nan"))
            row["padj_density"]        = float(row["padj_density"])
            row["frac_missing_biogrid"]= float(row["frac_missing_biogrid"])
            row["size_bin"]            = size_bin(row["n_proteins"])
            row["functional_cluster"]  = tag_cluster(row["pathway_name"])
            rows.append(row)

    valid = [r for r in rows if r["density"] == r["density"]]

    # ── Distribution by size bin ─────────────────────────────────────────────
    print("=== Density distribution by size bin ===")
    bins: dict = defaultdict(list)
    for r in valid:
        bins[r["size_bin"]].append(r["density"])
    for label in ["10-20", "21-50", "51-100", ">100"]:
        ds = bins[label]
        if not ds: continue
        ds_s = sorted(ds)
        print(f"  n={label:6s}  count={len(ds):4d}  "
              f"mean={np.mean(ds):.3f}  median={np.median(ds):.3f}  "
              f"d>0.5={sum(1 for d in ds if d>0.5):3d}  "
              f"d=0={sum(1 for d in ds if d==0):2d}")

    # ── Significant pathways ─────────────────────────────────────────────────
    sig = [r for r in valid if r["padj_density"] < 0.05
           and r["low_coverage_flag"] == "False"]
    print(f"\nSignificant pathways (FDR < 0.05, excl. low-coverage): "
          f"{len(sig)} / {len(valid)}")

    # ── High-density tier ────────────────────────────────────────────────────
    high = [r for r in valid if r["density"] > 0.5]
    print(f"\n=== High-density pathways (d > 0.5): {len(high)} ===")
    clusters: dict = defaultdict(list)
    for r in high:
        clusters[r["functional_cluster"]].append(r)
    for cluster, rs in sorted(clusters.items(), key=lambda x: -len(x[1])):
        ds = [r["density"] for r in rs]
        ns = [r["n_proteins"] for r in rs]
        print(f"\n  {cluster} ({len(rs)} pathways, "
              f"d={min(ds):.3f}–{max(ds):.3f}, n={min(ns)}–{max(ns)})")
        for r in sorted(rs, key=lambda x: -x["density"])[:5]:
            print(f"    d={r['density']:.3f}  n={r['n_proteins']:3d}  "
                  f"{r['pathway_name'][:60]}")

    # ── Hidden signal: large pathways significant but low raw density ─────────
    hidden = [r for r in sig if r["density"] < 0.2 and r["n_proteins"] >= 50]
    print(f"\n=== Large pathways (n>=50) significant but d<0.2 (hidden signal): "
          f"{len(hidden)} ===")
    for r in sorted(hidden, key=lambda x: x["padj_density"])[:10]:
        print(f"  padj={r['padj_density']:.1e}  d={r['density']:.3f}  "
              f"n={r['n_proteins']:4d}  {r['pathway_name'][:60]}")

    # ── Low coverage ─────────────────────────────────────────────────────────
    low_cov = [r for r in rows if r["low_coverage_flag"] == "True"]
    print(f"\n=== Low-coverage pathways (>50% missing from BioGRID): "
          f"{len(low_cov)} ===")
    for r in sorted(low_cov, key=lambda x: -x["n_proteins"])[:10]:
        print(f"  n={r['n_proteins']:4d}  miss={r['frac_missing_biogrid']:.2f}  "
              f"{r['pathway_name'][:60]}")

    # ── Write annotated output ────────────────────────────────────────────────
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: x["padj_density"]))
    print(f"\nAnnotated table written → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", default="../pathway_ppi_density_corrected.tsv")
    parser.add_argument("--output",  default="../pathway_density_interpretation.tsv")
    args = parser.parse_args()
    interpret(args.density, args.output)


if __name__ == "__main__":
    main()
