"""
12_fi_interpathway_analysis.py
-------------------------------
Stratified topological analysis of Reactome Functional Interaction (FI) edges
by annotation type (inhibit, activate, catalyze, complex, expression, predicted).

For each FI edge, determines whether it is:
  intra-pathway  — both proteins share at least one Reactome pathway (n>=10)
  inter-pathway  — proteins belong to entirely different pathways

For inter-pathway FI edges, attaches per-protein topology features:
  local_degree    mean degree within each protein's own pathways
  global_degree   total BioGRID degree
  betweenness     approximate betweenness centrality
  ratio           mean(local/global degree)

Tests whether topology metrics differ across annotation classes using
pairwise Mann-Whitney U tests (BH FDR correction). Effect size is reported
as rank-biserial correlation (rbc).

Annotation class priority (for edges with multiple annotations):
  inhibit > activate > catalyze > expression > complex > predicted > other

Inputs:
    --fi        FIsInGene_04142025_with_annotations.txt
    --topology  protein_pathway_topology.tsv
    --classif   protein_classification.tsv
    --mapping   reactome_protein_pathway_human.tsv
    --cols      protein_pathway_matrix_uniprot_cols.tsv
Outputs:
    --out-edges   fi_interpathway_annotated.tsv
    --out-stats   fi_mw_results.tsv
"""

import argparse
import csv
import itertools
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


CLASSES = ["inhibit", "activate", "catalyze", "expression", "complex",
           "predicted", "other"]

METRICS = [
    "mean_local_degree",
    "mean_global_degree",
    "mean_betweenness",
    "ratio_g1",
    "ratio_g2",
]


def classify_annotation(annot: str) -> str:
    a = annot.lower()
    if "inhibit"    in a: return "inhibit"
    if "activat"    in a: return "activate"
    if "catalyz"    in a: return "catalyze"
    if "expression" in a: return "expression"
    if "complex"    in a: return "complex"
    if "predicted"  in a: return "predicted"
    return "other"


def load_fi(path: str) -> list:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append({
                "gene1":      row["Gene1"],
                "gene2":      row["Gene2"],
                "annotation": row["Annotation"],
                "direction":  row["Direction"],
                "score":      float(row["Score"]),
                "ann_class":  classify_annotation(row["Annotation"]),
            })
    return rows


def load_gene_pathways(mapping_path: str, n10_pathways: set) -> dict:
    gp: dict = defaultdict(set)
    with open(mapping_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pid = row["pathway_id"]
            if pid in n10_pathways:
                gp[row["gene_name"]].add(pid)
    return gp


def load_topology(topo_path: str) -> dict:
    """Returns dict[(gene_name, pathway_id) -> {local_degree, global_degree, ratio}]."""
    topo: dict = {}
    with open(topo_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row["gene_name"], row["pathway_id"])
            topo[key] = {
                "local_degree":  int(row["local_degree"]),
                "global_degree": int(row["global_degree"]),
                "ratio": (float(row["ratio"]) if row["ratio"] != "nan"
                          else float("nan")),
            }
    return topo


def load_classif(path: str) -> dict:
    """Returns dict[gene_name -> feature dict]."""
    c: dict = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            c[row["gene_name"]] = {
                "taxonomy":       row["taxonomy"],
                "global_degree":  int(row["global_degree"]),
                "betweenness":    float(row["betweenness"]),
                "shannon_H_norm": float(row["shannon_H_norm"]),
                "mean_ratio":     float(row["mean_ratio"]),
            }
    return c


def mean_topo(gene: str, pathways: set, topo: dict) -> tuple:
    """Return (mean_local_degree, mean_ratio) across all pathways of a gene."""
    lds, ratios = [], []
    for pw in pathways:
        t = topo.get((gene, pw))
        if t:
            lds.append(t["local_degree"])
            if t["ratio"] == t["ratio"]:
                ratios.append(t["ratio"])
    return (
        np.mean(lds) if lds else 0.0,
        np.mean(ratios) if ratios else float("nan"),
    )


def run(
    fi_path: str,
    topo_path: str,
    classif_path: str,
    mapping_path: str,
    cols_path: str,
    out_edges: str,
    out_stats: str,
) -> None:
    # Load n>=10 pathway set
    n10: set = set()
    with open(cols_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            n10.add(row["pathway_id"])

    print("Loading data ...")
    fi_edges   = load_fi(fi_path)
    gene_pw    = load_gene_pathways(mapping_path, n10)
    topo       = load_topology(topo_path)
    classif    = load_classif(classif_path)
    print(f"  FI edges: {len(fi_edges):,}  |  n>=10 pathways: {len(n10):,}")

    # Classify edges as intra / inter
    inter, intra, skipped = [], [], 0
    for e in fi_edges:
        g1, g2 = e["gene1"], e["gene2"]
        pw1 = gene_pw.get(g1, set())
        pw2 = gene_pw.get(g2, set())
        if not pw1 or not pw2:
            skipped += 1
            continue
        if pw1 & pw2:
            intra.append(e)
        else:
            inter.append(e)

    print(f"  Intra-pathway: {len(intra):,}  Inter-pathway: {len(inter):,}  "
          f"Skipped: {skipped:,}")

    # Annotate inter-pathway edges
    annotated = []
    for e in inter:
        g1, g2 = e["gene1"], e["gene2"]
        pw1 = gene_pw[g1]; pw2 = gene_pw[g2]
        ld1, r1 = mean_topo(g1, pw1, topo)
        ld2, r2 = mean_topo(g2, pw2, topo)
        c1 = classif.get(g1, {}); c2 = classif.get(g2, {})
        annotated.append({
            **e,
            "n_pathways_g1":     len(pw1),
            "n_pathways_g2":     len(pw2),
            "local_degree_g1":   round(ld1, 3),
            "local_degree_g2":   round(ld2, 3),
            "mean_local_degree": round((ld1 + ld2) / 2, 3),
            "ratio_g1":          round(r1, 4) if r1 == r1 else float("nan"),
            "ratio_g2":          round(r2, 4) if r2 == r2 else float("nan"),
            "global_degree_g1":  c1.get("global_degree", 0),
            "global_degree_g2":  c2.get("global_degree", 0),
            "mean_global_degree":
                (c1.get("global_degree", 0) + c2.get("global_degree", 0)) / 2,
            "betweenness_g1":    c1.get("betweenness", 0.0),
            "betweenness_g2":    c2.get("betweenness", 0.0),
            "mean_betweenness":
                (c1.get("betweenness", 0.0) + c2.get("betweenness", 0.0)) / 2,
            "taxonomy_g1":       c1.get("taxonomy", "unknown"),
            "taxonomy_g2":       c2.get("taxonomy", "unknown"),
        })

    with open(out_edges, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t",
                                fieldnames=annotated[0].keys())
        writer.writeheader()
        writer.writerows(annotated)
    print(f"Written: {out_edges}")

    # Mann-Whitney U tests
    class_data = {cls: [e for e in annotated if e["ann_class"] == cls]
                  for cls in CLASSES}
    print("\nInter-pathway edge counts per class:")
    for cls in CLASSES:
        print(f"  {cls:12s}: {len(class_data[cls]):,}")

    mw_results = []
    for metric in METRICS:
        for cls_a, cls_b in itertools.combinations(CLASSES, 2):
            vals_a = [e[metric] for e in class_data[cls_a]
                      if e[metric] == e[metric]]
            vals_b = [e[metric] for e in class_data[cls_b]
                      if e[metric] == e[metric]]
            if len(vals_a) < 5 or len(vals_b) < 5:
                continue
            stat, pval = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            n1, n2 = len(vals_a), len(vals_b)
            rbc = 1 - (2 * stat) / (n1 * n2)
            mw_results.append({
                "metric":        metric,
                "class_a":       cls_a,
                "class_b":       cls_b,
                "n_a":           n1,
                "n_b":           n2,
                "median_a":      round(np.median(vals_a), 4),
                "median_b":      round(np.median(vals_b), 4),
                "U_stat":        round(stat, 1),
                "pval":          pval,
                "rank_biserial": round(rbc, 4),
            })

    pvals = [r["pval"] for r in mw_results]
    _, padj, _, _ = multipletests(pvals, method="fdr_bh")
    for i, r in enumerate(mw_results):
        r["padj_bh"] = padj[i]

    with open(out_stats, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t",
                                fieldnames=mw_results[0].keys())
        writer.writeheader()
        writer.writerows(sorted(mw_results, key=lambda x: x["padj_bh"]))
    print(f"Written: {out_stats}")

    # Key comparison: inhibit vs activate
    print("\n=== inhibit vs activate (key hypothesis) ===")
    for metric in METRICS:
        r = next((x for x in mw_results
                  if x["metric"] == metric and
                  {x["class_a"], x["class_b"]} == {"inhibit", "activate"}), None)
        if not r:
            continue
        m_inh = r["median_a"] if r["class_a"] == "inhibit" else r["median_b"]
        m_act = r["median_b"] if r["class_b"] == "activate" else r["median_a"]
        print(f"  {metric:25s}: inhibit={m_inh:.4f}  activate={m_act:.4f}  "
              f"padj={r['padj_bh']:.3e}  rbc={r['rank_biserial']:.4f}")

    sig = [r for r in mw_results if r["padj_bh"] < 0.05]
    print(f"\nSignificant tests (FDR<0.05): {len(sig)} / {len(mw_results)}")
    print("\nTop 10 by |rbc|:")
    for r in sorted(sig, key=lambda x: -abs(x["rank_biserial"]))[:10]:
        print(f"  |rbc|={abs(r['rank_biserial']):.4f}  padj={r['padj_bh']:.2e}  "
              f"{r['metric']:25s}  {r['class_a']} vs {r['class_b']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fi",       default="../FIsInGene_04142025_with_annotations.txt")
    parser.add_argument("--topology", default="../protein_pathway_topology.tsv")
    parser.add_argument("--classif",  default="../protein_classification.tsv")
    parser.add_argument("--mapping",  default="../reactome_protein_pathway_human.tsv")
    parser.add_argument("--cols",     default="../protein_pathway_matrix_uniprot_cols.tsv")
    parser.add_argument("--out-edges",default="../fi_interpathway_annotated.tsv")
    parser.add_argument("--out-stats",default="../fi_mw_results.tsv")
    args = parser.parse_args()

    run(args.fi, args.topology, args.classif, args.mapping, args.cols,
        args.out_edges, args.out_stats)


if __name__ == "__main__":
    main()
