"""
01_build_reactome_mapping.py
----------------------------
Downloads the Reactome human protein-pathway mapping (all hierarchy levels)
and writes a TSV with one row per (protein, pathway) pair.

Inputs:  none (downloads from Reactome v96)
Outputs: ../reactome_protein_pathway_human.tsv

Columns:
    gene_name       primary HGNC symbol (from UniProt)
    uniprot_id      UniProt base accession (isoform suffix stripped)
    pathway_id      Reactome stable ID (R-HSA-...)
    pathway_name    Reactome pathway display name
"""

import requests
import csv
import time
from collections import defaultdict

REACTOME_URL = (
    "https://reactome.org/download/current/UniProt2Reactome_All_Levels.txt"
)
UNIPROT_IDMAP_RUN = "https://rest.uniprot.org/idmapping/run"
UNIPROT_IDMAP_RESULTS = "https://rest.uniprot.org/idmapping/uniprotkb/results/{job_id}"
BATCH_SIZE = 500
OUT_PATH = "../reactome_protein_pathway_human.tsv"


def download_human_rows(url: str) -> tuple[dict, set]:
    """Download Reactome mapping and return human rows.

    Returns
    -------
    pathway_proteins : dict[pathway_id -> set of base UniProt accessions]
    uniprot_ids      : set of all base UniProt accessions
    """
    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()

    pathway_proteins: dict = defaultdict(set)
    pathway_names: dict = {}
    uniprot_ids: set = set()

    for line in r.iter_lines():
        decoded = line.decode("utf-8")
        if not decoded.endswith("Homo sapiens"):
            continue
        parts = decoded.split("\t")
        if len(parts) < 4:
            continue
        raw_acc, pathway_id, _, pathway_name = (
            parts[0], parts[1], parts[2], parts[3]
        )
        base_acc = raw_acc.split("-")[0]
        pathway_proteins[pathway_id].add(base_acc)
        pathway_names[pathway_id] = pathway_name
        uniprot_ids.add(base_acc)

    print(
        f"  {len(uniprot_ids):,} unique proteins across "
        f"{len(pathway_proteins):,} human pathways"
    )
    return pathway_proteins, pathway_names, uniprot_ids


def fetch_gene_names(uniprot_ids: set, batch_size: int = BATCH_SIZE) -> dict:
    """Map UniProt accessions to primary gene symbols via UniProt ID mapping API.

    Returns
    -------
    dict[accession -> primary_gene_symbol]
    """
    id_list = sorted(uniprot_ids)
    gene_map: dict = {}
    n_batches = (len(id_list) + batch_size - 1) // batch_size

    for i in range(0, len(id_list), batch_size):
        batch = id_list[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Gene name batch {batch_num}/{n_batches} ({len(batch)} IDs) ...",
              end=" ", flush=True)

        # Submit job
        r = requests.post(
            UNIPROT_IDMAP_RUN,
            data={"from": "UniProtKB_AC-ID", "to": "UniProtKB", "ids": ",".join(batch)},
            timeout=30,
        )
        r.raise_for_status()
        job_id = r.json()["jobId"]

        # Poll until results endpoint is ready
        url = UNIPROT_IDMAP_RESULTS.format(job_id=job_id)
        for _ in range(30):
            time.sleep(2)
            res = requests.get(
                url,
                params={"fields": "accession,gene_names", "format": "tsv", "size": 500},
                timeout=60,
            )
            if res.status_code == 200:
                break

        # Parse TSV (columns: From | Entry | Gene Names)
        for line in res.text.strip().split("\n")[1:]:
            parts = line.split("\t")
            if len(parts) >= 3:
                acc = parts[0].strip()
                genes = parts[2].strip()
                gene_map[acc] = genes.split()[0] if genes else acc

        print(f"got {len(gene_map)} cumulative")

    return gene_map


def write_mapping(
    pathway_proteins: dict,
    pathway_names: dict,
    gene_map: dict,
    out_path: str,
) -> None:
    rows_written = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["gene_name", "uniprot_id", "pathway_id", "pathway_name"])
        for pid in sorted(pathway_proteins):
            pname = pathway_names[pid]
            for acc in sorted(pathway_proteins[pid]):
                gene = gene_map.get(acc, acc)
                writer.writerow([gene, acc, pid, pname])
                rows_written += 1
    print(f"Written {rows_written:,} rows → {out_path}")


if __name__ == "__main__":
    pathway_proteins, pathway_names, uniprot_ids = download_human_rows(REACTOME_URL)
    print("\nFetching gene names from UniProt ...")
    gene_map = fetch_gene_names(uniprot_ids)
    print(f"\nGene names resolved: {len(gene_map):,} / {len(uniprot_ids):,}")
    write_mapping(pathway_proteins, pathway_names, gene_map, OUT_PATH)
