"""
Downloads the external source data consumed by generate_Bressin19.py, generate_InterPro.py,
and generate_RIC.py into ${REPOSITORY}/data/data_original/. Run this once before the
"Build the harmonised raw tables" step in the pipeline (see README.md / pipeline.sh).

Sources:
- bressin19: TriPepSVM allData (https://github.com/marsicoLab/TriPepSVM/tree/master/allData)
- InterPro:  EBI InterPro API, reviewed/complete-sequence proteins per taxon, with go_terms
- RIC:       RBPbase descriptive-ID tables (https://apps.embl.de/rbpbase/), exported as .xlsx
             and converted to .tsv (the format generate_RIC.py reads)

NOTE ON REPRODUCIBILITY: InterPro and RBPbase are living databases that get updated over
time (new UniProt entries, new RIC studies added as extra columns). Re-running this script
months/years later will not byte-for-byte reproduce an older data_original/ snapshot - a few
proteins may be added/removed and RBPbase may gain extra annotation columns. It was verified
(2026-07-07) that a fresh download differs from the data_original/ used for prior experiments
by <0.1% of InterPro proteins and 0 mismatches in the RIC columns actually used by
generate_RIC.py, so it is a good-enough substitute, but not a cryptographic match. Bressin19
is a static GitHub file and reproduces byte-for-byte.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(".").absolute()))
from scripts.initialize import DATA_ORIGINAL  # noqa: E402 (reuses same REPOSITORY resolution as the rest of the pipeline)

import time
import requests
import pandas as pd

BRESSIN_URL_ROOT = "https://raw.githubusercontent.com/marsicoLab/TriPepSVM/master/allData"
BRESSIN_FILES = [
    "NRBP_9606.fasta",
    "RBP_9606.fasta",
    "NRBP_590.fasta",
    "RBP_590.fasta",
    "NRBP_561.fasta",
    "RBP_561.fasta",
]

INTERPRO_API_ROOT = "https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry/InterPro/taxonomy/uniprot"
INTERPRO_FILES = {  # output filename : taxon ID (must match generate_InterPro.py's InterProFiles)
    "HUMAN_9606_rev_complete_extra-go-terms.json": 9606,
    "MOUSE_10090_rev_complete_extra-go-terms.json": 10090,
    "DROS_7227_rev_complete_extra-go-terms.json": 7227,
    "ECOLI_83333_rev_complete_extra-go-terms.json": 83333,
    "ARATH_3702_rev_complete_extra-go-terms.json": 3702,
    "SAL_590_rev_complete_extra-go-terms.json": 590,
}

RBPBASE_URL_ROOT = "https://apps.embl.de/rbpbase/data"
RBPBASE_CODES = ["Hs", "Mm", "Sc", "Dm", "At"]  # must match generate_RIC.py's RICfiles


def download_file(url, outPath, chunkSize=1 << 20):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(outPath, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunkSize):
                f.write(chunk)


def get_json_with_retry(url, attempts=6, timeout=20, backoffSeconds=3):
    # the InterPro API occasionally stalls on an otherwise valid request; a plain retry
    # on a fresh connection reliably succeeds (observed empirically, 2026-07-07)
    for attempt in range(attempts):
        try:
            return requests.get(url, timeout=timeout).json()
        except requests.exceptions.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(backoffSeconds)


def get_bressin19(folderPath):
    folderPath.mkdir(parents=True, exist_ok=True)
    for fileName in BRESSIN_FILES:
        outPath = folderPath.joinpath(fileName)
        if outPath.exists():
            print(f"  skip {fileName} (already present)")
            continue
        download_file(f"{BRESSIN_URL_ROOT}/{fileName}", outPath)
        print(f"  downloaded {fileName}")


def get_interpro(folderPath):
    folderPath.mkdir(parents=True, exist_ok=True)
    for fileName, taxonID in INTERPRO_FILES.items():
        outPath = folderPath.joinpath(fileName)
        if outPath.exists():
            print(f"  skip {fileName} (already present)")
            continue

        proteins = []
        url = f"{INTERPRO_API_ROOT}/{taxonID}/?is_fragment=false&extra_fields=go_terms&page_size=200"
        page = 0
        while url is not None:
            response = get_json_with_retry(url)
            proteins.extend(response["results"])
            url = response["next"]
            page += 1
            if page % 10 == 0:
                print(f"    ...{fileName}: {len(proteins)} proteins so far (page {page})", flush=True)

        with open(outPath, "w") as f:
            import json

            json.dump(proteins, f)
        print(f"  downloaded {fileName} ({len(proteins)} proteins)")


def get_ric(folderPath):
    folderPath.mkdir(parents=True, exist_ok=True)
    for code in RBPBASE_CODES:
        tsvPath = folderPath.joinpath(f"RBPbase_{code}_DescriptiveID.tsv")
        if tsvPath.exists():
            print(f"  skip RBPbase_{code}_DescriptiveID.tsv (already present)")
            continue

        xlsxPath = folderPath.joinpath(f"RBPbase_{code}_DescriptiveID.xlsx")
        download_file(f"{RBPBASE_URL_ROOT}/RBPbase_{code}_DescriptiveID.xlsx", xlsxPath)
        pd.read_excel(xlsxPath).to_csv(tsvPath, sep="\t", index=False)
        xlsxPath.unlink()
        print(f"  downloaded RBPbase_{code}_DescriptiveID.tsv")


if __name__ == "__main__":
    print("[1/3] Bressin19 (TriPepSVM allData, GitHub)")
    get_bressin19(DATA_ORIGINAL.joinpath("bressin19"))

    print("[2/3] InterPro (EBI InterPro API export)")
    get_interpro(DATA_ORIGINAL.joinpath("InterPro"))

    print("[3/3] RIC (RBPbase descriptive-ID tables, EMBL)")
    get_ric(DATA_ORIGINAL.joinpath("RIC"))

    print(f"Done. Data staged under {DATA_ORIGINAL}")
