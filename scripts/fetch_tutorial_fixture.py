"""Fetch a bounded, patient-disjoint PTB-XL fixture for software validation.

The normal-only training subset and separate abnormal examples intentionally
exercise software interfaces. Their scores are not a clinical benchmark.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
from itertools import islice
import json
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

PTB = "https://physionet.org/files/ptb-xl/1.0.3/"
PLUS = "https://physionet.org/files/ptb-xl-plus/1.0.1/"


def csv_subset(url, target, count):
    with urlopen(url, timeout=90) as response:
        reader = csv.DictReader(io.TextIOWrapper(response, encoding="utf-8-sig"))
        rows = list(islice(reader, count))
        columns = reader.fieldnames
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urlopen(url, timeout=90) as response:
            data = response.read()
        partial.write_bytes(data)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def fetch(output, per_split=10, metadata_rows=100):
    output = Path(output).resolve()
    if per_split < 1 or metadata_rows < 2 * per_split:
        raise ValueError("Use a positive per-split count and at least twice as many metadata rows.")
    rows = csv_subset(PTB + "ptbxl_database.csv", output / "ptb-xl/ptbxl_database.csv", metadata_rows)
    train, heldout, patients = [], [], set()
    for row in rows:
        codes = ast.literal_eval(row["scp_codes"])
        group = train if "NORM" in codes else heldout
        if row["patient_id"] not in patients and len(group) < per_split:
            group.append(row)
            patients.add(row["patient_id"])
    if len(train) != per_split or len(heldout) != per_split:
        raise ValueError("Insufficient distinct patients. Increase --metadata-rows.")
    csv_subset(PLUS + "labels/12sl_statements.csv",
               output / "ptb-xl-plus/labels/12sl_statements.csv", metadata_rows)
    download(PTB + "scp_statements.csv", output / "ptb-xl/scp_statements.csv")
    download(PLUS + "labels/mapping/12slv23ToSNOMED.csv",
             output / "ptb-xl-plus/labels/mapping/12slv23ToSNOMED.csv")
    selected = train + heldout

    def recording(row):
        relative = PurePosixPath(row["filename_hr"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe record path in source metadata.")
        for suffix in (".hea", ".dat"):
            name = relative.as_posix() + suffix
            download(PTB + name, output / "ptb-xl" / name)
        print(f"Downloaded ECG {row['ecg_id']}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(recording, row) for row in selected]):
            future.result()
    manifest = {
        "purpose": "Software validation only. Not a clinical benchmark.",
        "source_versions": {"PTB-XL": "1.0.3", "PTB-XL+": "1.0.1"},
        "source_urls": [PTB, PLUS], "metadata_rows": metadata_rows,
        "training_ids": [int(row["ecg_id"]) for row in train],
        "heldout_ids": [int(row["ecg_id"]) for row in heldout],
        "training_records": [row["filename_hr"] for row in train],
        "heldout_records": [row["filename_hr"] for row in heldout],
        "patient_disjoint": not set(row["patient_id"] for row in train).intersection(
            row["patient_id"] for row in heldout),
        "sha256": {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(output.rglob("*")) if path.is_file() and path.name != "fixture_manifest.json"},
    }
    (output / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {2 * per_split} public records and fixture_manifest.json in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/tutorial")
    parser.add_argument("--per-split", type=int, default=10)
    parser.add_argument("--metadata-rows", type=int, default=100)
    args = parser.parse_args()
    fetch(args.output, args.per_split, args.metadata_rows)
