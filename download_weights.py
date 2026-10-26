"""Download pinned weights and check every file against its SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import shutil

MANIFEST = Path(__file__).resolve().parent / "weights_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(kind, output=None):
    specs = json.loads(MANIFEST.read_text())["artifacts"][kind]
    output = Path(output or specs["directory"])
    output.mkdir(parents=True, exist_ok=True)
    for item in specs["files"]:
        target = output / item["name"]
        if target.exists() and sha256(target) == item["sha256"]:
            print(f"Verified {target}")
            continue
        url = f"https://huggingface.co/{specs['repository']}/resolve/{specs['revision']}/{item['name']}"
        partial = target.with_suffix(target.suffix + ".part")
        print(f"Downloading {target}", flush=True)
        try:
            with urlopen(url, timeout=120) as response, partial.open("wb") as file:
                shutil.copyfileobj(response, file)
            if partial.stat().st_size != item["size"] or sha256(partial) != item["sha256"]:
                raise ValueError(f"Size or SHA-256 mismatch for {item['name']}")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    (output / "weights_manifest.json").write_text(json.dumps(specs, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["ensemble", "pretrained", "all"], default="ensemble")
    parser.add_argument("--output", help="Override destination for a single kind.")
    args = parser.parse_args()
    if args.kind == "all" and args.output:
        parser.error("--output requires --kind ensemble or --kind pretrained")
    for kind in (["ensemble", "pretrained"] if args.kind == "all" else [args.kind]):
        download(kind, args.output)


if __name__ == "__main__":
    main()
