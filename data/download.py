#!/usr/bin/env python3
"""Restore a chromosome BAM and its index into data/<platform>/ (Python 3.9+)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
BUFFER = 8 * 1024 * 1024


def matches(path, size, expected):
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(BUFFER), b""):
            digest.update(block)
    return digest.hexdigest() == expected


def fetch(item, target):
    """Retain verified downloads; replace a file only after verification."""
    if matches(target, item["bytes"], item["sha256"]):
        return
    temporary = target.with_name(target.name + ".partial")
    for attempt in range(3):
        try:
            print(f"Downloading {item['name']}...", flush=True)
            request = urllib.request.Request(
                item["url"], headers={"User-Agent": "QHap-data-downloader"})
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, BUFFER)
            if not matches(temporary, item["bytes"], item["sha256"]):
                raise ValueError(f"Checksum or size mismatch: {item['name']}")
            temporary.replace(target)
            return
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2)


def restore(record, output, keep_parts=False):
    """Join the manifest's ordered parts and check the complete file."""
    target = output / record["file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if matches(target, record["bytes"], record["sha256"]):
        print(f"Already verified: {target}", flush=True)
        return target
    cache = target.parent / ".downloads"
    cache.mkdir(exist_ok=True)
    pieces = []
    for part in record["parts"]:
        piece = cache / part["name"]
        fetch(part, piece)
        pieces.append(piece)
    temporary = target.with_name(target.name + ".restoring")
    if len(pieces) == 1 and not keep_parts:
        # Reusing the verified complete download avoids another full BAM copy.
        if not matches(pieces[0], record["bytes"], record["sha256"]):
            raise ValueError(f"Complete-file checksum mismatch: {record['file']}")
        pieces[0].replace(target)
    else:
        with temporary.open("wb") as destination:
            for piece in pieces:
                with piece.open("rb") as source:
                    shutil.copyfileobj(source, destination, BUFFER)
        if not matches(temporary, record["bytes"], record["sha256"]):
            raise ValueError(f"Restored checksum mismatch: {record['file']}")
        temporary.replace(target)
        if not keep_parts:
            for piece in pieces:
                piece.unlink()
    print(f"Verified: {target}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("platform", choices=("WY", "HiFi", "ONT"))
    parser.add_argument("chromosome", choices=("chr6", "chr22"))
    parser.add_argument("--output", type=Path, default=HERE,
                        help="Data directory (default: this script's directory)")
    parser.add_argument("--keep-parts", action="store_true",
                        help="Keep downloaded parts after successful restoration")
    args = parser.parse_args()
    manifest = json.loads((HERE / "alignments.json").read_text(encoding="utf-8"))
    selected = [r for r in manifest
                if r["platform"] == args.platform and r["chromosome"] == args.chromosome]
    expected = {f"{args.platform}/{args.chromosome}.bam",
                f"{args.platform}/{args.chromosome}.bam.bai"}
    if len(selected) != 2 or {r["file"] for r in selected} != expected:
        raise ValueError("Expected exactly one BAM and its matching BAI")
    for record in sorted(selected, key=lambda r: r["bytes"]):
        restore(record, args.output.resolve(), args.keep_parts)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"Download failed: {error}")
