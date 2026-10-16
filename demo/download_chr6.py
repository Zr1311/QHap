#!/usr/bin/env python3
"""Download, verify and restore the chromosome 6 demo BAM using Python 3.9+."""

import argparse
import hashlib
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.request


BASE_URL = 'https://github.com/Zr1311/QHap/releases/download/demo-chr6-v1'
DEMO_DIR = Path(__file__).resolve().parent
PARTS = tuple(f'chr6.bam.part{i:02d}' for i in range(1, 4))
BUFFER_SIZE = 8 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            block = stream.read(BUFFER_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_checksums():
    checksums = {}
    for line in (DEMO_DIR / 'SHA256SUMS').read_text(encoding='ascii').splitlines():
        if not line.strip():
            continue
        digest, name = line.split()
        if not re.fullmatch(r'[0-9a-f]{64}', digest) or name in checksums:
            raise ValueError('Invalid SHA256SUMS entry')
        checksums[name] = digest
    if set(checksums) != {'chr6.bam', 'chr6.bam.bai', *PARTS}:
        raise ValueError('SHA256SUMS does not list the expected demo files')
    return checksums


def matches(path, expected):
    return path.is_file() and sha256(path) == expected


def download(name, expected):
    destination = DEMO_DIR / name
    if matches(destination, expected):
        print(f'Already verified: {name}', flush=True)
        return
    temporary = destination.with_name(name + '.partial')
    for attempt in range(1, 4):
        try:
            print(f'Downloading {name} (attempt {attempt}/3)...', flush=True)
            request = urllib.request.Request(
                f'{BASE_URL}/{name}', headers={'User-Agent': 'QHap-chr6-demo'})
            digest = hashlib.sha256()
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary.open('wb') as output:
                    while True:
                        block = response.read(BUFFER_SIZE)
                        if not block:
                            break
                        output.write(block)
                        digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f'SHA-256 mismatch: {name}')
            temporary.replace(destination)
            print(f'Verified: {name}', flush=True)
            return
        except (OSError, ValueError) as error:
            if attempt == 3:
                raise
            print(f'{error}; retrying...', flush=True)
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keep-parts', action='store_true',
                        help='Keep downloaded BAM parts after successful restoration')
    args = parser.parse_args()
    checksums = read_checksums()
    download('chr6.bam.bai', checksums['chr6.bam.bai'])
    bam = DEMO_DIR / 'chr6.bam'
    if not matches(bam, checksums['chr6.bam']):
        for name in PARTS:
            download(name, checksums[name])
        temporary = DEMO_DIR / 'chr6.bam.partial'
        print('Restoring chr6.bam...', flush=True)
        with temporary.open('wb') as output:
            for name in PARTS:
                with (DEMO_DIR / name).open('rb') as part:
                    shutil.copyfileobj(part, output, BUFFER_SIZE)
        if sha256(temporary) != checksums['chr6.bam']:
            raise ValueError('SHA-256 mismatch after restoring chr6.bam')
        temporary.replace(bam)
    if not args.keep_parts:
        for name in PARTS:
            (DEMO_DIR / name).unlink(missing_ok=True)
    print('Ready: demo/chr6.bam and demo/chr6.bam.bai (SHA-256 verified).')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f'Demo download failed: {error}')
