"""Download the Quick, Draw! simplified corpus.

    python download_quickdraw.py --out_dir qd_raw

345 category files from Google Cloud Storage, about 22 GB in total, CC-BY-4.0.
Downloads run in parallel and skip files already present, so an interrupted run
resumes by being run again. Pass --categories to fetch a subset.
"""

import argparse
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
LIST = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"


def _get(url, dest=None):
    """Fetch a URL, falling back to curl where Python's SSL roots are missing."""
    try:
        if dest:
            urllib.request.urlretrieve(url, dest)
            return None
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read().decode()
    except Exception:
        import subprocess
        cmd = ["curl", "-fsSL", url] + (["-o", dest] if dest else [])
        out = subprocess.run(cmd, capture_output=not dest, check=True)
        return None if dest else out.stdout.decode()


def category_names():
    return [line.strip() for line in _get(LIST).splitlines() if line.strip()]


def fetch(name, out_dir, retries=3):
    """Download one category, leaving an existing complete file alone."""
    path = os.path.join(out_dir, f"{name}.ndjson")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return name, os.path.getsize(path), "cached"
    url = BASE + urllib.parse.quote(name) + ".ndjson"
    partial = path + ".part"
    for attempt in range(retries):
        try:
            _get(url, partial)
            os.replace(partial, path)          # only a complete file gets the real name
            return name, os.path.getsize(path), "ok"
        except Exception as e:
            if attempt == retries - 1:
                if os.path.exists(partial):
                    os.remove(partial)
                return name, 0, f"failed: {e}"
    return name, 0, "failed"


def main():
    p = argparse.ArgumentParser(description="Download Quick, Draw!")
    p.add_argument("--out_dir", type=str, default="qd_raw")
    p.add_argument("--categories", type=str, default="",
                   help="comma-separated subset; default is all 345")
    p.add_argument("--threads", type=int, default=16)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    names = ([c.strip() for c in args.categories.split(",") if c.strip()]
             or category_names())
    print(f"{len(names)} categories -> {args.out_dir}")

    done = failed = total = 0
    with ThreadPoolExecutor(args.threads) as pool:
        for name, size, status in pool.map(lambda n: fetch(n, args.out_dir), names):
            done += 1
            total += size
            if status.startswith("failed"):
                failed += 1
                print(f"[{done}/{len(names)}] {name:28s} {status}", flush=True)
            elif done % 25 == 0 or done == len(names):
                print(f"[{done}/{len(names)}] {total / 1e9:5.1f} GB", flush=True)

    print(f"\n{done - failed} files, {total / 1e9:.1f} GB in {args.out_dir}")
    if failed:
        print(f"{failed} failed; run again to retry just those")


if __name__ == "__main__":
    main()
