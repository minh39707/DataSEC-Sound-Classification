"""Download the pinned DataSEC release (Zenodo DOI 10.5281/zenodo.17033970), verify MD5, and extract.

Downloads in parallel byte-range chunks (resumable: finished chunks are skipped on re-run).

Usage:
    python scripts/download_data.py --out data/raw [--workers 8] [--no-extract]
"""
import argparse
import hashlib
import json
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RECORD_ID = "17033970"
DOI = "10.5281/zenodo.17033970"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
CHUNK = 64 * 1024 * 1024


def md5sum(path: Path, block: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(block):
            h.update(b)
    return h.hexdigest()


def fetch_chunk(url: str, start: int, end: int, part: Path, retries: int = 30) -> int:
    """Download bytes [start, end] into `part`, resuming a partial part file."""
    want = end - start + 1
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return want
        if have > want:
            part.unlink()
            have = 0
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "ab") as f:
                while b := resp.read(1 << 18):
                    f.write(b)
        except Exception as e:  # network hiccup -> back off and resume
            print(f"\n[retry {attempt + 1}] {part.name}: {e}", flush=True)
            time.sleep(min(5 * (attempt + 1), 60))
    raise RuntimeError(f"Failed to download {part.name}")


def parallel_download(url: str, dest: Path, size: int, workers: int) -> None:
    if dest.exists() and dest.stat().st_size == size:
        print(f"[skip] {dest.name} already assembled")
        return
    parts_dir = dest.with_suffix(dest.suffix + ".parts")
    parts_dir.mkdir(exist_ok=True)
    ranges = [(i, s, min(s + CHUNK, size) - 1) for i, s in enumerate(range(0, size, CHUNK))]
    t0, done = time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch_chunk, url, s, e, parts_dir / f"{i:05d}.part") for i, s, e in ranges]
        for n, fut in enumerate(as_completed(futs), 1):
            done += fut.result()
            rate = done / max(time.time() - t0, 1e-6) / 1e6
            print(f"chunks {n}/{len(ranges)}  ~{rate:.2f} MB/s (this session)", flush=True)

    print(f"Assembling {dest.name} ...", flush=True)
    with open(dest, "wb") as out:
        for i, _, _ in ranges:
            with open(parts_dir / f"{i:05d}.part", "rb") as f:
                while b := f.read(1 << 22):
                    out.write(b)
    assert dest.stat().st_size == size, "assembled size mismatch"
    for p in parts_dir.iterdir():
        p.unlink()
    parts_dir.rmdir()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(API_URL, timeout=60) as r:
        record = json.load(r)
    assert record["doi"] == DOI, f"Unexpected DOI {record['doi']}"
    (out / "zenodo_record.json").write_text(json.dumps(record, indent=2))

    for f in record["files"]:
        dest = out / f["key"]
        algo, expected_md5 = f["checksum"].split(":")
        assert algo == "md5"
        parallel_download(f["links"]["self"], dest, f["size"], args.workers)

        print(f"Verifying MD5 of {dest.name} ...", flush=True)
        got = md5sum(dest)
        if got != expected_md5:
            raise SystemExit(f"MD5 mismatch for {dest.name}: got {got}, expected {expected_md5}")
        print(f"[ok] md5 {got}", flush=True)

        if not args.no_extract and dest.suffix == ".zip":
            print(f"Extracting {dest.name} ...", flush=True)
            with zipfile.ZipFile(dest) as z:
                z.extractall(out)
            print("[ok] extracted", flush=True)


if __name__ == "__main__":
    main()
