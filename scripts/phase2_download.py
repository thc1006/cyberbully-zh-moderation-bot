"""Phase 2 download CLI — defaults to dry-run.

Usage:
  python scripts/phase2_download.py                # dry-run, writes manifest
  python scripts/phase2_download.py --live         # actually download
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cyberpuppy.data.phase2_download import (Phase2Downloader,
                                              load_default_specs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", default="data/raw")
    ap.add_argument("--manifest", default="reports/phase2_download_manifest.json")
    ap.add_argument("--live", action="store_true",
                    help="Actually download (default: dry-run)")
    args = ap.parse_args()

    d = Phase2Downloader(target_dir=Path(args.target_dir), dry_run=not args.live)
    manifest = d.fetch_all(load_default_specs(), manifest_path=Path(args.manifest))
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nManifest written to {args.manifest}  (live={args.live})")


if __name__ == "__main__":
    main()
