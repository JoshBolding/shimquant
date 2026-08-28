#!/usr/bin/env python3
"""Create a UTF-8-safe byte-limited corpus excerpt for reproducible PPL tests."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--bytes", type=int, default=100_000)
    args = parser.parse_args()

    raw = args.source.read_bytes()[: args.bytes]
    text = raw.decode("utf-8", errors="ignore")
    args.destination.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.destination} ({args.destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
