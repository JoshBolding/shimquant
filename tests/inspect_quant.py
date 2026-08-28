#!/usr/bin/env python3
"""Summarize quant types and PadQuant widths without loading tensor payloads."""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import GGUFReader  # noqa: E402


PAD_PREFIX = "lattice.pad.orig_ne0."
EXPERT_SUFFIXES = ("ffn_down_exps.weight", "ffn_up_exps.weight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", type=Path, nargs="+")
    args = parser.parse_args()

    for model in args.models:
        reader = GGUFReader(model, mode="r")
        type_counts = collections.Counter(t.tensor_type.name for t in reader.tensors)
        pad_widths = {
            field.name[len(PAD_PREFIX) :]: int(field.contents())
            for field in reader.fields.values()
            if field.name.startswith(PAD_PREFIX)
        }
        experts = [t for t in reader.tensors if t.name.endswith(EXPERT_SUFFIXES)]
        expert_counts = collections.Counter(
            (
                "ffn_down_exps" if t.name.endswith("ffn_down_exps.weight") else "ffn_up_exps",
                t.tensor_type.name,
                int(t.shape[0]),
                pad_widths.get(t.name),
            )
            for t in experts
        )

        print(model)
        print(f"  tensors={len(reader.tensors)} pad_keys={len(pad_widths)}")
        print(f"  types={dict(sorted(type_counts.items()))}")
        print("  expert_groups:")
        for key, count in sorted(expert_counts.items()):
            family, quant_type, physical_ne0, original_ne0 = key
            print(
                f"    {family}: count={count} type={quant_type} "
                f"physical_ne0={physical_ne0} original_ne0={original_ne0}"
            )


if __name__ == "__main__":
    main()
