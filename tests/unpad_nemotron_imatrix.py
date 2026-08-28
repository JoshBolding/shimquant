#!/usr/bin/env python3
"""Restore the original Nemotron widths in a pre-padded imatrix GGUF.

PadQuant consumes the original-width imatrix and performs its own zero padding,
which is the path this experiment is intended to verify.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402


ORIGINAL_WIDTHS = {
    "ffn_down_exps.weight.in_sum2": 1856,
    "ffn_down_shexp.weight.in_sum2": 3712,
    "ffn_up_exps.weight.in_sum2": 2688,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    reader = GGUFReader(args.input)
    writer = GGUFWriter(args.output, "imatrix")

    for field in reader.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)

    tensors: list[tuple[str, np.ndarray, object]] = []
    changed = 0
    for source in reader.tensors:
        data = np.array(source.data)
        for suffix, width in ORIGINAL_WIDTHS.items():
            if source.name.endswith(suffix):
                if data.shape[-1] < width:
                    raise ValueError(f"{source.name}: width {data.shape[-1]} is smaller than {width}")
                if data.shape[-1] > width:
                    print(f"{source.name}: {data.shape[-1]} -> {width}")
                    data = data[..., :width].copy()
                    changed += 1
                break
        tensors.append((source.name, data, source.tensor_type))

    if changed == 0:
        raise ValueError("no pre-padded Nemotron imatrix tensors were found")

    for name, data, tensor_type in tensors:
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, data, _ in tensors:
        writer.write_tensor_data(data)
    writer.close()

    print(f"wrote {args.output} with {changed} tensors restored to original widths")


if __name__ == "__main__":
    main()
