#!/usr/bin/env python3
"""Create a deterministic one-layer Llama GGUF with n_ff=320 for PadQuant tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import GGUFWriter, TokenType  # noqa: E402


def tensor(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    return (rng.standard_normal((rows, cols), dtype=np.float32) * np.float32(0.02)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    n_vocab = 259
    n_embd = 64
    n_ff = 320
    n_head = 4
    n_layer = 1
    rng = np.random.default_rng(0x50414451)

    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
    assert len(tokens) == n_vocab
    token_types = [TokenType.UNKNOWN, TokenType.CONTROL, TokenType.CONTROL] + [TokenType.BYTE] * 256

    writer = GGUFWriter(args.output, "llama")
    writer.add_name("PadQuant deterministic toy Llama")
    writer.add_context_length(128)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_ff)
    writer.add_rope_dimension_count(n_embd // n_head)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head)
    writer.add_layer_norm_rms_eps(1.0e-5)
    writer.add_rope_freq_base(10000.0)
    writer.add_file_type(0)

    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * n_vocab)
    writer.add_token_types(token_types)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)
    writer.add_add_space_prefix(True)

    # NumPy is row-major; gguf-py reverses these shapes to GGML [ne0, ne1].
    writer.add_tensor("token_embd.weight", tensor(rng, n_vocab, n_embd))
    writer.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))
    writer.add_tensor("output.weight", tensor(rng, n_vocab, n_embd))

    writer.add_tensor("blk.0.attn_norm.weight", np.ones(n_embd, dtype=np.float32))
    writer.add_tensor("blk.0.attn_q.weight", tensor(rng, n_embd, n_embd))
    writer.add_tensor("blk.0.attn_k.weight", tensor(rng, n_embd, n_embd))
    writer.add_tensor("blk.0.attn_v.weight", tensor(rng, n_embd, n_embd))
    writer.add_tensor("blk.0.attn_output.weight", tensor(rng, n_embd, n_embd))

    writer.add_tensor("blk.0.ffn_norm.weight", np.ones(n_embd, dtype=np.float32))
    writer.add_tensor("blk.0.ffn_gate.weight", tensor(rng, n_ff, n_embd))
    writer.add_tensor("blk.0.ffn_up.weight", tensor(rng, n_ff, n_embd))
    writer.add_tensor("blk.0.ffn_down.weight", tensor(rng, n_embd, n_ff))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"wrote {args.output}")
    print(f"architecture=llama layers={n_layer} n_embd={n_embd} n_ff={n_ff} vocab={n_vocab}")


if __name__ == "__main__":
    main()
