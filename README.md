# ShimQuant

A patch for `llama.cpp` that lets k-quants and i-quants apply to tensors whose first
dimension isn't divisible by 256, by zero-padding the row to the next multiple of 256 at
quantize time and slicing it back at inference time.

On models where the widths are close to a multiple of 256, this removes a hard floor. A
30B MoE that could not go below **4.70 bits per weight** at any label now runs at **3.07 bpw**
with *less* measured divergence from its own Q8 reference than the 4.70-bit file it replaces.

This is a research prototype. Read the [limitations](#limitations) before you get excited;
one of them is a model where the technique loses.

## The problem

k-quants and i-quants operate on 256-element superblocks, so they require
`tensor->ne[0] % 256 == 0`. When a model's width doesn't satisfy that, `llama-quantize`
substitutes a 32-block type instead — usually `IQ4_NL` at about 4.5 bpw — and keeps going.
The behaviour is intentional and has been in llama.cpp since
[PR #3747](https://github.com/ggml-org/llama.cpp/pull/3747) (2023), and it prints a warning.

The consequence is that on affected models the entire low-bit ladder collapses. For
NVIDIA-Nemotron-3.5-Lightning-30B-A3B (`n_embd` 2688, expert widths 1856/3712), all four IQ2
rungs — nominally 2.06 to 2.56 bpw — are the same 4.58 bpw file under four names. See
[ggufaudit](https://github.com/JoshBolding/ggufaudit) for a census of how widespread this is
across published GGUFs.

The floor is packaging, not physics. Nothing about those weights requires 4.5 bits. The
quantizer just can't address them.

## What the patch does

**At quantize time** (`src/llama-quant.cpp`): a tensor eligible for padding whose target type
needs 256-blocks gets `ne[0]` rounded up to the next multiple of 256, with the added columns
zeroed. The original width is written to GGUF metadata as `lattice.pad.orig_ne0.<tensor>`.
For 3D expert banks every expert is padded independently.

**At load time** (`src/llama-model-loader.cpp`, `src/llama-model.cpp`): the metadata is read
back so the model knows each padded tensor's true width.

**At inference** (`src/llama-graph.cpp`): activations entering a padded matmul are padded to
match, and the result is sliced back to the original width, so the graph is numerically
equivalent to the unpadded model apart from quantization error.

Padding is automatic for 256-block target types. `--pad-quant-force` widens eligible tensors
even when the target type doesn't require it, which is only useful for experiments.

540 lines across nine files, on top of upstream
[`e70802a01f`](https://github.com/ggml-org/llama.cpp/commit/e70802a01f03f0ed31a26338a5664796f3824371).

## Results

Nemotron-3.5-Lightning-30B-A3B. Every row is scored against **one** shared answer key: the
Q8_0 logits dumped once from the stock build, teacher-forced over a fixed corpus. KLD is mean
per-token KL divergence in nats, lower is closer to Q8.

| build | measured bpw | size | KLD |
|---|---:|---:|---:|
| stock IQ3_XXS | 4.70 | 18.01 GiB | 0.0398 |
| stock IQ2_M | 4.70 | 18.00 GiB | 0.2722 |
| **shimmed IQ3_XXS** | **3.58** | **13.71 GiB** | **0.2541** |
| **shimmed IQ2_M** | **3.13** | **12.01 GiB** | 0.5229 |
| **shimmed, tuned recipe** | **3.07** | **11.77 GiB** | **0.1230** |

The third row is the headline: **24% smaller than stock IQ2_M and less damaged**, which is a
strict improvement rather than a trade. Note that stock IQ3_XXS and stock IQ2_M are the same
4.70 bpw file size apart from rounding — that is the floor, visible.

The tuned recipe (Q6_K base, experts crushed, everything else protected) scored **94.0%
pass@1 on HumanEval** (47/50, greedy, executed tests) on a single 16 GB RTX 5080, a card the
stock file does not fit on at any label.

```bash
llama-quantize --imatrix nemotron.imatrix \
  --tensor-type "blk.52.=q8_0" \
  --tensor-type "ffn_(gate|up)_exps=iq2_xxs" \
  --tensor-type "ffn_down_exps=iq2_s" \
  model-BF16.gguf model-shim-tuned.gguf Q6_K 24
```

(`blk.52` is the MTP block: decode-only, so it gets no imatrix data, and `llama-quantize`
refuses to put a low-bit type on a tensor with no importance statistics. Pin it.)

## Limitations

**The payoff scales with how close the width already is to a multiple of 256.** Padding costs
you the zeros:

| original width | padded to | overhead | verdict |
|---:|---:|---:|---|
| 1856 | 2048 | 9.4% | excellent — this is the Nemotron expert case above |
| 640 | 768 | 16.7% | good |
| 320 or 160 | 512 / 256 | 37.5% | marginal, you may pay more in zeros than you win in bits |

**Embedding and lookup tables are deliberately never padded.** They are addressed by
`get_rows`, not matmul, and v1 is matmul-only. On models where an embedding table carries most
of the forced mass this caps the benefit hard.

**It does not always win.** On Qwen3.8-Flash-Next (177B, expert widths 640/320/160), padding
works mechanically — 572 expert tensors reached genuine `iq2_xxs`, which the stock quantizer
cannot produce — but the result was *worse* than the existing published file: 65.3 GB at KLD
0.548, against unsloth's UD-IQ1_S at 72.5 GB and 0.404. A more aggressive attempt was worse
still (60.3 GB, 0.815). Two reasons: 29% of that model is a width-160 per-layer embedding
table that padding won't touch, and the forced 4.5-bit floor on a third of its expert tensors
was partly *protecting* it. Removing a constraint is not the same as improving a model.

**Files built with this patch only load with this patch.** A padded GGUF carries metadata and
tensor shapes that stock llama.cpp does not understand.

**Tested on:** `nemotron_h_moe` (Nemotron-3.5-Lightning, Nemotron-3-Nano) and `qwen4exp`
(Qwen3.8-Flash-Next). Other architectures are unexercised.

## Build

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout e70802a01f03f0ed31a26338a5664796f3824371
git apply /path/to/shimquant.patch
cmake -B build -DGGML_CUDA=ON
cmake --build build -j
```

The base commit is pinned because **every number in this README was measured on a build of
it**. As of 2026-08-28 the patch also applies cleanly to upstream `master`
(`50f068fff`), verified with `git apply --check`, so you can skip the `git checkout` if you
prefer to be current — but then you are running a build I have not measured.

Then quantize as usual. Padding engages automatically whenever a target type needs 256-blocks
and the tensor width doesn't provide it; the log prints
`PadQuant will widen <tensor> from N to M columns` for each one.

## Tests

`tests/` contains the scaffolding used during development: toy dense and MoE model builders
with deliberately non-divisible widths, a GGUF tensor-type inspector, and a helper for
un-padding an imatrix so it can be reused across padded and unpadded builds.

## License

MIT, matching llama.cpp. See [LICENSE](LICENSE).

The patch is against llama.cpp, which is MIT-licensed and copyright its contributors.
