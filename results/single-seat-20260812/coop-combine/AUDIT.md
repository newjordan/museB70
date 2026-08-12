# Cooperative PB32 combine proposal (design only)

Date: 2026-08-12

Source inspected: `/home/frosty40/turbo/worktrees/muse-serve`, commit
`c31785423`, including the in-flight decode-mask-bounds edits visible at review
time.  No repository source, build output, process, or GPU state was changed by
this task.

Proposed artifact: `coop-combine-pb32.apply_patch`

## Scope and gate

The proposal adds a second, opt-in combine kernel selected only when all of the
following are true:

- `GGML_SYCL_LX_FATTN_COOP_COMBINE_PB32` is nonempty and does not start with
  `0`;
- the compile-time value dimension is exactly `DV == 128`;
- runtime `parallel_blocks == 32`; and
- `Q->ne[1] == 1` (decode, not prefill).

Every other case submits the existing `flash_attn_combine_results<DV>` lambda
with its existing byte local accessor and arguments.  The generic device
function is not edited.  In particular, merely compiling this experiment does
not route gate-off requests through the new kernel.

The intended paired setting for a measurement is therefore:

```text
GGML_SYCL_LX_FATTN_PARALLEL_BLOCKS=32
GGML_SYCL_LX_FATTN_COOP_COMBINE_PB32=1
```

If the KV length clamps split-K below 32, the cooperative kernel is not used.

## What it changes

At D=128/PB32 the current kernel gives one output dimension to each of 128
work-items.  Every work-item independently scans all 32 metadata entries,
recomputes the same maximum, and evaluates the same 32 exponentials.  The
proposal does the common work once per work-group:

1. Work-item 0 computes `kqmax` in the exact stock order, `l = 0..31`, then a
   work-group broadcast distributes that value.
2. Work-items 0..31 each compute one scale and store that scale plus the
   corresponding rowsum in two non-overlapping local arrays.
3. A local-memory barrier publishes both arrays.
4. All 128 work-items retain the stock `l = 0..31` numerator and denominator
   loops for their output dimension.

This is deliberately the conservative first experiment.  It removes redundant
max/exp work while leaving the coalesced `VKQ_parts[l*128 + tid]` access and the
per-output accumulation order intact.  A scheme that assigns split-K lanes to
one output coordinate shortens the dependency chain, but makes each subgroup
read `VKQ_parts` with a 512-byte stride; staging the whole 16 KiB partial matrix
would add another local write/read and barrier.  That is not a good first patch.

## SYCL legality and synchronization audit

- The launch remains one 128-work-item group per output vector and retains
  `[[sycl::reqd_sub_group_size(warp_size)]]` (16 on the VEC path).
- `sycl::group_broadcast(item_ct1.get_group(), kqmax)` is a work-group
  collective, not a subgroup collective.  Every work-item reaches it on a
  converged path; only the producer computation is conditional.
- Every work-item also reaches the local-memory barrier.  There are no early
  returns or divergent barriers.
- The new submission uses `sycl::local_accessor<float, 1>` with 64 floats (256
  bytes).  This avoids depending on the alignment of a byte accessor cast to a
  `float *`.
- `scales` covers floats `[0, 32)` and `rowsums` covers `[32, 64)`; the regions
  do not alias.  Each element has one writer, and the explicit local-space
  barrier happens before any reader.
- The helper performs no cross-work-group communication.  Producer FA kernels
  and the combine kernel remain ordered through the same queue/submission path
  as the stock implementation.
- The default kernel and its existing local-memory alias are untouched.  The
  proposal does not attempt to make claims about that pre-existing byte-to-
  `float2` cast.

## Empty partials and edge behavior

The VEC kernel initializes an empty split partial to approximately
`(-FLT_MAX/2, 0)` with a zero numerator.  With at least one nonempty partial,
its cooperative scale underflows to zero exactly as in the current combine, so
it contributes neither numerator nor denominator.  Interior empty partials are
not skipped or reordered.

The all-empty case remains undefined in the same way as stock: all maxima are
the sentinel, scales are one, and the final division has a zero denominator.
Valid attention rows should always have at least one unmasked key.  Do not add
an experiment-only fallback without first defining the desired stock behavior.

The lane-0 max uses `sycl::max`, matching the current function.  This matters
for unusual NaN/signed-zero behavior; changing it to `sycl::fmax` would not be
an exact transcription.

## Numerical risk

This layout is designed to be bit-stable, but bit identity must be measured,
not assumed.

- Max order is unchanged: one lane executes the same sequential 0..31 chain.
- Each scale uses the same `sycl::native::exp(meta.x - kqmax)` expression.  A
  float is stored to local memory before reuse; the stock scale is already a
  float temporary.
- Each output work-item accumulates numerator and denominator in the same
  0..31 order using the same multiply-add expressions.
- Metadata and partial values are neither omitted nor reassigned.

A separate kernel can nevertheless provoke different IGC inlining, FMA
contraction, or native-math instruction selection.  Required gates are exact
backend-op comparison on adversarial masks/empty partials, deterministic greedy
token/content comparison, and the established KLD corpus.  Treat even small
logit differences as a real numerical change until explained.

## Performance expectation

Per output vector at PB32, the scalar work count changes approximately as
follows:

| Work | Stock | Cooperative |
|---|---:|---:|
| native exponentials | 4,096 | 32 |
| max updates | 3,968 | 31 |
| numerator terms | 4,096 | 4,096 |
| denominator terms | 4,096 | 4,096 |
| `VKQ_parts` float loads | 4,096 | 4,096 |

Logical local-memory traffic in the accumulation phase stays essentially the
same.  The proposal adds roughly 128 bytes of logical global metadata reads per
output vector because lane 0 scans maxima and lanes 0..31 subsequently fetch
their `float2`; those reads should be cache-hot but are not free.  It also adds
one work-group broadcast.  The large partial matrix read and both accumulation
chains remain unchanged.

Consequently, expect a material combine-kernel speedup only if redundant native
exp/max instructions are currently on its critical path.  A defensible initial
expectation is roughly 5-25% for the combine kernel itself and 0.2-1.0% for
end-to-end one-seat decode at 131k; zero or a regression is plausible if launch
and memory traffic dominate.  This proposal should not be described as a
serving win until a real `llama-server -np 1 -c 131072` A/B clears noise.

## Measurement and acceptance plan

1. Apply the artifact and build once.  Record the SYCL library hash.
2. With the cooperative gate unset, rerun the existing exact op/golden check to
   prove that the default dispatch remains unchanged.
3. Screen the same binary at `d=131072`, PB32, alternating cooperative OFF/ON;
   use enough repeats to resolve a sub-1% effect.  Do not compare different
   builds or a warm arm against a cold arm.
4. If the screen is positive, run the real server path with one slot, full
   131072 context, identical prompt/KV depth, f16 KV, and identical sampling.
   Compare prompt and predicted token accounting as well as wall time.
5. Run adversarial backend-op cases covering normal PB32, leading/trailing and
   interior empty partials, and a short context that clamps PB below 32 (which
   must fall back to stock).
6. Run deterministic greedy token/content checks and the established KLD gate.
7. Reject or leave disabled if the server gain is below about 0.5%, confidence
   intervals overlap materially, any gate-off result changes, or quality output
   changes without a separately approved tolerance analysis.

No benchmark was run for this design-only task, per the explicit no-GPU scope.
