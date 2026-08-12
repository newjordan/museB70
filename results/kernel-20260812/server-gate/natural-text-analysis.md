# Natural-text concurrent gate analysis

The strict slot-by-slot comparison failed, but this workload also diverged
within the unchanged control arm. Both arms produced six unique 64-token
continuations from eight identical seeded requests.

Candidate-vs-control common-prefix lengths by slot were
`35, 50, 21, 64, 24, 33, 6, 33` tokens. Six of the eight complete candidate
continuations were byte-identical to a complete control continuation assigned
to a different slot. Pairwise within-arm common-prefix statistics were also
nearly identical: control mean/median `32.43/33`, candidate `32.04/35`.

The schedules were not equivalent: control launched all eight slots together,
while the candidate launched seven and admitted slot 3 about 3.92 seconds
later. Eight 1,386-token prompts also exceeded `-ub 4096`, creating different
partial-prefill cohorts. This is the already documented master-series
batch/scheduler numerical instability, so this receipt is neither a kernel
quality pass nor an attributable regression.

The attribution gate therefore used the previously stable repetitive workload
in `../server-isolation/`: control A/A, the rebuilt binary with the feature off,
the feature on, and the final promoted binary all matched byte-for-byte and
token-for-token in all eight slots for 64 generated tokens.
