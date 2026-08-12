# Kernel provenance

The validated source worktree is based on private llama.cpp commit
`9ace4046d` (Muse Glimmer architecture support). The two serving kernels are
preserved as mail patches in commit order:

1. `c317854233a9e4989fad7a68c03899314e0b11f1` - reordered Q4_K weight
   hoisting across small-N columns.
2. `3ce44d373c3fe2b2f4c88f196f9d063d3eefb267` - mask-derived SWA prefix
   skipping for full-context decode.

Apply them to the matching llama.cpp source base with:

```bash
git am /path/to/museB70/patches/kernel/*.patch
```

The exact release artifacts, source commit, model identity and validation
receipts are recorded in `results/serve-quality-manifest-20260812T1712Z.txt`.
