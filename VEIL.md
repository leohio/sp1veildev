# VEIL Integration for Post-Quantum ZK-STARKs in SP1

## Overview

This fork adds **VEIL zero-knowledge masking** to SP1's Shrink STARK prover. VEIL (`slop/crates/veil/`) adds hiding to multilinear IOPs via hash-based masking, making proofs **post-quantum secure** and **transparent** (no trusted setup, no elliptic curves).

## Current Status: All Critical Issues Resolved

### Resolved

- **Single commitment**: One `ZkProverCtx::commit_mle(&mle)` call per proof. The digest is extracted via `committed_digests()` and observed into the main challenger. No double-commit.
- **Eval point binding**: The VEIL evaluation point is constructed from the STARK zerocheck evaluation point (row variables) concatenated with VEIL-sampled RLC coordinates (column variables). Both prover and verifier construct the same point.
- **Verifier challenger sync**: The verifier observes `masked_digest` into the main challenger before `main_commit` when `veil_proof.is_some()`, matching the prover's ordering.
- **Sub-protocol masking**: All trace-leaking values from LogUp-GKR and zerocheck (circuit outputs, round proofs, sumcheck coefficients, chip evaluations, opened values) are fed through `veil_ctx.send_values()`. The verifier reads the same elements from the VEIL transcript. VEIL's mask-proof covers all transcript data.
- **Dummy data eliminated**: `ShrinkProver::prove_zk()` delegates to `setup_and_prove_shard_zk()` with real execution traces.
- **Clone eliminated**: `ZkPcsProver::commit_mle` takes `&Mle` (reference). No caller-side clone needed.
- **Error propagation**: All ZK methods return `Result<..., VeilProvingError>`.
- **reqwest regression**: `rustls-tls` feature restored.
- **Trace stacking**: `stack_traces_for_veil()` preserves multi-column structure.

### Known Limitations

- **`VeilMaskingProof` hardcoded to `SP1GlobalContext`**: VEIL only implements `ZkIopCtx` for `KoalaBearDegree4Duplex`. Non-ZK proofs always have `veil_proof: None`, so this is safe.
- **Mask length is estimated**: The VEIL mask length is computed with a conservative overestimate of transcript elements. This costs slightly more memory but is always safe.

## Architecture

### Prover Flow

```
prove_shard_with_data_zk():
  │
  ├── commit_traces() [standard]            → main_commit, main_data
  ├── stack_traces_for_veil()               → stacked_mle
  │
  ├── ZkProverCtx::commit_mle(&mle) [VEIL] → veil_commit (single commitment)
  ├── committed_digests()                    → masked_digest
  │
  ├── observe(masked_digest) into challenger
  ├── observe(main_commit) into challenger
  │
  ├── LogUp-GKR proof [standard]            → logup_gkr_proof
  ├── Zerocheck [standard]                  → zerocheck_proof + opened_values
  ├── prove_trusted_evaluations [standard]  → evaluation_proof
  │
  ├── send_values() [VEIL]                  ← mask all sub-protocol transcript data
  │   ├── GKR circuit output (num/den MLEs)
  │   ├── GKR round proofs + sumcheck coefficients
  │   ├── Chip evaluations (main + preprocessed)
  │   ├── Zerocheck sumcheck coefficients
  │   └── Zerocheck opened values
  │
  ├── construct eval point:
  │   evaluation_point [from zerocheck] ++ sample_point(log_num_polys) [from VEIL]
  ├── eval_at + send_value + assert_mle_eval + prove() [VEIL] → zk_proof
  │
  └── ShardProof { ..., veil_proof: Some(VeilMaskingProof { masked_digest, zk_proof, ... }) }
```

### Verifier Flow

When `veil_proof.is_some()`:

1. Deserializes `VeilMaskingProof` and observes `masked_digest` into the main challenger before `main_commit`.
2. Runs standard STARK verification (LogUp-GKR, zerocheck, PCS).
3. Initializes `ZkVerifierCtx` with `zk_proof` and a `ZkStackedPcsVerifier`.
4. Reads oracle commitment from VEIL transcript.
5. Reads all masked sub-protocol transcript elements (matching prover's `send_values` order).
6. Constructs the evaluation point: `zerocheck_point ++ VEIL-sampled RLC`.
7. Reads eval claim, asserts MLE evaluation, runs `verify()`.

### Modified Files

| File | Change |
|---|---|
| `slop/crates/veil/src/zk/inner/pcs_traits.rs` | `ZkPcsProver::commit_mle` takes `&Mle` (reference) |
| `slop/crates/veil/src/zk/stacked_pcs/prover.rs` | Updated `commit_mle` impl for `&Mle` |
| `slop/crates/veil/src/zk/stacked_pcs/utils.rs` | `stack_mle` takes `&Mle` |
| `slop/crates/veil/src/zk/inner/prover.rs` | `ZkProverContext::commit_mle` takes `&Mle` |
| `slop/crates/veil/src/zk/prover_ctx.rs` | `ZkProverCtx::commit_mle` takes `&Mle`. Added `committed_digests()`. |
| `crates/hypercube/src/verifier/proof.rs` | `VeilMaskingProof` struct, `veil_proof` field on `ShardProof` |
| `crates/hypercube/src/prover/shard.rs` | Full ZK proving pipeline with single-commit, eval point binding, sub-protocol masking |
| `crates/hypercube/src/verifier/shard.rs` | VEIL verification with challenger sync, transcript element reading, eval point reconstruction |
| `crates/prover/src/worker/prover/recursion.rs` | `ShrinkProver::prove_zk()` delegates to `setup_and_prove_shard_zk` |
| `crates/prover/Cargo.toml` | Restored `rustls-tls`, removed unused VEIL deps |

## How to Build

```bash
cargo build -p sp1-hypercube
cargo build -p sp1-prover  # requires protoc
cargo run --release -p slop-veil --example root
```

## Key Types

| Type | Location | Purpose |
|---|---|---|
| `VeilMaskingProof<GC>` | `crates/hypercube/src/verifier/proof.rs` | VEIL ZK proof attached to `ShardProof` |
| `VeilProvingError` | `crates/hypercube/src/prover/shard.rs` | Error type for VEIL proving failures |
| `ZkProverCtx` | `slop/crates/veil/src/zk/prover_ctx.rs` | High-level VEIL prover context |
| `ZkVerifierCtx` | `slop/crates/veil/src/zk/verifier_ctx.rs` | High-level VEIL verifier context |
| `initialize_zk_prover_and_verifier()` | `slop/crates/veil/src/zk/stacked_pcs/` | Create matched prover/verifier pair |

## References

- [VEIL paper](slop/crates/veil/paper/veil.pdf) - "Verifiable Encapsulation of Interactive proofs with Low overhead"
- [VEIL README](slop/crates/veil/README.md) - Usage guide and examples
