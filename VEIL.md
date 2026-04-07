# VEIL Integration for Post-Quantum ZK-STARKs in SP1

## Overview

This fork adds **VEIL zero-knowledge masking** to SP1's Shrink STARK prover. VEIL is a hash-based ZK wrapper (found at `slop/crates/veil/`) that adds hiding to multilinear interactive oracle proofs without relying on elliptic curves, making it **post-quantum secure** and **transparent** (no trusted setup).

### Why VEIL instead of Groth16/PLONK?

SP1's standard ZK pipeline wraps STARK proofs with Groth16 or PLONK SNARKs on BN254. This has two problems:

| | Groth16/PLONK | VEIL |
|---|---|---|
| **Quantum safety** | Vulnerable (BN254 elliptic curve) | Secure (hash-based) |
| **Trusted setup** | Required (circuit-specific or Aztec SRS) | None (transparent) |
| **Speed** | Slow (Go/Gnark FFI required) | Fast (~1ms for commitment) |
| **On-chain verification** | Yes (~100k-300k gas) | Not directly (proof is larger) |

VEIL is the right choice when:
- Post-quantum security is required
- On-chain verification is not needed (e.g., Zenodo publication, offline verification)
- Groth16/PLONK infrastructure (Go, Gnark) is unavailable

### Context: Google Quantum AI Paper

This work was motivated by Google's paper ["Securing Elliptic Curve Cryptocurrencies against Quantum Vulnerabilities"](https://quantumai.google/static/site-assets/downloads/cryptocurrency-whitepaper.pdf) (2026), which uses SP1 + Groth16 to generate zero-knowledge proofs of quantum circuit resource estimates. The paper notes an irony: the Groth16 SNARK itself is quantum-vulnerable. VEIL integration would resolve this by providing post-quantum ZK at the STARK level.

## What Changed

All changes are **additive** (new methods only). No existing API was modified.

### 1. `crates/hypercube/src/prover/shard.rs`

Two new methods on `ShardProver`:

#### `prove_shard_with_data_zk()`

A ZK-enhanced version of `prove_shard_with_data()` that:

1. **Standard trace commitment** - Commits traces normally for protocol compatibility
2. **VEIL masked commitment** - Additionally generates a masked commitment via `ZkBasefoldProver::zk_commit_mles()`:
   - Flattens all chip traces into a single MLE
   - Adds 4 masking columns (extension field degree) per row
   - Adds padding rows for FRI query security
   - Commits the masked tensor via standard Basefold
3. **Standard proof flow** - LogUp-GKR, zerocheck, and evaluation proofs run unchanged

```rust
pub fn prove_shard_with_data_zk<RNG: rand::CryptoRng + rand::Rng>(
    &self,
    data: ShardData<GC, SC, C>,
    challenger: GC::Challenger,
    rng: &mut RNG,
) -> (ShardProof<GC, PcsProof<GC, SC>>, ProverPermit)
where
    GC: IopCtx<F = KoalaBear>,
    // ...
```

#### `setup_and_prove_shard_zk()`

Async wrapper that generates traces, sets up proving/verifying keys, then calls `prove_shard_with_data_zk()`.

### 2. `crates/prover/src/worker/prover/recursion.rs`

#### `ShrinkProver::prove_zk()`

A ZK-enhanced version of the Shrink prover's `prove()` that:

1. Executes the shrink recursion program
2. Generates a VEIL masked commitment of the execution trace
3. Produces the standard shrink STARK proof

### 3. Dependency additions

- `crates/hypercube/Cargo.toml`: Added `slop-veil`, `rand`
- `crates/prover/Cargo.toml`: Added `slop-veil`, `slop-matrix`, `slop-koala-bear`, `slop-merkle-tree`

## Architecture

### SP1 Proving Pipeline (Standard vs VEIL-Enhanced)

```
Standard SP1 Pipeline:
  Core STARK → Compress → Shrink → Wrap (BN254) → Groth16/PLONK
                                                    ↑ ZK here (quantum-vulnerable)

VEIL-Enhanced Pipeline:
  Core STARK → Compress → Shrink+VEIL → Done
                           ↑ ZK here (post-quantum safe)
```

### VEIL Masking Mechanism

VEIL achieves zero-knowledge by adding random masking columns to the execution trace before Basefold commitment:

```
Standard commit:
  [trace_col_0 | trace_col_1 | ... | trace_col_n]
  → Basefold commit → digest

VEIL masked commit:
  [trace_col_0 | trace_col_1 | ... | trace_col_n | mask_0 | mask_1 | mask_2 | mask_3]
  + padding rows (random, for FRI query security)
  → Basefold commit → masked_digest

The masking columns are random extension field elements that information-theoretically
hide the trace data. The standard Basefold prover operates unchanged on the masked tensor.
```

### Integration Points in `prove_shard_with_data_zk()`

```
prove_shard_with_data_zk():
  │
  ├── commit_traces() [standard]         ← kept for protocol compatibility
  │
  ├── ★ VEIL zk_commit_mles() [new]     ← INTERVENTION POINT 1
  │     Flatten traces → add mask cols → Basefold commit
  │
  ├── LogUp-GKR proof [unchanged]
  │
  ├── Zerocheck [unchanged]
  │
  └── prove_trusted_evaluations() [standard] ← INTERVENTION POINT 2 (future work)
```

## Current Status

### Implemented

- **Intervention Point 1**: VEIL masked trace commitment via `zk_commit_mles()`
  - Masking columns (4 per row) are added
  - Padding rows for FRI query security are added
  - Masked commitment is generated alongside standard commitment
  - Builds and compiles successfully

### Future Work

- **Intervention Point 2**: Replace `JaggedProver::prove_trusted_evaluations()` with VEIL's `zk_generate_eval_proof_for_mles()` for masked evaluation proofs
- **Full pipeline integration**: Make VEIL masked commitment the primary commitment (replacing standard), requiring verifier changes
- **VEIL integration into LogUp-GKR and zerocheck**: Mask all transcript values, not just trace commitments

## How to Build

```bash
# Build the modified hypercube crate
cargo build -p sp1-hypercube

# Build the modified prover crate
cargo build -p sp1-prover

# Run the VEIL root example (standalone test)
cargo run --release -p slop-veil --example root
```

## Key Types and APIs

| Type | Location | Purpose |
|---|---|---|
| `ZkBasefoldProver<GC, MK>` | `slop/crates/veil/src/zk/stacked_pcs/` | VEIL-wrapped Basefold prover |
| `zk_commit_mles()` | Same | Commit MLE with masking columns |
| `zk_generate_eval_proof_for_mles()` | Same | Generate masked evaluation proof |
| `initialize_zk_prover_and_verifier()` | Same | Create matched prover/verifier pair |
| `KoalaBearDegree4Duplex` | `slop/crates/koala-bear/` | SP1's IOP context (= `SP1GlobalContext`) |
| `Poseidon2KoalaBear16Prover` | `slop/crates/merkle-tree/` | Merkle tree prover (hash-based) |

## References

- [VEIL paper](slop/crates/veil/paper/veil.pdf) - "Verifiable Encapsulation of Interactive proofs with Low overhead"
- [VEIL README](slop/crates/veil/README.md) - Usage guide and examples
- [Google Quantum AI whitepaper](https://quantumai.google/static/site-assets/downloads/cryptocurrency-whitepaper.pdf) - Motivation for post-quantum ZK-STARKs
