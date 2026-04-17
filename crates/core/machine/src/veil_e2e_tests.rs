//! VEIL ZK end-to-end tests at the core shard level.
//!
//! NOTE: Core machine traces are too large (2048 columns × 131K rows) for VEIL
//! in practice. Pipeline-level VEIL E2E tests (through the shrink step) are in
//! `crates/prover/src/worker/node/full/mod.rs` (`test_veil_e2e_*`).
//!
//! The `prove_core_zk` / `run_test_core_zk` functions are kept for future use
//! with smaller circuits or reduced trace sizes.
