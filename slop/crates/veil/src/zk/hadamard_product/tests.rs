#![allow(clippy::disallowed_types, clippy::disallowed_methods, dead_code)]
use crate::zk::error_correcting_code::RsInterpolation;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use slop_challenger::IopCtx;
use slop_koala_bear::KoalaBearDegree4Duplex;
use slop_merkle_tree::Poseidon2KoalaBear16Prover;

use std::time::Instant;

use super::{
    hadamard_product, verify_zk_hadamard_product, zk_hadamard_product_commitment,
    zk_hadamard_product_proof,
};

#[tokio::test]
async fn test_zk_hadamard_product_honest() {
    const LENGTH: usize = 3000;
    type GC = KoalaBearDegree4Duplex;

    // Generate three random vectors where c = a * b (Hadamard product)
    let mut rng = ChaCha20Rng::from_entropy();
    let a_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let b_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let c_vec = hadamard_product(&a_vec, &b_vec);

    // Compute and time commitment + proof generation
    let start = Instant::now();
    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) = zk_hadamard_product_commitment::<
        GC,
        _,
        _,
        RsInterpolation<_>,
    >(&a_vec, &b_vec, &c_vec, &mut rng, &merkleizer);

    let mut challenger_prove = GC::default_challenger();
    let total_proof = zk_hadamard_product_proof::<GC, _, RsInterpolation<_>>(
        commitment,
        prover_secret_data,
        &mut challenger_prove,
        &merkleizer,
    );
    let duration = start.elapsed();
    eprintln!("Commitment + proof generation time: {:?}", duration);
    eprintln!("Proof gamma: {:?}", total_proof.proof.gamma);

    // Compute and time verification
    let start = Instant::now();
    let mut challenger_ver = GC::default_challenger();
    verify_zk_hadamard_product::<GC, RsInterpolation<_>>(
        &commitment,
        &total_proof,
        &mut challenger_ver,
    )
    .unwrap();
    let duration = start.elapsed();
    eprintln!("Verification time: {:?}", duration);
}

#[tokio::test]
async fn test_zk_hadamard_product_small() {
    const LENGTH: usize = 100;
    type GC = KoalaBearDegree4Duplex;

    // Generate three small random vectors
    let mut rng = ChaCha20Rng::from_entropy();
    let a_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let b_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let c_vec = hadamard_product(&a_vec, &b_vec);

    eprintln!("Testing with LENGTH={}", LENGTH);

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) = zk_hadamard_product_commitment::<
        GC,
        _,
        _,
        RsInterpolation<_>,
    >(&a_vec, &b_vec, &c_vec, &mut rng, &merkleizer);

    let mut challenger_prove = GC::default_challenger();
    let total_proof = zk_hadamard_product_proof::<GC, _, RsInterpolation<_>>(
        commitment,
        prover_secret_data,
        &mut challenger_prove,
        &merkleizer,
    );

    let mut challenger_ver = GC::default_challenger();
    verify_zk_hadamard_product::<GC, RsInterpolation<_>>(
        &commitment,
        &total_proof,
        &mut challenger_ver,
    )
    .unwrap();

    eprintln!("Small test passed!");
}

#[tokio::test]
#[should_panic(expected = "FDotZInconsistency")]
async fn test_zk_hadamard_product_invalid() {
    const LENGTH: usize = 100;
    type GC = KoalaBearDegree4Duplex;

    // Generate vectors where c != a * b (invalid Hadamard product)
    let mut rng = ChaCha20Rng::from_entropy();
    let a_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let b_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let mut c_vec = hadamard_product(&a_vec, &b_vec);

    // Corrupt one element of c
    use slop_algebra::AbstractField;
    c_vec[50] += <GC as IopCtx>::EF::one();

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) = zk_hadamard_product_commitment::<
        GC,
        _,
        _,
        RsInterpolation<_>,
    >(&a_vec, &b_vec, &c_vec, &mut rng, &merkleizer);

    let mut challenger_prove = GC::default_challenger();
    let total_proof = zk_hadamard_product_proof::<GC, _, RsInterpolation<_>>(
        commitment,
        prover_secret_data,
        &mut challenger_prove,
        &merkleizer,
    );

    let mut challenger_ver = GC::default_challenger();
    verify_zk_hadamard_product::<GC, RsInterpolation<_>>(
        &commitment,
        &total_proof,
        &mut challenger_ver,
    )
    .unwrap();
    // This should panic
}

#[tokio::test]
async fn test_zk_hadamard_product_different_sizes() {
    type GC = KoalaBearDegree4Duplex;

    for length in [10, 50, 100, 500, 1000] {
        eprintln!("\nTesting with LENGTH={}", length);

        let mut rng = ChaCha20Rng::from_entropy();
        let a_vec: Vec<<GC as IopCtx>::EF> = (0..length).map(|_| rng.gen()).collect();
        let b_vec: Vec<<GC as IopCtx>::EF> = (0..length).map(|_| rng.gen()).collect();
        let c_vec = hadamard_product(&a_vec, &b_vec);

        let merkleizer = Poseidon2KoalaBear16Prover::default();
        let (commitment, prover_secret_data) = zk_hadamard_product_commitment::<
            GC,
            _,
            _,
            RsInterpolation<_>,
        >(
            &a_vec, &b_vec, &c_vec, &mut rng, &merkleizer
        );

        let mut challenger_prove = GC::default_challenger();
        let total_proof = zk_hadamard_product_proof::<GC, _, RsInterpolation<_>>(
            commitment,
            prover_secret_data,
            &mut challenger_prove,
            &merkleizer,
        );

        let mut challenger_ver = GC::default_challenger();
        verify_zk_hadamard_product::<GC, RsInterpolation<_>>(
            &commitment,
            &total_proof,
            &mut challenger_ver,
        )
        .unwrap();

        eprintln!("Test passed for LENGTH={}", length);
    }
}

#[tokio::test]
async fn test_zk_hadamard_and_dots() {
    use super::{verify_zk_hadamard_and_dots, zk_hadamard_and_dots_proof};

    const LENGTH: usize = 100;
    type GC = KoalaBearDegree4Duplex;
    type Code = RsInterpolation<<GC as IopCtx>::EF>;

    eprintln!("Testing combined Hadamard + dot product proofs with LENGTH={}", LENGTH);

    // Generate three random vectors where c = a * b (Hadamard product)
    let mut rng = ChaCha20Rng::from_entropy();
    let a_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let b_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let c_vec = hadamard_product(&a_vec, &b_vec);

    // Create a random vector to dot product with
    let dot_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();

    // ==================== PROVER ====================
    let (commitment, total_proof) = {
        let merkleizer = Poseidon2KoalaBear16Prover::default();
        let mut challenger = GC::default_challenger();

        // Commit to the three vectors for Hadamard product
        eprintln!("\n=== Committing vectors ===");
        let (commitment, prover_secret_data) = zk_hadamard_product_commitment::<GC, _, _, Code>(
            &a_vec,
            &b_vec,
            &c_vec,
            &mut rng,
            &merkleizer,
        );

        // Generate combined proofs with shared indices
        eprintln!("\n=== Generating combined Hadamard + dot product proofs ===");
        let total_proof = zk_hadamard_and_dots_proof::<GC, _, Code>(
            commitment,
            &dot_vec,
            prover_secret_data,
            &mut challenger,
            &merkleizer,
        );

        (commitment, total_proof)
    };

    // ==================== VERIFIER ====================
    {
        let mut challenger = GC::default_challenger();

        // Verify combined proofs
        eprintln!("\n=== Verifying combined proofs ===");
        verify_zk_hadamard_and_dots::<GC, Code>(
            &commitment,
            &dot_vec,
            &total_proof,
            &mut challenger,
        )
        .unwrap();
        eprintln!("All proofs verified successfully!");
    }

    eprintln!("\n=== Test passed! ===");
}

// ============================================================================
// New tests: edge cases, code variants, corruption, parametric sizes
// ============================================================================

/// Test Hadamard product with very small vectors (length 2).
#[tokio::test]
async fn test_zk_hadamard_product_tiny() {
    const LENGTH: usize = 2;
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();
    let a_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let b_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let c_vec = hadamard_product(&a_vec, &b_vec);

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) = zk_hadamard_product_commitment::<
        GC,
        _,
        _,
        RsInterpolation<_>,
    >(&a_vec, &b_vec, &c_vec, &mut rng, &merkleizer);

    let mut challenger_prove = GC::default_challenger();
    let total_proof = zk_hadamard_product_proof::<GC, _, RsInterpolation<_>>(
        commitment,
        prover_secret_data,
        &mut challenger_prove,
        &merkleizer,
    );

    let mut challenger_ver = GC::default_challenger();
    verify_zk_hadamard_product::<GC, RsInterpolation<_>>(
        &commitment,
        &total_proof,
        &mut challenger_ver,
    )
    .unwrap();
}

/// Test Hadamard product with power-of-two size (256).
#[tokio::test]
async fn test_zk_hadamard_product_power_of_two() {
    const LENGTH: usize = 256;
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();
    let a_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let b_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let c_vec = hadamard_product(&a_vec, &b_vec);

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) = zk_hadamard_product_commitment::<
        GC,
        _,
        _,
        RsInterpolation<_>,
    >(&a_vec, &b_vec, &c_vec, &mut rng, &merkleizer);

    let mut challenger_prove = GC::default_challenger();
    let total_proof = zk_hadamard_product_proof::<GC, _, RsInterpolation<_>>(
        commitment,
        prover_secret_data,
        &mut challenger_prove,
        &merkleizer,
    );

    let mut challenger_ver = GC::default_challenger();
    verify_zk_hadamard_product::<GC, RsInterpolation<_>>(
        &commitment,
        &total_proof,
        &mut challenger_ver,
    )
    .unwrap();
}

/// Test Hadamard product with all-one vectors (identity-like).
#[tokio::test]
async fn test_zk_hadamard_product_identity() {
    use slop_algebra::AbstractField;
    const LENGTH: usize = 100;
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();
    // a = random, b = all-ones → c = a
    let a_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let b_vec: Vec<<GC as IopCtx>::EF> = vec![<GC as IopCtx>::EF::one(); LENGTH];
    let c_vec = hadamard_product(&a_vec, &b_vec);
    assert_eq!(a_vec, c_vec);

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) = zk_hadamard_product_commitment::<
        GC,
        _,
        _,
        RsInterpolation<_>,
    >(&a_vec, &b_vec, &c_vec, &mut rng, &merkleizer);

    let mut challenger_prove = GC::default_challenger();
    let total_proof = zk_hadamard_product_proof::<GC, _, RsInterpolation<_>>(
        commitment,
        prover_secret_data,
        &mut challenger_prove,
        &merkleizer,
    );

    let mut challenger_ver = GC::default_challenger();
    verify_zk_hadamard_product::<GC, RsInterpolation<_>>(
        &commitment,
        &total_proof,
        &mut challenger_ver,
    )
    .unwrap();
}

/// Test Hadamard product with all-zero vector (annihilator).
#[tokio::test]
async fn test_zk_hadamard_product_zero_annihilator() {
    use slop_algebra::AbstractField;
    const LENGTH: usize = 50;
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();
    let a_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let b_vec: Vec<<GC as IopCtx>::EF> = vec![<GC as IopCtx>::EF::zero(); LENGTH];
    let c_vec = hadamard_product(&a_vec, &b_vec);
    assert!(c_vec.iter().all(|x| *x == <GC as IopCtx>::EF::zero()));

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) = zk_hadamard_product_commitment::<
        GC,
        _,
        _,
        RsInterpolation<_>,
    >(&a_vec, &b_vec, &c_vec, &mut rng, &merkleizer);

    let mut challenger_prove = GC::default_challenger();
    let total_proof = zk_hadamard_product_proof::<GC, _, RsInterpolation<_>>(
        commitment,
        prover_secret_data,
        &mut challenger_prove,
        &merkleizer,
    );

    let mut challenger_ver = GC::default_challenger();
    verify_zk_hadamard_product::<GC, RsInterpolation<_>>(
        &commitment,
        &total_proof,
        &mut challenger_ver,
    )
    .unwrap();
}

/// Test Hadamard soundness: corrupt multiple elements of c.
#[tokio::test]
#[should_panic(expected = "FDotZInconsistency")]
async fn test_zk_hadamard_product_multiple_corruptions() {
    use slop_algebra::AbstractField;
    const LENGTH: usize = 200;
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();
    let a_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let b_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let mut c_vec = hadamard_product(&a_vec, &b_vec);

    // Corrupt 10 elements
    for i in (0..LENGTH).step_by(20) {
        c_vec[i] += <GC as IopCtx>::EF::one();
    }

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) = zk_hadamard_product_commitment::<
        GC,
        _,
        _,
        RsInterpolation<_>,
    >(&a_vec, &b_vec, &c_vec, &mut rng, &merkleizer);

    let mut challenger_prove = GC::default_challenger();
    let total_proof = zk_hadamard_product_proof::<GC, _, RsInterpolation<_>>(
        commitment,
        prover_secret_data,
        &mut challenger_prove,
        &merkleizer,
    );

    let mut challenger_ver = GC::default_challenger();
    verify_zk_hadamard_product::<GC, RsInterpolation<_>>(
        &commitment,
        &total_proof,
        &mut challenger_ver,
    )
    .unwrap();
}

/// Test Hadamard combined with dots at various sizes.
#[tokio::test]
async fn test_zk_hadamard_and_dots_various_sizes() {
    use super::{verify_zk_hadamard_and_dots, zk_hadamard_and_dots_proof};

    type GC = KoalaBearDegree4Duplex;
    type Code = RsInterpolation<<GC as IopCtx>::EF>;

    for length in [10, 50, 200, 500] {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_vec: Vec<<GC as IopCtx>::EF> = (0..length).map(|_| rng.gen()).collect();
        let b_vec: Vec<<GC as IopCtx>::EF> = (0..length).map(|_| rng.gen()).collect();
        let c_vec = hadamard_product(&a_vec, &b_vec);
        let dot_vec: Vec<<GC as IopCtx>::EF> = (0..length).map(|_| rng.gen()).collect();

        let (commitment, total_proof) = {
            let merkleizer = Poseidon2KoalaBear16Prover::default();
            let mut challenger = GC::default_challenger();
            let (commitment, prover_secret_data) = zk_hadamard_product_commitment::<GC, _, _, Code>(
                &a_vec,
                &b_vec,
                &c_vec,
                &mut rng,
                &merkleizer,
            );
            let total_proof = zk_hadamard_and_dots_proof::<GC, _, Code>(
                commitment,
                &dot_vec,
                prover_secret_data,
                &mut challenger,
                &merkleizer,
            );
            (commitment, total_proof)
        };

        let mut challenger = GC::default_challenger();
        verify_zk_hadamard_and_dots::<GC, Code>(
            &commitment,
            &dot_vec,
            &total_proof,
            &mut challenger,
        )
        .unwrap();
    }
}

/// Test Hadamard product with large non-power-of-two size (2001).
#[tokio::test]
async fn test_zk_hadamard_product_large_non_pow2() {
    const LENGTH: usize = 2001;
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();
    let a_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let b_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let c_vec = hadamard_product(&a_vec, &b_vec);

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) = zk_hadamard_product_commitment::<
        GC,
        _,
        _,
        RsInterpolation<_>,
    >(&a_vec, &b_vec, &c_vec, &mut rng, &merkleizer);

    let mut challenger_prove = GC::default_challenger();
    let total_proof = zk_hadamard_product_proof::<GC, _, RsInterpolation<_>>(
        commitment,
        prover_secret_data,
        &mut challenger_prove,
        &merkleizer,
    );

    let mut challenger_ver = GC::default_challenger();
    verify_zk_hadamard_product::<GC, RsInterpolation<_>>(
        &commitment,
        &total_proof,
        &mut challenger_ver,
    )
    .unwrap();
}

/// Test soundness: swap a and b vectors (c = a*b should still verify, but with swapped inputs).
#[tokio::test]
async fn test_zk_hadamard_product_commutative() {
    const LENGTH: usize = 300;
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();
    let a_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let b_vec: Vec<<GC as IopCtx>::EF> = (0..LENGTH).map(|_| rng.gen()).collect();
    let c_vec = hadamard_product(&a_vec, &b_vec);
    // Commutativity: hadamard(b, a) == hadamard(a, b)
    let c_vec_swapped = hadamard_product(&b_vec, &a_vec);
    assert_eq!(c_vec, c_vec_swapped);

    // Prove with swapped order (b, a, c)
    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) = zk_hadamard_product_commitment::<
        GC,
        _,
        _,
        RsInterpolation<_>,
    >(&b_vec, &a_vec, &c_vec, &mut rng, &merkleizer);

    let mut challenger_prove = GC::default_challenger();
    let total_proof = zk_hadamard_product_proof::<GC, _, RsInterpolation<_>>(
        commitment,
        prover_secret_data,
        &mut challenger_prove,
        &merkleizer,
    );

    let mut challenger_ver = GC::default_challenger();
    verify_zk_hadamard_product::<GC, RsInterpolation<_>>(
        &commitment,
        &total_proof,
        &mut challenger_ver,
    )
    .unwrap();
}
