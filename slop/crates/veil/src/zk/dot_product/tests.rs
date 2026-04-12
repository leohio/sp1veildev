#![allow(clippy::disallowed_types, clippy::disallowed_methods)]
use crate::zk::error_correcting_code::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use slop_challenger::IopCtx;
use slop_koala_bear::KoalaBearDegree4Duplex;
use slop_merkle_tree::Poseidon2KoalaBear16Prover;

use slop_algebra::AbstractField;

use super::verifier::ZkDotProductError;
use super::{
    dot_product, verify_zk_dot_product, verify_zk_dot_products, zk_dot_product_commitment,
    zk_dot_product_proof, zk_dot_products_proof,
};

#[tokio::test]
async fn test_zk_dot_product() {
    const LENGTH: usize = 3000;
    type GC = KoalaBearDegree4Duplex;
    type Code = RsFromCoefficients<<GC as IopCtx>::EF>;

    let mut rng = ChaCha20Rng::from_entropy();
    let in_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
    let dot_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
    let expected = dot_product(&in_vec, &dot_vec);

    // Prover
    let (commitment, total_proof) = {
        let merkleizer = Poseidon2KoalaBear16Prover::default();
        let (commitment, prover_secret_data) =
            zk_dot_product_commitment::<GC, _, _, Code>(&[in_vec], &mut rng, &merkleizer);
        let mut challenger = GC::default_challenger();
        let total_proof = zk_dot_product_proof::<GC, _, Code>(
            &dot_vec,
            &commitment,
            prover_secret_data,
            &mut challenger,
            &merkleizer,
        );
        (commitment, total_proof)
    };

    assert_eq!(total_proof.proof.claimed_dot_products[0], expected);

    // Verifier
    let mut challenger = GC::default_challenger();
    verify_zk_dot_product::<GC, Code>(&commitment, &dot_vec, &total_proof, &mut challenger)
        .unwrap();
}

#[tokio::test]
async fn test_zk_dot_products_100() {
    const LENGTH: usize = 3000;
    const NUM_DOT_VECS: usize = 100;
    type GC = KoalaBearDegree4Duplex;
    type Code = RsFromCoefficients<<GC as IopCtx>::EF>;

    let mut rng = ChaCha20Rng::from_entropy();
    let in_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
    let dot_vecs: Vec<Vec<<GC as IopCtx>::EF>> =
        std::iter::repeat_with(|| std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect())
            .take(NUM_DOT_VECS)
            .collect();

    // Prover
    let (commitment, total_proof) = {
        let merkleizer = Poseidon2KoalaBear16Prover::default();
        let (commitment, prover_secret_data) =
            zk_dot_product_commitment::<GC, _, _, Code>(&[in_vec], &mut rng, &merkleizer);
        let mut challenger = GC::default_challenger();
        let total_proof = zk_dot_products_proof::<GC, _, Code>(
            &dot_vecs,
            commitment,
            prover_secret_data,
            &mut challenger,
            &merkleizer,
        );
        (commitment, total_proof)
    };

    // Verifier
    let mut challenger = GC::default_challenger();
    verify_zk_dot_products::<GC, RsFromCoefficients<_>>(
        &commitment,
        &dot_vecs,
        &total_proof,
        &mut challenger,
    )
    .unwrap();
}

#[tokio::test]
async fn test_zk_dot_products_100_corrupted() {
    const LENGTH: usize = 3000;
    const NUM_DOT_VECS: usize = 100;
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();
    let in_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
    let dot_vecs: Vec<Vec<<GC as IopCtx>::EF>> =
        std::iter::repeat_with(|| std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect())
            .take(NUM_DOT_VECS)
            .collect();

    // Prover
    let (commitment, total_proof) = {
        let merkleizer = Poseidon2KoalaBear16Prover::default();
        let (commitment, prover_secret_data) = zk_dot_product_commitment::<
            GC,
            _,
            _,
            RsFromCoefficients<_>,
        >(&[in_vec], &mut rng, &merkleizer);
        let mut challenger = GC::default_challenger();
        let total_proof = zk_dot_products_proof::<GC, _, RsFromCoefficients<_>>(
            &dot_vecs,
            commitment,
            prover_secret_data,
            &mut challenger,
            &merkleizer,
        );
        (commitment, total_proof)
    };

    // Corrupt one random dot_vec entry
    let mut corrupted_dot_vecs = dot_vecs.clone();
    corrupted_dot_vecs[rng.gen_range(0..NUM_DOT_VECS)][rng.gen_range(0..LENGTH)] = rng.gen();

    // Verifier (should fail)
    let mut challenger = GC::default_challenger();
    let result = verify_zk_dot_products::<GC, RsFromCoefficients<_>>(
        &commitment,
        &corrupted_dot_vecs,
        &total_proof,
        &mut challenger,
    );
    match result {
        Err(ZkDotProductError::RLCDotInconsistency) => {}
        other => panic!("Expected RLCDotInconsistency error, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_zk_dot_product_batched() {
    const LENGTH: usize = 3000;
    const NUM_INPUT_VECS: usize = 100;
    type GC = KoalaBearDegree4Duplex;
    type Code = RsFromCoefficients<<GC as IopCtx>::EF>;

    let mut rng = ChaCha20Rng::from_entropy();
    let in_vecs: Vec<Vec<<GC as IopCtx>::EF>> =
        std::iter::repeat_with(|| std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect())
            .take(NUM_INPUT_VECS)
            .collect();
    let dot_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();

    let expected: Vec<_> = in_vecs.iter().map(|v| dot_product(v, &dot_vec)).collect();

    // Prover
    let (commitment, total_proof) = {
        let merkleizer = Poseidon2KoalaBear16Prover::default();
        let (commitment, prover_secret_data) =
            zk_dot_product_commitment::<GC, _, _, Code>(&in_vecs, &mut rng, &merkleizer);
        let mut challenger = GC::default_challenger();
        let total_proof = zk_dot_product_proof::<GC, _, Code>(
            &dot_vec,
            &commitment,
            prover_secret_data,
            &mut challenger,
            &merkleizer,
        );
        (commitment, total_proof)
    };

    assert_eq!(total_proof.proof.claimed_dot_products, expected);

    // Verifier
    let mut challenger = GC::default_challenger();
    verify_zk_dot_product::<GC, Code>(&commitment, &dot_vec, &total_proof, &mut challenger)
        .unwrap();
}

#[tokio::test]
async fn test_zk_dot_product_batched_corrupted() {
    const LENGTH: usize = 3000;
    const NUM_INPUT_VECS: usize = 5;
    type GC = KoalaBearDegree4Duplex;
    type Code = RsFromCoefficients<<GC as IopCtx>::EF>;

    let mut rng = ChaCha20Rng::from_entropy();
    let in_vecs: Vec<Vec<<GC as IopCtx>::EF>> =
        std::iter::repeat_with(|| std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect())
            .take(NUM_INPUT_VECS)
            .collect();
    let dot_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();

    // Prover
    let (commitment, mut total_proof) = {
        let merkleizer = Poseidon2KoalaBear16Prover::default();
        let (commitment, prover_secret_data) =
            zk_dot_product_commitment::<GC, _, _, Code>(&in_vecs, &mut rng, &merkleizer);
        let mut challenger = GC::default_challenger();
        let total_proof = zk_dot_product_proof::<GC, _, Code>(
            &dot_vec,
            &commitment,
            prover_secret_data,
            &mut challenger,
            &merkleizer,
        );
        (commitment, total_proof)
    };

    // Corrupt a random claimed dot product
    total_proof.proof.claimed_dot_products[rng.gen_range(0..NUM_INPUT_VECS)] +=
        rng.gen::<<GC as IopCtx>::EF>();

    // Verifier (should fail with RLCDotInconsistency)
    let mut challenger = GC::default_challenger();
    let result =
        verify_zk_dot_product::<GC, Code>(&commitment, &dot_vec, &total_proof, &mut challenger);
    match result {
        Err(ZkDotProductError::RLCDotInconsistency) => {}
        other => panic!("Expected RLCDotInconsistency, got: {:?}", other),
    }
}

// ============================================================================
// New tests: various sizes, edge cases, and corruption patterns
// ============================================================================

/// Test with a single-element vector.
#[tokio::test]
async fn test_zk_dot_product_single_element() {
    const LENGTH: usize = 1;
    type GC = KoalaBearDegree4Duplex;
    type Code = RsFromCoefficients<<GC as IopCtx>::EF>;

    let mut rng = ChaCha20Rng::from_entropy();
    let in_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
    let dot_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
    let expected = dot_product(&in_vec, &dot_vec);

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) =
        zk_dot_product_commitment::<GC, _, _, Code>(&[in_vec], &mut rng, &merkleizer);
    let mut challenger = GC::default_challenger();
    let total_proof = zk_dot_product_proof::<GC, _, Code>(
        &dot_vec,
        &commitment,
        prover_secret_data,
        &mut challenger,
        &merkleizer,
    );

    assert_eq!(total_proof.proof.claimed_dot_products[0], expected);

    let mut challenger = GC::default_challenger();
    verify_zk_dot_product::<GC, Code>(&commitment, &dot_vec, &total_proof, &mut challenger)
        .unwrap();
}

/// Test with power-of-two length (512).
#[tokio::test]
async fn test_zk_dot_product_power_of_two_512() {
    const LENGTH: usize = 512;
    type GC = KoalaBearDegree4Duplex;
    type Code = RsFromCoefficients<<GC as IopCtx>::EF>;

    let mut rng = ChaCha20Rng::from_entropy();
    let in_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
    let dot_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
    let expected = dot_product(&in_vec, &dot_vec);

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) =
        zk_dot_product_commitment::<GC, _, _, Code>(&[in_vec], &mut rng, &merkleizer);
    let mut challenger = GC::default_challenger();
    let total_proof = zk_dot_product_proof::<GC, _, Code>(
        &dot_vec,
        &commitment,
        prover_secret_data,
        &mut challenger,
        &merkleizer,
    );

    assert_eq!(total_proof.proof.claimed_dot_products[0], expected);

    let mut challenger = GC::default_challenger();
    verify_zk_dot_product::<GC, Code>(&commitment, &dot_vec, &total_proof, &mut challenger)
        .unwrap();
}

/// Test with non-power-of-two length (777).
#[tokio::test]
async fn test_zk_dot_product_non_power_of_two_777() {
    const LENGTH: usize = 777;
    type GC = KoalaBearDegree4Duplex;
    type Code = RsFromCoefficients<<GC as IopCtx>::EF>;

    let mut rng = ChaCha20Rng::from_entropy();
    let in_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
    let dot_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) =
        zk_dot_product_commitment::<GC, _, _, Code>(&[in_vec], &mut rng, &merkleizer);
    let mut challenger = GC::default_challenger();
    let total_proof = zk_dot_product_proof::<GC, _, Code>(
        &dot_vec,
        &commitment,
        prover_secret_data,
        &mut challenger,
        &merkleizer,
    );

    let mut challenger = GC::default_challenger();
    verify_zk_dot_product::<GC, Code>(&commitment, &dot_vec, &total_proof, &mut challenger)
        .unwrap();
}

/// Test dot product using the RsInterpolation code variant.
#[tokio::test]
async fn test_zk_dot_product_interpolation_code() {
    const LENGTH: usize = 500;
    type GC = KoalaBearDegree4Duplex;
    type Code = RsInterpolation<<GC as IopCtx>::EF>;

    let mut rng = ChaCha20Rng::from_entropy();
    let in_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
    let dot_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
    let expected = dot_product(&in_vec, &dot_vec);

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) =
        zk_dot_product_commitment::<GC, _, _, Code>(&[in_vec], &mut rng, &merkleizer);
    let mut challenger = GC::default_challenger();
    let total_proof = zk_dot_product_proof::<GC, _, Code>(
        &dot_vec,
        &commitment,
        prover_secret_data,
        &mut challenger,
        &merkleizer,
    );

    assert_eq!(total_proof.proof.claimed_dot_products[0], expected);

    let mut challenger = GC::default_challenger();
    verify_zk_dot_product::<GC, Code>(&commitment, &dot_vec, &total_proof, &mut challenger)
        .unwrap();
}

/// Test batch dot products using RsInterpolation code.
#[tokio::test]
async fn test_zk_dot_products_interpolation_code() {
    const LENGTH: usize = 200;
    const NUM_DOT_VECS: usize = 10;
    type GC = KoalaBearDegree4Duplex;
    type Code = RsInterpolation<<GC as IopCtx>::EF>;

    let mut rng = ChaCha20Rng::from_entropy();
    let in_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
    let dot_vecs: Vec<Vec<<GC as IopCtx>::EF>> =
        std::iter::repeat_with(|| std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect())
            .take(NUM_DOT_VECS)
            .collect();

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) =
        zk_dot_product_commitment::<GC, _, _, Code>(&[in_vec], &mut rng, &merkleizer);
    let mut challenger = GC::default_challenger();
    let total_proof = zk_dot_products_proof::<GC, _, Code>(
        &dot_vecs,
        commitment,
        prover_secret_data,
        &mut challenger,
        &merkleizer,
    );

    let mut challenger = GC::default_challenger();
    verify_zk_dot_products::<GC, Code>(&commitment, &dot_vecs, &total_proof, &mut challenger)
        .unwrap();
}

/// Test dot product with all-zero input vector.
#[tokio::test]
async fn test_zk_dot_product_zero_input() {
    const LENGTH: usize = 100;
    type GC = KoalaBearDegree4Duplex;
    type Code = RsFromCoefficients<<GC as IopCtx>::EF>;

    let mut rng = ChaCha20Rng::from_entropy();
    let in_vec: Vec<<GC as IopCtx>::EF> = vec![<GC as IopCtx>::EF::zero(); LENGTH];
    let dot_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
    let expected = dot_product(&in_vec, &dot_vec);
    assert_eq!(expected, <GC as IopCtx>::EF::zero());

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) =
        zk_dot_product_commitment::<GC, _, _, Code>(&[in_vec], &mut rng, &merkleizer);
    let mut challenger = GC::default_challenger();
    let total_proof = zk_dot_product_proof::<GC, _, Code>(
        &dot_vec,
        &commitment,
        prover_secret_data,
        &mut challenger,
        &merkleizer,
    );

    assert_eq!(total_proof.proof.claimed_dot_products[0], <GC as IopCtx>::EF::zero());

    let mut challenger = GC::default_challenger();
    verify_zk_dot_product::<GC, Code>(&commitment, &dot_vec, &total_proof, &mut challenger)
        .unwrap();
}

/// Test dot product with all-one vectors.
#[tokio::test]
async fn test_zk_dot_product_all_ones() {
    const LENGTH: usize = 256;
    type GC = KoalaBearDegree4Duplex;
    type Code = RsFromCoefficients<<GC as IopCtx>::EF>;

    let mut rng = ChaCha20Rng::from_entropy();
    let in_vec: Vec<<GC as IopCtx>::EF> = vec![<GC as IopCtx>::EF::one(); LENGTH];
    let dot_vec: Vec<<GC as IopCtx>::EF> = vec![<GC as IopCtx>::EF::one(); LENGTH];
    let expected = dot_product(&in_vec, &dot_vec);

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) =
        zk_dot_product_commitment::<GC, _, _, Code>(&[in_vec], &mut rng, &merkleizer);
    let mut challenger = GC::default_challenger();
    let total_proof = zk_dot_product_proof::<GC, _, Code>(
        &dot_vec,
        &commitment,
        prover_secret_data,
        &mut challenger,
        &merkleizer,
    );

    assert_eq!(total_proof.proof.claimed_dot_products[0], expected);

    let mut challenger = GC::default_challenger();
    verify_zk_dot_product::<GC, Code>(&commitment, &dot_vec, &total_proof, &mut challenger)
        .unwrap();
}

/// Test with many batched input vectors (large batch).
#[tokio::test]
async fn test_zk_dot_product_large_batch_500_inputs() {
    const LENGTH: usize = 100;
    const NUM_INPUT_VECS: usize = 500;
    type GC = KoalaBearDegree4Duplex;
    type Code = RsFromCoefficients<<GC as IopCtx>::EF>;

    let mut rng = ChaCha20Rng::from_entropy();
    let in_vecs: Vec<Vec<<GC as IopCtx>::EF>> =
        std::iter::repeat_with(|| std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect())
            .take(NUM_INPUT_VECS)
            .collect();
    let dot_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) =
        zk_dot_product_commitment::<GC, _, _, Code>(&in_vecs, &mut rng, &merkleizer);
    let mut challenger = GC::default_challenger();
    let total_proof = zk_dot_product_proof::<GC, _, Code>(
        &dot_vec,
        &commitment,
        prover_secret_data,
        &mut challenger,
        &merkleizer,
    );

    let mut challenger = GC::default_challenger();
    verify_zk_dot_product::<GC, Code>(&commitment, &dot_vec, &total_proof, &mut challenger)
        .unwrap();
}

/// Test corruption of the commitment (wrong commitment given to verifier).
#[tokio::test]
async fn test_zk_dot_product_wrong_commitment() {
    const LENGTH: usize = 200;
    type GC = KoalaBearDegree4Duplex;
    type Code = RsFromCoefficients<<GC as IopCtx>::EF>;

    let mut rng = ChaCha20Rng::from_entropy();
    let in_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
    let dot_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) = zk_dot_product_commitment::<GC, _, _, Code>(
        std::slice::from_ref(&in_vec),
        &mut rng,
        &merkleizer,
    );
    let mut challenger = GC::default_challenger();
    let total_proof = zk_dot_product_proof::<GC, _, Code>(
        &dot_vec,
        &commitment,
        prover_secret_data,
        &mut challenger,
        &merkleizer,
    );

    // Create a different commitment from a different vector
    let in_vec2: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
    let (wrong_commitment, _) =
        zk_dot_product_commitment::<GC, _, _, Code>(&[in_vec2], &mut rng, &merkleizer);

    // Verification with wrong commitment should fail
    let mut challenger = GC::default_challenger();
    let result = verify_zk_dot_product::<GC, Code>(
        &wrong_commitment,
        &dot_vec,
        &total_proof,
        &mut challenger,
    );
    assert!(result.is_err(), "Verification should fail with wrong commitment");
}

/// Test various sizes in a parameterized loop using RsFromCoefficients.
#[tokio::test]
async fn test_zk_dot_product_various_sizes_coefficients() {
    type GC = KoalaBearDegree4Duplex;
    type Code = RsFromCoefficients<<GC as IopCtx>::EF>;

    for length in [2, 7, 16, 63, 128, 255, 1024] {
        let mut rng = ChaCha20Rng::from_entropy();
        let in_vec: Vec<<GC as IopCtx>::EF> =
            std::iter::repeat_with(|| rng.gen()).take(length).collect();
        let dot_vec: Vec<<GC as IopCtx>::EF> =
            std::iter::repeat_with(|| rng.gen()).take(length).collect();

        let merkleizer = Poseidon2KoalaBear16Prover::default();
        let (commitment, prover_secret_data) =
            zk_dot_product_commitment::<GC, _, _, Code>(&[in_vec], &mut rng, &merkleizer);
        let mut challenger = GC::default_challenger();
        let total_proof = zk_dot_product_proof::<GC, _, Code>(
            &dot_vec,
            &commitment,
            prover_secret_data,
            &mut challenger,
            &merkleizer,
        );

        let mut challenger = GC::default_challenger();
        verify_zk_dot_product::<GC, Code>(&commitment, &dot_vec, &total_proof, &mut challenger)
            .unwrap();
    }
}

/// Soundness test: corrupted dot product claim in single-vector proof.
#[tokio::test]
async fn test_zk_dot_product_corrupted_claim_single() {
    const LENGTH: usize = 500;
    type GC = KoalaBearDegree4Duplex;
    type Code = RsFromCoefficients<<GC as IopCtx>::EF>;

    let mut rng = ChaCha20Rng::from_entropy();
    let in_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
    let dot_vec: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();

    let merkleizer = Poseidon2KoalaBear16Prover::default();
    let (commitment, prover_secret_data) =
        zk_dot_product_commitment::<GC, _, _, Code>(&[in_vec], &mut rng, &merkleizer);
    let mut challenger = GC::default_challenger();
    let mut total_proof = zk_dot_product_proof::<GC, _, Code>(
        &dot_vec,
        &commitment,
        prover_secret_data,
        &mut challenger,
        &merkleizer,
    );

    // Corrupt the claimed dot product
    total_proof.proof.claimed_dot_products[0] += <GC as IopCtx>::EF::one();

    let mut challenger = GC::default_challenger();
    let result =
        verify_zk_dot_product::<GC, Code>(&commitment, &dot_vec, &total_proof, &mut challenger);
    assert!(result.is_err(), "Verification should fail with corrupted claim");
}

/// Test multi-dot with different number of dot vectors.
#[tokio::test]
async fn test_zk_dot_products_various_counts() {
    const LENGTH: usize = 200;
    type GC = KoalaBearDegree4Duplex;
    type Code = RsFromCoefficients<<GC as IopCtx>::EF>;

    for num_dot_vecs in [1, 2, 5, 20, 50] {
        let mut rng = ChaCha20Rng::from_entropy();
        let in_vec: Vec<<GC as IopCtx>::EF> =
            std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();
        let dot_vecs: Vec<Vec<<GC as IopCtx>::EF>> =
            std::iter::repeat_with(|| std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect())
                .take(num_dot_vecs)
                .collect();

        let merkleizer = Poseidon2KoalaBear16Prover::default();
        let (commitment, prover_secret_data) =
            zk_dot_product_commitment::<GC, _, _, Code>(&[in_vec], &mut rng, &merkleizer);
        let mut challenger = GC::default_challenger();
        let total_proof = zk_dot_products_proof::<GC, _, Code>(
            &dot_vecs,
            commitment,
            prover_secret_data,
            &mut challenger,
            &merkleizer,
        );

        let mut challenger = GC::default_challenger();
        verify_zk_dot_products::<GC, Code>(&commitment, &dot_vecs, &total_proof, &mut challenger)
            .unwrap();
    }
}
