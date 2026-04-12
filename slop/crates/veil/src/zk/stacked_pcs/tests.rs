use crate::zk::inner::{
    compute_mask_length, ConstraintContextInnerExt, MleCommitmentIndex, ZkCnstrAndReadingCtxInner,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use slop_challenger::IopCtx;
use slop_koala_bear::KoalaBearDegree4Duplex;
use slop_merkle_tree::Poseidon2KoalaBear16Prover;
use slop_multilinear::{Mle, Point};

use super::{
    initialize_zk_prover_and_verifier, prover::StackedPcsZkProverContext,
    utils::compute_padding_amount, verifier::StackedPcsZkVerificationContext,
};

type GC = KoalaBearDegree4Duplex;
type MK = Poseidon2KoalaBear16Prover;

/// Data read from transcript that mirrors prover's commitment and eval claim.
struct PcsTranscriptData<Expr> {
    commitment_index: MleCommitmentIndex,
    eval_claim: Expr,
}

/// Reads proof data from the transcript for a single evaluation claim.
fn read_all<C: ZkCnstrAndReadingCtxInner<GC>>(
    context: &mut C,
    num_vars: usize,
    log_num_polys: usize,
) -> PcsTranscriptData<C::Expr> {
    let commitment_index = context
        .read_next_pcs_commitment(num_vars, log_num_polys)
        .expect("Failed to read PCS commitment");

    let eval_claim = context.read_one().expect("Failed to read eval claim");

    PcsTranscriptData { commitment_index, eval_claim }
}

/// Uniform constraint generation function (called by both prover and verifier).
/// Registers a single evaluation claim for the commitment.
fn build_all_constraints<C: ConstraintContextInnerExt<<GC as IopCtx>::EF>>(
    transcript_data: PcsTranscriptData<C::Expr>,
    point: &Point<<GC as IopCtx>::EF>,
    context: &mut C,
) {
    context.assert_mle_eval(
        transcript_data.commitment_index,
        point.clone(),
        transcript_data.eval_claim,
    );
}

/// Helper to run the full ZK stacked PCS prove-verify workflow with one claim.
fn run_zk_stacked_pcs_test(num_encoding_variables: u32, log_num_polynomials: u32, verbose: bool) {
    let mut rng = ChaCha20Rng::from_entropy();

    let num_variables = log_num_polynomials + num_encoding_variables;

    if verbose {
        eprintln!("Test configuration:");
        eprintln!("  Total variables: {}", num_variables);
        eprintln!("  Log num polynomials: {}", log_num_polynomials);
        eprintln!("  Variables per column: {}", num_encoding_variables);
    }

    let original_mle = Mle::<<GC as IopCtx>::F>::rand(&mut rng, 1, num_variables);
    let eval_point = Point::<<GC as IopCtx>::EF>::rand(&mut rng, num_variables);
    let expected_eval = original_mle.eval_at(&eval_point);
    let expected_eval_value = expected_eval.evaluations().as_slice()[0];

    if verbose {
        eprintln!("  Expected evaluation: {:?}", expected_eval_value);
    }

    let (zk_basefold_prover, zk_stacked_verifier) =
        initialize_zk_prover_and_verifier::<GC, MK>(1, num_encoding_variables);

    let masks_length = compute_mask_length::<GC, _, _, _>(
        |ctx| read_all(ctx, num_encoding_variables as usize, log_num_polynomials as usize),
        |data, ctx| build_all_constraints(data, &eval_point, ctx),
    );

    // Prover Side
    let prover_start = std::time::Instant::now();
    let zkproof = {
        let mut prover_context: StackedPcsZkProverContext<GC, MK> =
            StackedPcsZkProverContext::initialize_only_lin_constraints(masks_length, &mut rng);

        let commitment_index = prover_context
            .commit_mle(&original_mle, log_num_polynomials as usize, &zk_basefold_prover, &mut rng)
            .expect("Failed to commit MLEs");

        let claim = prover_context.add_value(expected_eval_value);

        let transcript_data = PcsTranscriptData { commitment_index, eval_claim: claim };
        build_all_constraints(transcript_data, &eval_point, &mut prover_context);

        prover_context.prove(&mut rng, Some(&zk_basefold_prover))
    };
    if verbose {
        eprintln!("Prover time: {:?}", prover_start.elapsed());
    }

    // Verifier Side
    let verifier_start = std::time::Instant::now();
    {
        let mut context: StackedPcsZkVerificationContext<GC> = zkproof.open();

        let transcript_data =
            read_all(&mut context, num_encoding_variables as usize, log_num_polynomials as usize);
        build_all_constraints(transcript_data, &eval_point, &mut context);

        context.verify(Some(&zk_stacked_verifier)).expect("Failed to verify proof");
    }
    if verbose {
        eprintln!("Verifier time: {:?}", verifier_start.elapsed());
    }
}

#[test]
fn test_zk_stacked_pcs_commit_and_prove() {
    eprintln!("\n=== ZK Stacked PCS Commit and Prove Test ===");
    run_zk_stacked_pcs_test(14, 8, true);
    eprintln!("\n=== TEST PASSED ===");
}

#[test]
fn test_zk_stacked_pcs_small_mle() {
    eprintln!("Testing with small MLE");
    run_zk_stacked_pcs_test(12, 6, true);
    eprintln!("Small MLE test PASSED");
}

#[test]
fn test_zk_stacked_pcs_large_mle() {
    eprintln!("Testing with large MLE");
    run_zk_stacked_pcs_test(20, 8, true);
    eprintln!("Large MLE test PASSED");
}

#[test]
fn test_compute_padding_amount() {
    let (zk_prover, _) = initialize_zk_prover_and_verifier::<GC, MK>(1, 16);

    let codeword_length = 1 << 16;
    let security_bits = 100;
    let inverse_rate = 1 << zk_prover.inner.encoder.config().log_blowup;
    let padding = compute_padding_amount(inverse_rate, codeword_length, security_bits).unwrap();
    eprintln!("Corrected computed padding amount: {}", padding);

    let inverse_rate = 1 << zk_prover.inner.encoder.config().log_blowup;
    let rho = (inverse_rate as f64).recip();
    let b = security_bits as f64;
    let lambda = -(0.5 + 0.5 * rho).log2();
    let out64 = b / lambda;
    let standard_padding = out64.ceil() as usize;
    eprintln!("Standard computed padding amount: {}", standard_padding);
}

// ============================================================================
// New tests: various dimension combinations, multiple claims, edge cases
// ============================================================================

/// Test with minimal encoding variables and stacking.
#[test]
fn test_zk_stacked_pcs_minimal_dimensions() {
    eprintln!("Testing with minimal dimensions (10, 4)");
    run_zk_stacked_pcs_test(10, 4, true);
    eprintln!("Minimal dimensions test PASSED");
}

/// Test with large stacking height.
#[test]
fn test_zk_stacked_pcs_large_stacking() {
    eprintln!("Testing with large stacking height (12, 10)");
    run_zk_stacked_pcs_test(12, 10, true);
    eprintln!("Large stacking test PASSED");
}

/// Test with equal encoding and stacking dimensions.
#[test]
fn test_zk_stacked_pcs_equal_dimensions() {
    eprintln!("Testing with equal dimensions (10, 10)");
    run_zk_stacked_pcs_test(10, 10, true);
    eprintln!("Equal dimensions test PASSED");
}

/// Parametric test across several dimension combinations.
#[test]
fn test_zk_stacked_pcs_dimension_sweep() {
    let configs = [(10, 4), (12, 4), (14, 6), (10, 8), (16, 4)];
    for (enc, log_poly) in configs {
        eprintln!("Sweep: encoding_vars={}, log_num_polys={}", enc, log_poly);
        run_zk_stacked_pcs_test(enc, log_poly, false);
    }
    eprintln!("Dimension sweep PASSED");
}

/// Test that multiple evaluation claims on the same commitment panics
/// (known limitation: breaks zero-knowledge but not soundness).
#[test]
#[should_panic(expected = "Multiple eval claims on the same PCS commitment")]
fn test_zk_stacked_pcs_multiple_eval_claims() {
    use rand::Rng;

    let mut rng = ChaCha20Rng::from_entropy();

    let num_encoding_variables: u32 = 12;
    let log_num_polynomials: u32 = 6;
    let num_variables = log_num_polynomials + num_encoding_variables;

    let original_mle = Mle::<<GC as IopCtx>::F>::rand(&mut rng, 1, num_variables);

    // Two different evaluation points
    let eval_point_1 = Point::<<GC as IopCtx>::EF>::rand(&mut rng, num_variables);
    let eval_point_2 = Point::<<GC as IopCtx>::EF>::rand(&mut rng, num_variables);

    let expected_eval_1 = original_mle.eval_at(&eval_point_1).evaluations().as_slice()[0];
    let expected_eval_2 = original_mle.eval_at(&eval_point_2).evaluations().as_slice()[0];

    let (zk_basefold_prover, zk_stacked_verifier) =
        initialize_zk_prover_and_verifier::<GC, MK>(1, num_encoding_variables);

    /// Reads two evaluation claims from transcript.
    fn read_all<C: ZkCnstrAndReadingCtxInner<GC>>(
        context: &mut C,
    ) -> (MleCommitmentIndex, C::Expr, C::Expr) {
        let commitment_index = context.read_next_pcs_commitment(12, 6).unwrap();
        let eval_claim_1 = context.read_one().unwrap();
        let eval_claim_2 = context.read_one().unwrap();
        (commitment_index, eval_claim_1, eval_claim_2)
    }

    fn build_all_constraints<C: ConstraintContextInnerExt<<GC as IopCtx>::EF>>(
        commitment_index: MleCommitmentIndex,
        eval_claim_1: C::Expr,
        eval_claim_2: C::Expr,
        point_1: &Point<<GC as IopCtx>::EF>,
        point_2: &Point<<GC as IopCtx>::EF>,
        context: &mut C,
    ) {
        context.assert_mle_eval(commitment_index, point_1.clone(), eval_claim_1);
        context.assert_mle_eval(commitment_index, point_2.clone(), eval_claim_2);
    }

    let masks_length = compute_mask_length::<GC, _, _, _>(
        |ctx: &mut _| read_all(ctx),
        |data: (MleCommitmentIndex, _, _), ctx| {
            build_all_constraints(data.0, data.1, data.2, &eval_point_1, &eval_point_2, ctx)
        },
    );

    // Prover
    let zkproof = {
        let mut prover_context: StackedPcsZkProverContext<GC, MK> =
            StackedPcsZkProverContext::initialize_only_lin_constraints(masks_length, &mut rng);

        let commitment_index = prover_context
            .commit_mle(&original_mle, log_num_polynomials as usize, &zk_basefold_prover, &mut rng)
            .expect("Failed to commit MLE");

        let claim_1 = prover_context.add_value(expected_eval_1);
        let claim_2 = prover_context.add_value(expected_eval_2);

        build_all_constraints(
            commitment_index,
            claim_1,
            claim_2,
            &eval_point_1,
            &eval_point_2,
            &mut prover_context,
        );

        prover_context.prove(&mut rng, Some(&zk_basefold_prover))
    };

    // Verifier
    {
        let mut context: StackedPcsZkVerificationContext<GC> = zkproof.open();

        let (commitment_index, claim_1, claim_2) = read_all(&mut context);
        build_all_constraints(
            commitment_index,
            claim_1,
            claim_2,
            &eval_point_1,
            &eval_point_2,
            &mut context,
        );

        context.verify(Some(&zk_stacked_verifier)).expect("Multiple claims verification failed");
    }
}

/// Test with two independent committed MLEs, each with one eval claim.
#[test]
fn test_zk_stacked_pcs_two_independent_mles() {
    use rand::Rng;

    let mut rng = ChaCha20Rng::from_entropy();

    let num_encoding_variables: u32 = 12;
    let log_num_polynomials: u32 = 6;
    let num_variables = log_num_polynomials + num_encoding_variables;

    let mle_1 = Mle::<<GC as IopCtx>::F>::rand(&mut rng, 1, num_variables);
    let mle_2 = Mle::<<GC as IopCtx>::F>::rand(&mut rng, 1, num_variables);

    let eval_point = Point::<<GC as IopCtx>::EF>::rand(&mut rng, num_variables);
    let expected_1 = mle_1.eval_at(&eval_point).evaluations().as_slice()[0];
    let expected_2 = mle_2.eval_at(&eval_point).evaluations().as_slice()[0];

    let (zk_basefold_prover, zk_stacked_verifier) =
        initialize_zk_prover_and_verifier::<GC, MK>(1, num_encoding_variables);

    fn read_all<C: ZkCnstrAndReadingCtxInner<GC>>(
        context: &mut C,
    ) -> (MleCommitmentIndex, MleCommitmentIndex, C::Expr, C::Expr) {
        let ci1 = context.read_next_pcs_commitment(12, 6).unwrap();
        let ci2 = context.read_next_pcs_commitment(12, 6).unwrap();
        let claim1 = context.read_one().unwrap();
        let claim2 = context.read_one().unwrap();
        (ci1, ci2, claim1, claim2)
    }

    fn build_constraints<C: ConstraintContextInnerExt<<GC as IopCtx>::EF>>(
        ci1: MleCommitmentIndex,
        ci2: MleCommitmentIndex,
        claim1: C::Expr,
        claim2: C::Expr,
        point: &Point<<GC as IopCtx>::EF>,
        ctx: &mut C,
    ) {
        ctx.assert_mle_eval(ci1, point.clone(), claim1);
        ctx.assert_mle_eval(ci2, point.clone(), claim2);
    }

    let masks_length = compute_mask_length::<GC, _, _, _>(
        |ctx: &mut _| read_all(ctx),
        |data: (_, _, _, _), ctx| {
            build_constraints(data.0, data.1, data.2, data.3, &eval_point, ctx)
        },
    );

    let zkproof = {
        let mut prover_context: StackedPcsZkProverContext<GC, MK> =
            StackedPcsZkProverContext::initialize_only_lin_constraints(masks_length, &mut rng);

        let ci1 = prover_context
            .commit_mle(&mle_1, log_num_polynomials as usize, &zk_basefold_prover, &mut rng)
            .expect("commit mle_1");
        let ci2 = prover_context
            .commit_mle(&mle_2, log_num_polynomials as usize, &zk_basefold_prover, &mut rng)
            .expect("commit mle_2");

        let claim1 = prover_context.add_value(expected_1);
        let claim2 = prover_context.add_value(expected_2);

        build_constraints(ci1, ci2, claim1, claim2, &eval_point, &mut prover_context);

        prover_context.prove(&mut rng, Some(&zk_basefold_prover))
    };

    {
        let mut context: StackedPcsZkVerificationContext<GC> = zkproof.open();
        let (ci1, ci2, claim1, claim2) = read_all(&mut context);
        build_constraints(ci1, ci2, claim1, claim2, &eval_point, &mut context);
        context.verify(Some(&zk_stacked_verifier)).expect("Two MLEs verification failed");
    }
}
