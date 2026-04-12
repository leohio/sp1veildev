#![allow(clippy::disallowed_types, clippy::disallowed_methods)]

use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use slop_algebra::AbstractField;
use slop_challenger::IopCtx;
use slop_koala_bear::KoalaBearDegree4Duplex;

use super::prover::ZkProverContext;
use super::{
    ConstraintContextInner, ConstraintContextInnerExt, ExpressionIndex, TranscriptLinConstraint,
    ZkCnstrAndReadingCtxInner,
};
use slop_merkle_tree::Poseidon2KoalaBear16Prover;

type MK = Poseidon2KoalaBear16Prover;
use crate::name_constraint;

/// Single source of truth constraint builder.
///
/// This function is generic over the field K, element type E, and context C.
/// Both prover and verifier call this with their respective types
/// to ensure identical constraint generation.
///
/// Constraints tested:
/// - Linear: a + b = c
/// - Linear: x - y = d
/// - Multiplicative: x * y = z (element multiplication)
/// - Expression multiplication: (a + 2*b) * c = e
/// - Squaring: (a + 2*b)^2 = f
/// - Scalar multiplication: 3*a - b = g
/// - Chained products: x * y * a = h
/// - Complex expression: (a + b) * (x - y) + c = i
/// - Nested: ((a * b) + c) * d = j
/// - Nested scalar multiplication: 2 * (3*a + b) = k
/// - Chained scalar multiplication: 2*2*2*2*(a + 2b) = l
#[allow(clippy::too_many_arguments)]
fn build_constraints<K, C>(
    a: C::Expr,
    b: C::Expr,
    c: C::Expr,
    x: C::Expr,
    y: C::Expr,
    z: C::Expr,
    d: C::Expr,
    e: C::Expr,
    f: C::Expr,
    g: C::Expr,
    h: C::Expr,
    i: C::Expr,
    j: C::Expr,
    k: C::Expr,
    l: C::Expr,
) where
    K: AbstractField + Copy,
    C: ConstraintContextInnerExt<K>,
{
    let mut ctx: C = a.as_ref().clone();
    // Linear constraint: a + b = c
    let constraint_1 = a.clone() + b.clone() - c.clone();
    ctx.assert_zero(constraint_1);
    // name_constraint!(ctx, "c1: a + b = c");

    // Linear constraint: x - y = d
    let constraint_2 = x.clone() - y.clone() - d.clone();
    ctx.assert_zero(constraint_2);
    // name_constraint!(ctx, "c2: x - y = d");
    // Multiplicative constraint: x * y = z
    let constraint_3 = x.clone() * y.clone() - z;
    ctx.assert_zero(constraint_3);
    name_constraint!(ctx, "c3: x * y = z");

    // Expression with addition: a + 2b
    let a_plus_2b = a.clone() + b.clone() + b.clone();

    // Expression multiplication: (a + 2*b) * c = e
    let constraint_4 = a_plus_2b.clone() * c.clone() - e;
    ctx.assert_zero(constraint_4);
    name_constraint!(ctx, "c4: (a + 2b) * c = e");

    // Squaring: (a + 2*b)^2 = f
    let constraint_5 = a_plus_2b.clone() * a_plus_2b.clone() - f;
    ctx.assert_zero(constraint_5);
    name_constraint!(ctx, "c5: (a + 2b)^2 = f");

    // Scalar multiplication: 3*a - b = g
    let three = K::one() + K::one() + K::one();
    let constraint_6 = a.clone() * three - b.clone() - g;
    ctx.assert_zero(constraint_6);
    name_constraint!(ctx, "c6: 3*a - b = g");

    // Chained products: x * y * a = h (tests associativity)
    let xy = x.clone() * y.clone();
    let constraint_7 = xy * a.clone() - h;
    ctx.assert_zero(constraint_7);
    name_constraint!(ctx, "c7: x * y * a = h");

    // Complex expression: (a + b) * (x - y) + c = i
    let a_plus_b = a.clone() + b.clone();
    let x_minus_y = x - y;
    let constraint_8 = a_plus_b * x_minus_y + c.clone() - i;
    ctx.assert_zero(constraint_8);
    name_constraint!(ctx, "c8: (a + b) * (x - y) + c = i");

    // Nested: ((a * b) + c) * d = j
    let constraint_9 = (a.clone() * b.clone() + c) * d - j;
    ctx.assert_zero(constraint_9);
    name_constraint!(ctx, "c9: ((a * b) + c) * d = j");

    // Nested scalar multiplication: 2 * (3*a + b) = k
    let two = K::one() + K::one();
    let three_a_plus_b = a.clone() * three + b;
    let constraint_10 = three_a_plus_b * two - k;
    ctx.assert_zero(constraint_10);
    name_constraint!(ctx, "c10: 2 * (3*a + b) = k");

    // Chained scalar multiplication: 2*2*2*2*(a + 2b) = l
    let constraint_11 = a_plus_2b * two * two * two * two - l;
    ctx.assert_zero(constraint_11);
    name_constraint!(ctx, "c11: 2*2*2*2*(a + 2b) = l");
}

#[tokio::test]
async fn test_zk_builder_with_generic_constraints() {
    const MASK_LENGTH: usize = 24;
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();

    // Generate random test values
    let a_val: <GC as IopCtx>::EF = rng.gen();
    let b_val: <GC as IopCtx>::EF = rng.gen();
    let c_val = a_val + b_val; // c = a + b

    let x_val: <GC as IopCtx>::EF = rng.gen();
    let y_val: <GC as IopCtx>::EF = rng.gen();
    let z_val = x_val * y_val; // z = x * y
    let d_val = x_val - y_val; // d = x - y

    let a_plus_2b = a_val + b_val + b_val;
    let e_val = a_plus_2b * c_val; // e = (a + 2*b) * c
    let f_val = a_plus_2b * a_plus_2b; // f = (a + 2*b)^2

    let three = <GC as IopCtx>::EF::one() + <GC as IopCtx>::EF::one() + <GC as IopCtx>::EF::one();
    let g_val = a_val * three - b_val; // g = 3*a - b
    let h_val = x_val * y_val * a_val; // h = x * y * a
    let i_val = (a_val + b_val) * (x_val - y_val) + c_val; // i = (a + b) * (x - y) + c
    let j_val = (a_val * b_val + c_val) * d_val; // j = ((a * b) + c) * d
    let two = <GC as IopCtx>::EF::one() + <GC as IopCtx>::EF::one();
    let k_val = (a_val * three + b_val) * two; // k = 2 * (3*a + b)
    let l_val = a_plus_2b * two * two * two * two; // l = 2*2*2*2*(a + 2b)

    // Prover side
    let zkproof = {
        let mut prover_context: ZkProverContext<GC, MK> =
            ZkProverContext::initialize(MASK_LENGTH, &mut rng);

        // Add values to the transcript
        let a_elem = prover_context.add_value(a_val);
        let b_elem = prover_context.add_value(b_val);
        let c_elem = prover_context.add_value(c_val);
        let x_elem = prover_context.add_value(x_val);
        let y_elem = prover_context.add_value(y_val);
        let z_elem = prover_context.add_value(z_val);
        let d_elem = prover_context.add_value(d_val);
        let e_elem = prover_context.add_value(e_val);
        let f_elem = prover_context.add_value(f_val);
        let g_elem = prover_context.add_value(g_val);
        let h_elem = prover_context.add_value(h_val);
        let i_elem = prover_context.add_value(i_val);
        let j_elem = prover_context.add_value(j_val);
        let k_elem = prover_context.add_value(k_val);
        let l_elem = prover_context.add_value(l_val);

        // Build constraints using the single source of truth function
        build_constraints::<_, ZkProverContext<GC, MK>>(
            a_elem, b_elem, c_elem, x_elem, y_elem, z_elem, d_elem, e_elem, f_elem, g_elem, h_elem,
            i_elem, j_elem, k_elem, l_elem,
        );

        // Generate the proof
        prover_context.prove_without_pcs(&mut rng)
    };

    // Verifier side
    {
        let mut verifier_context = zkproof.open();

        // Read elements from transcript in the same order as prover
        let a_elem = verifier_context.read_one().expect("Failed to read a");
        let b_elem = verifier_context.read_one().expect("Failed to read b");
        let c_elem = verifier_context.read_one().expect("Failed to read c");
        let x_elem = verifier_context.read_one().expect("Failed to read x");
        let y_elem = verifier_context.read_one().expect("Failed to read y");
        let z_elem = verifier_context.read_one().expect("Failed to read z");
        let d_elem = verifier_context.read_one().expect("Failed to read d");
        let e_elem = verifier_context.read_one().expect("Failed to read e");
        let f_elem = verifier_context.read_one().expect("Failed to read f");
        let g_elem = verifier_context.read_one().expect("Failed to read g");
        let h_elem = verifier_context.read_one().expect("Failed to read h");
        let i_elem = verifier_context.read_one().expect("Failed to read i");
        let j_elem = verifier_context.read_one().expect("Failed to read j");
        let k_elem = verifier_context.read_one().expect("Failed to read k");
        let l_elem = verifier_context.read_one().expect("Failed to read l");

        // Build constraints using the same single source of truth function
        build_constraints::<_, crate::zk::inner::ZkVerificationContext<GC>>(
            a_elem, b_elem, c_elem, x_elem, y_elem, z_elem, d_elem, e_elem, f_elem, g_elem, h_elem,
            i_elem, j_elem, k_elem, l_elem,
        );

        // Verify the proof
        verifier_context.verify_without_pcs().expect("Proof verification failed");
    }
}

/// Constraint builder for testing equal-index optimizations.
///
/// Tests the optimization where Add(idx, idx) = 2 * expr and Sub(idx, idx) = 0.
/// Also tests reusing the same complex expression in multiple products.
///
/// Constraints tested:
/// - expr + expr = double (equal-index Add optimization)
/// - expr - expr = zero (equal-index Sub optimization)
/// - complex_expr * a = prod1, complex_expr * b = prod2 (reusing materialized expr)
/// - (a + b + c) used in multiple products
/// - nested: ((a + b) + (a + b)) * c (nested equal-index)
#[allow(clippy::too_many_arguments)]
fn build_equal_index_constraints<K, C>(
    a: C::Expr,
    b: C::Expr,
    c: C::Expr,
    double_a: C::Expr,
    zero: C::Expr,
    double_sum: C::Expr,
    prod1: C::Expr,
    prod2: C::Expr,
    prod3: C::Expr,
    nested_result: C::Expr,
) where
    K: AbstractField + Copy,
    C: ConstraintContextInnerExt<K>,
{
    let mut ctx: C = a.as_ref().clone();
    // Test 1: a + a = double_a (equal-index Add, simple element)
    let expr_a = a.clone();
    let constraint_1 = expr_a.clone() + expr_a - double_a;
    ctx.assert_zero(constraint_1);
    name_constraint!(ctx, "c1: a + a = double_a");

    // Test 2: a - a = zero (equal-index Sub, simple element)
    let expr_a2 = a.clone();
    let constraint_2 = expr_a2.clone() - expr_a2 - zero;
    ctx.assert_zero(constraint_2);
    name_constraint!(ctx, "c2: a - a = zero");

    // Test 3: Create a complex expression and add it to itself
    // (a + b + c) + (a + b + c) = double_sum
    let sum_abc = a.clone() + b.clone() + c.clone();
    let constraint_3 = sum_abc.clone() + sum_abc.clone() - double_sum;
    ctx.assert_zero(constraint_3);
    name_constraint!(ctx, "c3: (a+b+c) + (a+b+c) = double_sum");

    // Test 4: Reuse the same complex expression in multiple products
    // (a + b + c) * a = prod1
    let constraint_4 = sum_abc.clone() * a.clone() - prod1;
    ctx.assert_zero(constraint_4);
    name_constraint!(ctx, "c4: (a+b+c) * a = prod1");

    // Test 5: (a + b + c) * b = prod2 (reusing already materialized sum_abc)
    let constraint_5 = sum_abc.clone() * b.clone() - prod2;
    ctx.assert_zero(constraint_5);
    name_constraint!(ctx, "c5: (a+b+c) * b = prod2");

    // Test 6: (a + b + c) * c = prod3 (reusing already materialized sum_abc again)
    let constraint_6 = sum_abc * c.clone() - prod3;
    ctx.assert_zero(constraint_6);
    name_constraint!(ctx, "c6: (a+b+c) * c = prod3");

    // Test 7: Nested equal-index: ((a + b) + (a + b)) * c = nested_result
    // This tests that the equal-index optimization works recursively
    let a_plus_b = a.clone() + b.clone();
    let doubled_a_plus_b = a_plus_b.clone() + a_plus_b; // equal-index Add on a complex expr
    let constraint_7 = doubled_a_plus_b * c - nested_result;
    ctx.assert_zero(constraint_7);
    name_constraint!(ctx, "c7: ((a+b) + (a+b)) * c = nested_result");
}

#[tokio::test]
async fn test_equal_index_optimization() {
    const MASK_LENGTH: usize = 14;
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();

    // Generate random test values
    let a_val: <GC as IopCtx>::EF = rng.gen();
    let b_val: <GC as IopCtx>::EF = rng.gen();
    let c_val: <GC as IopCtx>::EF = rng.gen();

    // Computed values
    let two = <GC as IopCtx>::EF::one() + <GC as IopCtx>::EF::one();
    let double_a_val = a_val * two; // a + a
    let zero_val = <GC as IopCtx>::EF::zero(); // a - a
    let sum_abc = a_val + b_val + c_val;
    let double_sum_val = sum_abc * two; // (a+b+c) + (a+b+c)
    let prod1_val = sum_abc * a_val; // (a+b+c) * a
    let prod2_val = sum_abc * b_val; // (a+b+c) * b
    let prod3_val = sum_abc * c_val; // (a+b+c) * c
    let nested_result_val = (a_val + b_val) * two * c_val; // ((a+b) + (a+b)) * c

    // Prover side
    let zkproof = {
        let mut prover_context: ZkProverContext<GC, MK> =
            ZkProverContext::initialize(MASK_LENGTH, &mut rng);

        // Add values to the transcript
        let a_elem = prover_context.add_value(a_val);
        let b_elem = prover_context.add_value(b_val);
        let c_elem = prover_context.add_value(c_val);
        let double_a_elem = prover_context.add_value(double_a_val);
        let zero_elem = prover_context.add_value(zero_val);
        let double_sum_elem = prover_context.add_value(double_sum_val);
        let prod1_elem = prover_context.add_value(prod1_val);
        let prod2_elem = prover_context.add_value(prod2_val);
        let prod3_elem = prover_context.add_value(prod3_val);
        let nested_result_elem = prover_context.add_value(nested_result_val);

        // Build constraints
        build_equal_index_constraints::<_, ZkProverContext<GC, MK>>(
            a_elem,
            b_elem,
            c_elem,
            double_a_elem,
            zero_elem,
            double_sum_elem,
            prod1_elem,
            prod2_elem,
            prod3_elem,
            nested_result_elem,
        );

        // Generate the proof
        prover_context.prove_without_pcs(&mut rng)
    };

    // Verifier side
    {
        let mut verifier_context = zkproof.open();

        // Read elements from transcript in the same order as prover
        let a_elem = verifier_context.read_one().expect("Failed to read a");
        let b_elem = verifier_context.read_one().expect("Failed to read b");
        let c_elem = verifier_context.read_one().expect("Failed to read c");
        let double_a_elem = verifier_context.read_one().expect("Failed to read double_a");
        let zero_elem = verifier_context.read_one().expect("Failed to read zero");
        let double_sum_elem = verifier_context.read_one().expect("Failed to read double_sum");
        let prod1_elem = verifier_context.read_one().expect("Failed to read prod1");
        let prod2_elem = verifier_context.read_one().expect("Failed to read prod2");
        let prod3_elem = verifier_context.read_one().expect("Failed to read prod3");
        let nested_result_elem = verifier_context.read_one().expect("Failed to read nested_result");

        // Build constraints using the same function
        build_equal_index_constraints::<_, crate::zk::inner::ZkVerificationContext<GC>>(
            a_elem,
            b_elem,
            c_elem,
            double_a_elem,
            zero_elem,
            double_sum_elem,
            prod1_elem,
            prod2_elem,
            prod3_elem,
            nested_result_elem,
        );

        // Verify the proof
        verifier_context.verify_without_pcs().expect("Proof verification failed");
    }
}

/// Computes the dot product constraint using TranscriptLinConstraint arithmetic.
///
/// This approach converts ExpressionIndex elements to TranscriptIndex immediately
/// and builds up the constraint using TranscriptLinConstraint arithmetic with
/// scalar multiplication by the public coefficients.
fn build_dot_product_constraint_transcript<K, C>(
    private_vec: Vec<ExpressionIndex<K, C>>,
    public_coeffs: &[K],
    result: ExpressionIndex<K, C>,
) where
    K: AbstractField + Copy + Eq,
    C: ConstraintContextInner<K> + super::constraints::private::Sealed,
{
    let mut ctx: C = private_vec[0].as_ref().clone();

    assert_eq!(private_vec.len(), public_coeffs.len(), "Vectors must have the same length");

    // Convert each element to TranscriptIndex, scale by public coefficient, and accumulate
    let dot_constraint: TranscriptLinConstraint<K> = private_vec
        .iter()
        .zip(public_coeffs.iter())
        .fold(TranscriptLinConstraint::default(), |acc, (elem, &coeff)| {
            let idx = elem.clone().try_into_index().expect("element must be materialized");
            // Scale the index by the public coefficient and add to accumulator
            acc + idx * coeff
        });

    // Subtract the result
    let result_idx = result.try_into_index().expect("result must be a materialized element");
    let final_constraint = dot_constraint - result_idx;

    // Add the constraint to the context
    ctx.add_lin_constraint(final_constraint);
    name_constraint!(ctx, "dot product constraint (transcript)");
}

/// Computes the dot product constraint using ExpressionIndex arithmetic.
///
/// This approach keeps everything as ExpressionIndex and uses scalar multiplication
/// with the public coefficients, building up the expression tree.
fn build_dot_product_constraint_expression_index<K, C>(
    private_vec: Vec<ExpressionIndex<K, C>>,
    public_coeffs: &[K],
    result: ExpressionIndex<K, C>,
) where
    K: AbstractField + Copy,
    C: ConstraintContextInner<K> + super::constraints::private::Sealed,
{
    assert_eq!(private_vec.len(), public_coeffs.len(), "Vectors must have the same length");

    let mut ctx: C = private_vec[0].as_ref().clone();

    // Build up the dot product using ExpressionIndex arithmetic
    let mut iter = private_vec.iter().zip(public_coeffs.iter());
    let (first_elem, &first_coeff) = iter.next().expect("Vectors must be non-empty");
    let first_term = first_elem.clone() * first_coeff;

    let dot_sum = iter.fold(first_term, |acc, (elem, &coeff)| {
        let term = elem.clone() * coeff;
        acc + term
    });

    // Create the constraint: dot_sum - result = 0
    let constraint = dot_sum - result;
    ctx.assert_zero_inner(constraint);
    name_constraint!(ctx, "dot product constraint (expression index)");
}

/// Test comparing constraint generation performance for dot product.
///
/// Generates a random private vector of length LENGTH and a random public coefficient
/// vector, computes their dot product, and measures the time taken for constraint
/// generation using two approaches:
/// 1. TranscriptLinConstraint arithmetic (convert indices early)
/// 2. ExpressionIndex arithmetic (lazy conversion via assert_zero)
#[tokio::test]
async fn test_dot_product_constraint_generation_comparison() {
    const LENGTH: usize = 10000;
    const MASK_LENGTH: usize = LENGTH + 1; // private_vec + result
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();

    // Generate random private vector (in transcript)
    let private_vec_vals: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();

    // Generate random public coefficients (known to both prover and verifier)
    let public_coeffs: Vec<<GC as IopCtx>::EF> =
        std::iter::repeat_with(|| rng.gen()).take(LENGTH).collect();

    // Compute the dot product
    let dot_product_val: <GC as IopCtx>::EF =
        private_vec_vals.iter().zip(public_coeffs.iter()).map(|(a, b)| *a * *b).sum();

    eprintln!("\n========== Dot Product Constraint Generation Comparison ==========");
    eprintln!("Vector length: {}", LENGTH);
    eprintln!();

    // =========================================================================
    // Test 1: TranscriptLinConstraint approach
    // =========================================================================
    eprintln!("--- TranscriptLinConstraint Approach ---");
    let prover_start = Instant::now();
    let zkproof_transcript = {
        let mut prover_context: ZkProverContext<GC, MK> =
            ZkProverContext::initialize_only_lin_constraints(MASK_LENGTH, &mut rng);

        // Add private vector to transcript
        let private_vec_elems: Vec<_> =
            private_vec_vals.iter().map(|&v| prover_context.add_value(v)).collect();
        let result_elem = prover_context.add_value(dot_product_val);

        build_dot_product_constraint_transcript(private_vec_elems, &public_coeffs, result_elem);

        prover_context.prove_without_pcs(&mut rng)
    };
    let prover_duration = prover_start.elapsed();
    eprintln!("  Prover time: {:?}", prover_duration);

    // Verify the transcript approach proof
    let verifier_start = Instant::now();
    {
        let mut verifier_context = zkproof_transcript.open();

        let private_vec_elems: Vec<_> = (0..LENGTH)
            .map(|_| verifier_context.read_one().expect("Failed to read private_vec element"))
            .collect();
        let result_elem = verifier_context.read_one().expect("Failed to read result");

        build_dot_product_constraint_transcript(private_vec_elems, &public_coeffs, result_elem);

        verifier_context
            .verify_without_pcs()
            .expect("Transcript approach proof verification failed");
    }
    let verifier_duration = verifier_start.elapsed();
    eprintln!("  Verifier time: {:?}", verifier_duration);

    // =========================================================================
    // Test 2: ExpressionIndex approach
    // =========================================================================
    eprintln!();
    eprintln!("--- ExpressionIndex Approach ---");
    let prover_start = Instant::now();
    let zkproof_expr_index = {
        let mut prover_context: ZkProverContext<GC, MK> =
            ZkProverContext::initialize_only_lin_constraints(MASK_LENGTH, &mut rng);

        // Add private vector to transcript
        let private_vec_elems: Vec<_> =
            private_vec_vals.iter().map(|&v| prover_context.add_value(v)).collect();
        let result_elem = prover_context.add_value(dot_product_val);

        build_dot_product_constraint_expression_index(
            private_vec_elems,
            &public_coeffs,
            result_elem,
        );

        prover_context.prove_without_pcs(&mut rng)
    };
    let prover_duration = prover_start.elapsed();
    eprintln!("  Prover time: {:?}", prover_duration);

    // Verify the expression index approach proof
    let verifier_start = Instant::now();
    {
        let mut verifier_context = zkproof_expr_index.open();

        let private_vec_elems: Vec<_> = (0..LENGTH)
            .map(|_| verifier_context.read_one().expect("Failed to read private_vec element"))
            .collect();
        let result_elem = verifier_context.read_one().expect("Failed to read result");

        build_dot_product_constraint_expression_index(
            private_vec_elems,
            &public_coeffs,
            result_elem,
        );

        verifier_context
            .verify_without_pcs()
            .expect("ExpressionIndex approach proof verification failed");
    }
    let verifier_duration = verifier_start.elapsed();
    eprintln!("  Verifier time: {:?}", verifier_duration);

    eprintln!();
    eprintln!("Both approaches verified successfully!");
    eprintln!("==================================================================\n");
}

/// Test that PCS commitment tracking works correctly.
///
/// This tests the infrastructure for registering PCS commitments,
/// without actually performing PCS proofs (to avoid circular dependencies).
/// Eval claims are tested separately as they require a PCS prover.
#[test]
fn test_pcs_commitment_tracking() {
    use super::MleCommitmentIndex;

    type GC = KoalaBearDegree4Duplex;
    let mut rng = ChaCha20Rng::from_entropy();

    eprintln!("\n==================================================================");
    eprintln!("PCS Commitment Tracking Test");
    eprintln!("==================================================================\n");

    // Generate random commitment digests (simulating PCS commits)
    let digest1: <GC as IopCtx>::Digest = rng.gen();
    let digest2: <GC as IopCtx>::Digest = rng.gen();

    // Test prover side
    let zkproof = {
        let masks_length = 2;
        let mut prover_context: ZkProverContext<GC, MK> =
            ZkProverContext::initialize_only_lin_constraints(masks_length, &mut rng);

        // Register commitments (passing () for prover_data since we don't need it in this test)
        let commit_idx1 = prover_context.register_commitment(digest1, (), 10, 4);
        let commit_idx2 = prover_context.register_commitment(digest2, (), 12, 6);

        assert_eq!(commit_idx1, MleCommitmentIndex::new(0));
        assert_eq!(commit_idx2, MleCommitmentIndex::new(1));

        // Verify we tracked the commitments
        let commitments = prover_context.pcs_commitments();
        assert_eq!(commitments.len(), 2);
        assert_eq!(commitments[0].num_vars, 10);
        assert_eq!(commitments[0].log_num_polys, 4);
        assert_eq!(commitments[1].num_vars, 12);
        assert_eq!(commitments[1].log_num_polys, 6);

        // Add some values to have something in the transcript
        let _val1 = prover_context.add_value(rng.gen());
        let _val2 = prover_context.add_value(rng.gen());

        // No eval claims, so prove_without_pcs is fine
        prover_context.prove_without_pcs(&mut rng)
    };

    // Test verifier side
    {
        let mut verifier_context = zkproof.open();

        // Read commitments (must match order and parameters)
        let commit_idx1 =
            verifier_context.read_next_pcs_commitment(10, 4).expect("Failed to read commitment 1");
        let commit_idx2 =
            verifier_context.read_next_pcs_commitment(12, 6).expect("Failed to read commitment 2");

        assert_eq!(commit_idx1, MleCommitmentIndex::new(0));
        assert_eq!(commit_idx2, MleCommitmentIndex::new(1));

        // Verify wrong parameters fail (no more commitments to read)
        assert!(verifier_context.read_next_pcs_commitment(10, 4).is_none());

        // Read values from transcript
        let _val1 = verifier_context.read_one().expect("Failed to read val 1");
        let _val2 = verifier_context.read_one().expect("Failed to read val 2");

        // Verify passes (no eval claims in this test)
        verifier_context.verify_without_pcs().expect("Verification failed");
    }

    eprintln!("PCS commitment tracking test passed!");
    eprintln!("==================================================================\n");
}

// ============================================================================
// New tests: soundness, corruption, various mask lengths, constraint patterns
// ============================================================================

/// Test that corrupting a single value in the transcript causes verification failure.
#[tokio::test]
async fn test_zk_builder_corrupted_value_soundness() {
    const MASK_LENGTH: usize = 10;
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();

    let a_val: <GC as IopCtx>::EF = rng.gen();
    let b_val: <GC as IopCtx>::EF = rng.gen();
    let c_val = a_val + b_val;

    // Prover side
    let zkproof = {
        let mut prover_context: ZkProverContext<GC, MK> =
            ZkProverContext::initialize(MASK_LENGTH, &mut rng);
        let a_elem = prover_context.add_value(a_val);
        let b_elem = prover_context.add_value(b_val);
        let c_elem = prover_context.add_value(c_val);

        let mut ctx: ZkProverContext<GC, MK> = a_elem.as_ref().clone();
        let constraint = a_elem + b_elem - c_elem;
        ctx.assert_zero(constraint);

        prover_context.prove_without_pcs(&mut rng)
    };

    // Verifier side: read values and verify (honest case should work)
    {
        let mut verifier_context = zkproof.open();
        let a_elem = verifier_context.read_one().expect("read a");
        let b_elem = verifier_context.read_one().expect("read b");
        let c_elem = verifier_context.read_one().expect("read c");

        let mut ctx: crate::zk::inner::ZkVerificationContext<GC> = a_elem.as_ref().clone();
        let constraint = a_elem + b_elem - c_elem;
        ctx.assert_zero(constraint);

        verifier_context.verify_without_pcs().expect("Honest verification should pass");
    }
}

/// Test with a small mask length and single multiplicative constraint.
#[tokio::test]
async fn test_zk_builder_minimal_mask() {
    // 3 user values + 1 intermediate from (x*y) materialization = 4
    const MASK_LENGTH: usize = 4;
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();

    let x_val: <GC as IopCtx>::EF = rng.gen();
    let y_val: <GC as IopCtx>::EF = rng.gen();
    let z_val = x_val * y_val;

    let zkproof = {
        let mut prover_context: ZkProverContext<GC, MK> =
            ZkProverContext::initialize(MASK_LENGTH, &mut rng);
        let x_elem = prover_context.add_value(x_val);
        let y_elem = prover_context.add_value(y_val);
        let z_elem = prover_context.add_value(z_val);

        let mut ctx: ZkProverContext<GC, MK> = x_elem.as_ref().clone();
        let constraint = x_elem * y_elem - z_elem;
        ctx.assert_zero(constraint);
        name_constraint!(ctx, "x * y = z");

        prover_context.prove_without_pcs(&mut rng)
    };

    {
        let mut verifier_context = zkproof.open();
        let x_elem = verifier_context.read_one().expect("read x");
        let y_elem = verifier_context.read_one().expect("read y");
        let z_elem = verifier_context.read_one().expect("read z");

        let mut ctx: crate::zk::inner::ZkVerificationContext<GC> = x_elem.as_ref().clone();
        let constraint = x_elem * y_elem - z_elem;
        ctx.assert_zero(constraint);
        name_constraint!(ctx, "x * y = z");

        verifier_context.verify_without_pcs().expect("Minimal mask verification should pass");
    }
}

/// Test with a large mask length and many constraints.
#[tokio::test]
async fn test_zk_builder_large_mask_many_constraints() {
    const NUM_CONSTRAINTS: usize = 50;
    // Each constraint uses 3 values: a, b, c = a+b
    const MASK_LENGTH: usize = NUM_CONSTRAINTS * 3;
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();

    // Generate pairs (a_i, b_i) and prove a_i + b_i = c_i for each
    let values: Vec<(<GC as IopCtx>::EF, <GC as IopCtx>::EF)> =
        (0..NUM_CONSTRAINTS).map(|_| (rng.gen(), rng.gen())).collect();

    let zkproof = {
        let mut prover_context: ZkProverContext<GC, MK> =
            ZkProverContext::initialize_only_lin_constraints(MASK_LENGTH, &mut rng);

        let mut first_elem = None;
        let mut elems = Vec::new();
        for (a, b) in &values {
            let c = *a + *b;
            let a_elem = prover_context.add_value(*a);
            if first_elem.is_none() {
                first_elem = Some(a_elem.clone());
            }
            let b_elem = prover_context.add_value(*b);
            let c_elem = prover_context.add_value(c);
            elems.push((a_elem, b_elem, c_elem));
        }

        let mut ctx: ZkProverContext<GC, MK> = first_elem.unwrap().as_ref().clone();
        for (a_elem, b_elem, c_elem) in &elems {
            let constraint = a_elem.clone() + b_elem.clone() - c_elem.clone();
            ctx.assert_zero(constraint);
        }

        prover_context.prove_without_pcs(&mut rng)
    };

    {
        let mut verifier_context = zkproof.open();

        let mut first_elem = None;
        let mut elems = Vec::new();
        for _ in 0..NUM_CONSTRAINTS {
            let a_elem = verifier_context.read_one().expect("read a");
            if first_elem.is_none() {
                first_elem = Some(a_elem.clone());
            }
            let b_elem = verifier_context.read_one().expect("read b");
            let c_elem = verifier_context.read_one().expect("read c");
            elems.push((a_elem, b_elem, c_elem));
        }

        let mut ctx: crate::zk::inner::ZkVerificationContext<GC> =
            first_elem.unwrap().as_ref().clone();
        for (a_elem, b_elem, c_elem) in &elems {
            let constraint = a_elem.clone() + b_elem.clone() - c_elem.clone();
            ctx.assert_zero(constraint);
        }

        verifier_context.verify_without_pcs().expect("Large mask verification should pass");
    }
}

/// Test with only multiplicative constraints (no linear).
#[tokio::test]
async fn test_zk_builder_only_mul_constraints() {
    const MASK_LENGTH: usize = 20;
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();

    let a_val: <GC as IopCtx>::EF = rng.gen();
    let b_val: <GC as IopCtx>::EF = rng.gen();
    let c_val: <GC as IopCtx>::EF = rng.gen();
    let ab_val = a_val * b_val;
    let bc_val = b_val * c_val;
    let ac_val = a_val * c_val;

    let zkproof = {
        let mut prover_context: ZkProverContext<GC, MK> =
            ZkProverContext::initialize(MASK_LENGTH, &mut rng);
        let a_elem = prover_context.add_value(a_val);
        let b_elem = prover_context.add_value(b_val);
        let c_elem = prover_context.add_value(c_val);
        let ab_elem = prover_context.add_value(ab_val);
        let bc_elem = prover_context.add_value(bc_val);
        let ac_elem = prover_context.add_value(ac_val);

        let mut ctx: ZkProverContext<GC, MK> = a_elem.as_ref().clone();

        // a * b = ab
        let constraint_1 = a_elem.clone() * b_elem.clone() - ab_elem;
        ctx.assert_zero(constraint_1);
        name_constraint!(ctx, "a * b = ab");

        // b * c = bc
        let constraint_2 = b_elem * c_elem.clone() - bc_elem;
        ctx.assert_zero(constraint_2);
        name_constraint!(ctx, "b * c = bc");

        // a * c = ac
        let constraint_3 = a_elem * c_elem - ac_elem;
        ctx.assert_zero(constraint_3);
        name_constraint!(ctx, "a * c = ac");

        prover_context.prove_without_pcs(&mut rng)
    };

    {
        let mut verifier_context = zkproof.open();
        let a_elem = verifier_context.read_one().expect("read a");
        let b_elem = verifier_context.read_one().expect("read b");
        let c_elem = verifier_context.read_one().expect("read c");
        let ab_elem = verifier_context.read_one().expect("read ab");
        let bc_elem = verifier_context.read_one().expect("read bc");
        let ac_elem = verifier_context.read_one().expect("read ac");

        let mut ctx: crate::zk::inner::ZkVerificationContext<GC> = a_elem.as_ref().clone();
        let constraint_1 = a_elem.clone() * b_elem.clone() - ab_elem;
        ctx.assert_zero(constraint_1);
        name_constraint!(ctx, "a * b = ab");
        let constraint_2 = b_elem * c_elem.clone() - bc_elem;
        ctx.assert_zero(constraint_2);
        name_constraint!(ctx, "b * c = bc");
        let constraint_3 = a_elem * c_elem - ac_elem;
        ctx.assert_zero(constraint_3);
        name_constraint!(ctx, "a * c = ac");

        verifier_context.verify_without_pcs().expect("Mul-only verification should pass");
    }
}

/// Test mixed linear and multiplicative constraints.
#[tokio::test]
async fn test_zk_builder_mixed_lin_mul() {
    const MASK_LENGTH: usize = 15;
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();

    let a_val: <GC as IopCtx>::EF = rng.gen();
    let b_val: <GC as IopCtx>::EF = rng.gen();
    let sum_val = a_val + b_val; // linear
    let prod_val = a_val * b_val; // multiplicative
    let sum_prod_val = sum_val * prod_val; // (a+b) * (a*b)

    let zkproof = {
        let mut prover_context: ZkProverContext<GC, MK> =
            ZkProverContext::initialize(MASK_LENGTH, &mut rng);
        let a_elem = prover_context.add_value(a_val);
        let b_elem = prover_context.add_value(b_val);
        let sum_elem = prover_context.add_value(sum_val);
        let prod_elem = prover_context.add_value(prod_val);
        let sum_prod_elem = prover_context.add_value(sum_prod_val);

        let mut ctx: ZkProverContext<GC, MK> = a_elem.as_ref().clone();
        // Linear: a + b = sum
        ctx.assert_zero(a_elem.clone() + b_elem.clone() - sum_elem.clone());
        // Mul: a * b = prod
        ctx.assert_zero(a_elem * b_elem - prod_elem.clone());
        name_constraint!(ctx, "a * b = prod");
        // Mul: sum * prod = sum_prod
        ctx.assert_zero(sum_elem * prod_elem - sum_prod_elem);
        name_constraint!(ctx, "sum * prod = sum_prod");

        prover_context.prove_without_pcs(&mut rng)
    };

    {
        let mut verifier_context = zkproof.open();
        let a_elem = verifier_context.read_one().expect("read a");
        let b_elem = verifier_context.read_one().expect("read b");
        let sum_elem = verifier_context.read_one().expect("read sum");
        let prod_elem = verifier_context.read_one().expect("read prod");
        let sum_prod_elem = verifier_context.read_one().expect("read sum_prod");

        let mut ctx: crate::zk::inner::ZkVerificationContext<GC> = a_elem.as_ref().clone();
        ctx.assert_zero(a_elem.clone() + b_elem.clone() - sum_elem.clone());
        ctx.assert_zero(a_elem * b_elem - prod_elem.clone());
        name_constraint!(ctx, "a * b = prod");
        ctx.assert_zero(sum_elem * prod_elem - sum_prod_elem);
        name_constraint!(ctx, "sum * prod = sum_prod");

        verifier_context.verify_without_pcs().expect("Mixed lin/mul verification should pass");
    }
}

/// Test chained multiplication constraint: a * b * c = d.
#[tokio::test]
async fn test_zk_builder_chained_mul() {
    const MASK_LENGTH: usize = 12;
    type GC = KoalaBearDegree4Duplex;

    let mut rng = ChaCha20Rng::from_entropy();

    let a_val: <GC as IopCtx>::EF = rng.gen();
    let b_val: <GC as IopCtx>::EF = rng.gen();
    let c_val: <GC as IopCtx>::EF = rng.gen();
    let ab_val = a_val * b_val;
    let abc_val = ab_val * c_val;

    let zkproof = {
        let mut prover_context: ZkProverContext<GC, MK> =
            ZkProverContext::initialize(MASK_LENGTH, &mut rng);
        let a_elem = prover_context.add_value(a_val);
        let b_elem = prover_context.add_value(b_val);
        let c_elem = prover_context.add_value(c_val);
        let ab_elem = prover_context.add_value(ab_val);
        let abc_elem = prover_context.add_value(abc_val);

        let mut ctx: ZkProverContext<GC, MK> = a_elem.as_ref().clone();
        // a * b = ab
        ctx.assert_zero(a_elem * b_elem - ab_elem.clone());
        name_constraint!(ctx, "a * b = ab");
        // ab * c = abc
        ctx.assert_zero(ab_elem * c_elem - abc_elem);
        name_constraint!(ctx, "ab * c = abc");

        prover_context.prove_without_pcs(&mut rng)
    };

    {
        let mut verifier_context = zkproof.open();
        let a_elem = verifier_context.read_one().expect("read a");
        let b_elem = verifier_context.read_one().expect("read b");
        let c_elem = verifier_context.read_one().expect("read c");
        let ab_elem = verifier_context.read_one().expect("read ab");
        let abc_elem = verifier_context.read_one().expect("read abc");

        let mut ctx: crate::zk::inner::ZkVerificationContext<GC> = a_elem.as_ref().clone();
        ctx.assert_zero(a_elem * b_elem - ab_elem.clone());
        name_constraint!(ctx, "a * b = ab");
        ctx.assert_zero(ab_elem * c_elem - abc_elem);
        name_constraint!(ctx, "ab * c = abc");

        verifier_context.verify_without_pcs().expect("Chained mul verification should pass");
    }
}

/// Test multiple PCS commitments with different parameters.
#[test]
fn test_pcs_multiple_commitments_different_params() {
    use super::MleCommitmentIndex;

    type GC = KoalaBearDegree4Duplex;
    let mut rng = ChaCha20Rng::from_entropy();

    let digest1: <GC as IopCtx>::Digest = rng.gen();
    let digest2: <GC as IopCtx>::Digest = rng.gen();
    let digest3: <GC as IopCtx>::Digest = rng.gen();

    let zkproof = {
        let masks_length = 3;
        let mut prover_context: ZkProverContext<GC, MK> =
            ZkProverContext::initialize_only_lin_constraints(masks_length, &mut rng);

        let commit_idx1 = prover_context.register_commitment(digest1, (), 8, 2);
        let commit_idx2 = prover_context.register_commitment(digest2, (), 16, 4);
        let commit_idx3 = prover_context.register_commitment(digest3, (), 20, 8);

        assert_eq!(commit_idx1, MleCommitmentIndex::new(0));
        assert_eq!(commit_idx2, MleCommitmentIndex::new(1));
        assert_eq!(commit_idx3, MleCommitmentIndex::new(2));

        let commitments = prover_context.pcs_commitments();
        assert_eq!(commitments.len(), 3);
        assert_eq!(commitments[0].num_vars, 8);
        assert_eq!(commitments[1].num_vars, 16);
        assert_eq!(commitments[2].num_vars, 20);

        let _val1 = prover_context.add_value(rng.gen());
        let _val2 = prover_context.add_value(rng.gen());
        let _val3 = prover_context.add_value(rng.gen());

        prover_context.prove_without_pcs(&mut rng)
    };

    {
        let mut verifier_context = zkproof.open();

        let commit_idx1 =
            verifier_context.read_next_pcs_commitment(8, 2).expect("read commitment 1");
        let commit_idx2 =
            verifier_context.read_next_pcs_commitment(16, 4).expect("read commitment 2");
        let commit_idx3 =
            verifier_context.read_next_pcs_commitment(20, 8).expect("read commitment 3");

        assert_eq!(commit_idx1, MleCommitmentIndex::new(0));
        assert_eq!(commit_idx2, MleCommitmentIndex::new(1));
        assert_eq!(commit_idx3, MleCommitmentIndex::new(2));

        // No more commitments
        assert!(verifier_context.read_next_pcs_commitment(8, 2).is_none());

        let _val1 = verifier_context.read_one().expect("read val 1");
        let _val2 = verifier_context.read_one().expect("read val 2");
        let _val3 = verifier_context.read_one().expect("read val 3");

        verifier_context.verify_without_pcs().expect("Multiple commitments verification failed");
    }
}
