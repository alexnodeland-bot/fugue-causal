//! Integration tests for fugue-causal end-to-end inference

use fugue_causal::*;

/// Generate synthetic data with known true ATE
///
/// Data generation:
/// X ~ N(0, 1) (covariate)
/// A ~ Bernoulli(0.5) (treatment)
/// Y = X + true_ate * A + N(0, 1) (outcome)
///
/// True causal effect: E[Y(1) - Y(0)] = true_ate
fn synthetic_ate_data(n: usize, true_ate: f64, seed: u64) -> Vec<(Vec<f64>, f64, f64)> {
    use rand::SeedableRng;
    use rand_distr::{Bernoulli, Distribution, Normal};

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal_x = Normal::new(0.0, 1.0).unwrap();
    let normal_y = Normal::new(0.0, 1.0).unwrap();
    let bernoulli_a = Bernoulli::new(0.5).unwrap();

    (0..n)
        .map(|_| {
            let x = normal_x.sample(&mut rng);
            let a = if bernoulli_a.sample(&mut rng) {
                1.0
            } else {
                0.0
            };
            let y = x + true_ate * a + normal_y.sample(&mut rng);
            (vec![x], a, y)
        })
        .collect()
}

/// Simple plugin nuisance estimator (returns conditional means)
struct PluginEstimator;

impl NuisanceEstimator for PluginEstimator {
    fn estimate_fold(
        &self,
        training_data: &[(Vec<f64>, f64, f64)],
        validation_data: &[(Vec<f64>, f64, f64)],
    ) -> Result<Vec<Vec<f64>>, String> {
        // Compute sample propensity (P(A=1))
        let p_treated = training_data.iter().filter(|(_, a, _)| *a > 0.5).count() as f64
            / training_data.len() as f64;

        // Compute conditional outcome means
        let (sum_y_treated, sum_y_control) =
            training_data
                .iter()
                .fold((0.0, 0.0), |(st, sc), (_, a, y)| {
                    if *a > 0.5 {
                        (st + y, sc)
                    } else {
                        (st, sc + y)
                    }
                });

        let count_treated = training_data.iter().filter(|(_, a, _)| *a > 0.5).count() as f64;
        let count_control = training_data.iter().filter(|(_, a, _)| *a < 0.5).count() as f64;

        let m_1 = if count_treated > 0.0 {
            sum_y_treated / count_treated
        } else {
            0.0
        };
        let m_0 = if count_control > 0.0 {
            sum_y_control / count_control
        } else {
            0.0
        };

        // Return [propensity, m_1, m_0] for each validation observation
        Ok(validation_data
            .iter()
            .map(|_| vec![p_treated, m_1, m_0])
            .collect())
    }

    fn name(&self) -> &str {
        "PluginEstimator"
    }
}

#[test]
fn test_ate_inference_on_synthetic_data() {
    let true_ate = 1.5;
    let n = 200;
    let data = synthetic_ate_data(n, true_ate, 42);

    let result = infer_causal(
        prior_ate(),
        identifier::DoublyRobust,
        Box::new(PluginEstimator),
        5,
        &data,
    );

    assert!(result.is_ok());

    let posterior = result.unwrap();

    // Point estimate should be close to true ATE
    eprintln!("True ATE: {}", true_ate);
    eprintln!("Posterior mean: {}", posterior.point_estimate);
    eprintln!("Posterior SD: {}", posterior.posterior_sd);

    // Allow some tolerance due to finite sample variation
    let error = (posterior.point_estimate - true_ate).abs();
    assert!(
        error < 0.5,
        "Point estimate {} too far from true ATE {}",
        posterior.point_estimate,
        true_ate
    );

    // Posterior SD should be positive
    assert!(posterior.posterior_sd > 0.0);

    // Credible interval should contain true value
    let (lower, upper) = posterior.credible_interval(0.95);
    eprintln!("95% credible interval: [{}, {}]", lower, upper);

    // This is a rough check; in larger samples it should almost always contain
    // the truth
    assert!(
        lower < true_ate && true_ate < upper,
        "Credible interval [{}, {}] should contain true ATE {}",
        lower,
        upper,
        true_ate
    );
}

#[test]
fn test_ate_inference_large_sample() {
    let true_ate = 2.0;
    let n = 500;
    let data = synthetic_ate_data(n, true_ate, 123);

    let result = infer_causal(
        prior_ate(),
        identifier::DoublyRobust,
        Box::new(PluginEstimator),
        5,
        &data,
    );

    assert!(result.is_ok());
    let posterior = result.unwrap();

    eprintln!(
        "Large sample (n={}): point est={}, truth={}",
        n, posterior.point_estimate, true_ate
    );

    // With larger sample, should be even closer to truth
    let error = (posterior.point_estimate - true_ate).abs();
    assert!(
        error < 0.3,
        "Large sample: point estimate {} far from true ATE {}",
        posterior.point_estimate,
        true_ate
    );
}

#[test]
fn test_ate_inference_small_effect() {
    let true_ate = 0.2; // Small effect
    let n = 300;
    let data = synthetic_ate_data(n, true_ate, 456);

    let result = infer_causal(
        prior_ate(),
        identifier::DoublyRobust,
        Box::new(PluginEstimator),
        5,
        &data,
    );

    assert!(result.is_ok());
    let posterior = result.unwrap();

    eprintln!(
        "Small effect: true={}, posterior mean={}",
        true_ate, posterior.point_estimate
    );

    // Even small effects should be detectable
    let error = (posterior.point_estimate - true_ate).abs();
    assert!(error < 0.4);
}

#[test]
fn test_different_identifiers() {
    let true_ate = 1.0;
    let n = 200;
    let data = synthetic_ate_data(n, true_ate, 789);

    // Test DR (Doubly Robust)
    let result_dr = infer_causal(
        prior_ate(),
        identifier::DoublyRobust,
        Box::new(PluginEstimator),
        5,
        &data,
    );
    assert!(result_dr.is_ok());
    eprintln!("DR estimate: {}", result_dr.unwrap().point_estimate);

    // Test RA (Regression Adjustment)
    let result_ra = infer_causal(
        prior_ate(),
        identifier::RegressionAdjustment,
        Box::new(PluginEstimator),
        5,
        &data,
    );
    assert!(result_ra.is_ok());
    eprintln!("RA estimate: {}", result_ra.unwrap().point_estimate);

    // Test IPW
    let result_ipw = infer_causal(
        prior_ate(),
        identifier::InverseProbabilityWeighting,
        Box::new(PluginEstimator),
        5,
        &data,
    );
    assert!(result_ipw.is_ok());
    eprintln!("IPW estimate: {}", result_ipw.unwrap().point_estimate);
}

#[test]
fn test_ate_with_strong_effect() {
    let true_ate = 3.0;
    let n = 150;
    let data = synthetic_ate_data(n, true_ate, 111);

    let result = infer_causal(
        prior_ate(),
        identifier::DoublyRobust,
        Box::new(PluginEstimator),
        5,
        &data,
    );

    assert!(result.is_ok());
    let posterior = result.unwrap();

    eprintln!(
        "Strong effect: true={}, posterior mean={}",
        true_ate, posterior.point_estimate
    );

    // With strong effect and decent sample size, error should be small
    let error = (posterior.point_estimate - true_ate).abs();
    assert!(error < 0.5);
}
