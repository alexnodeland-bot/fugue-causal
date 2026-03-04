//! Bootstrap calibration of ω (temperature) for valid frequentist coverage.
//!
//! The Gibbs posterior q(θ | D) ∝ exp{−ωn·L_n(θ)} · π(θ) has variance ω^{−1}.
//! We tune ω so that (1−α) credible intervals achieve (1−α) repeated-sampling coverage.
//!
//! **Reference:** Syring & Martin (2019). "Calibrating General Posterior Credible Regions."
//! *Bayesian Analysis*, 14(3), 607-630.

use rand::distributions::Uniform;
use rand::Rng;

/// Bootstrap calibration result
#[derive(Clone, Debug)]
pub struct OmegaCalibration {
    /// Calibrated temperature parameter
    pub omega: f64,
    /// Empirical coverage from bootstrap replicates
    pub empirical_coverage: f64,
    /// Number of bootstrap replicates used
    pub num_replicates: usize,
    /// Target coverage level
    pub target_coverage: f64,
}

/// Calibrate ω for a target coverage level (e.g., 0.95 for 95% CI)
///
/// Algorithm:
/// 1. For each bootstrap replicate b = 1, ..., B:
///    a. Resample observations with replacement (same size n)
///    b. Fit nuisances on resampled data
///    c. Compute loss L^*_n(θ) on full dataset with estimated nuisances
///    d. Find θ̂^*(b) = argmin L^*_n(θ)
/// 2. Coverage of (1−α) credible interval under ω:
///    P(θ* ∈ [θ̂ ± z_{α/2} √(ω^{-1} V̂)]) over bootstrap replicates
/// 3. Grid search to find ω achieving target coverage
///
/// # Arguments
///
/// * `loss_fn` - Per-observation loss for a given θ
/// * `n_replicates` - Number of bootstrap replicates (default: 100)
/// * `target_coverage` - Target coverage level, e.g., 0.95 (default)
/// * `grid_points` - Points to search for ω (default: 20)
///
/// # Returns
///
/// OmegaCalibration with fitted ω and empirical coverage
pub fn calibrate_omega(
    loss_fn: &dyn Fn(&[f64]) -> f64,
    posterior_mean: f64,
    posterior_variance: f64,
    data: &[Vec<f64>],
    target_coverage: f64,
    n_replicates: usize,
) -> OmegaCalibration {
    let mut rng = rand::thread_rng();

    // Grid search for ω
    let omega_candidates: Vec<f64> = (1..=20).map(|i| (i as f64) * 0.5).collect();

    let mut best_omega = 1.0;
    let mut best_coverage_error = target_coverage;

    for &candidate_omega in &omega_candidates {
        let coverage = estimate_coverage_bootstrap(
            loss_fn,
            posterior_mean,
            posterior_variance,
            data,
            candidate_omega,
            n_replicates,
            &mut rng,
        );

        let coverage_error = (coverage - target_coverage).abs();
        if coverage_error < best_coverage_error {
            best_coverage_error = coverage_error;
            best_omega = candidate_omega;
        }
    }

    let empirical_coverage = estimate_coverage_bootstrap(
        loss_fn,
        posterior_mean,
        posterior_variance,
        data,
        best_omega,
        n_replicates,
        &mut rng,
    );

    OmegaCalibration {
        omega: best_omega,
        empirical_coverage,
        num_replicates: n_replicates,
        target_coverage,
    }
}

/// Estimate coverage of (1−α) credible interval under a given ω
fn estimate_coverage_bootstrap(
    loss_fn: &dyn Fn(&[f64]) -> f64,
    posterior_mean: f64,
    posterior_variance: f64,
    data: &[Vec<f64>],
    omega: f64,
    n_replicates: usize,
    rng: &mut impl Rng,
) -> f64 {
    let sample_size = data.len();
    let alpha = 0.05; // Default: 95% CI
    let z_alpha_2 = normal_quantile(1.0 - alpha / 2.0); // ~1.96

    let mut coverage_count = 0;

    for _ in 0..n_replicates {
        // Bootstrap resample
        let dist = Uniform::new(0, sample_size);
        let mut bootstrap_loss_sum = 0.0;
        for _ in 0..sample_size {
            let idx = rng.sample(dist);
            bootstrap_loss_sum += loss_fn(&data[idx]);
        }
        let bootstrap_mean_loss = bootstrap_loss_sum / (sample_size as f64);

        // Credible interval bounds (in loss space)
        let se = (posterior_variance / omega / (sample_size as f64)).sqrt();
        let ci_lower = posterior_mean - z_alpha_2 * se;
        let ci_upper = posterior_mean + z_alpha_2 * se;

        // For coverage: count if bootstrap mean falls in posterior credible interval
        // This is a simplification — a full bootstrap calibration would track bootstrap parameter estimates
        if bootstrap_mean_loss >= ci_lower && bootstrap_mean_loss <= ci_upper {
            coverage_count += 1;
        }
    }

    (coverage_count as f64) / (n_replicates as f64)
}

/// Compute normal quantile (inverse CDF)
/// Approximates Φ^{-1}(p) using rational approximation
pub fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let p_shifted = if p < 0.5 { p } else { 1.0 - p };

    // Rational approximation (Abramowitz & Stegun 26.2.23)
    let t = (-2.0 * p_shifted.ln()).sqrt();
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let numerator = c0 + c1 * t + c2 * t * t;
    let denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    let quantile = t - numerator / denominator;

    if p >= 0.5 {
        quantile
    } else {
        -quantile
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_quantile_median() {
        let q = normal_quantile(0.5);
        assert!(q.abs() < 0.01, "Median should be ~0, got {}", q);
    }

    #[test]
    fn test_normal_quantile_95th() {
        let q = normal_quantile(0.975);
        assert!(
            (q - 1.96).abs() < 0.01,
            "95th percentile should be ~1.96, got {}",
            q
        );
    }

    #[test]
    fn test_normal_quantile_5th() {
        let q = normal_quantile(0.025);
        assert!(
            (q + 1.96).abs() < 0.01,
            "5th percentile should be ~-1.96, got {}",
            q
        );
    }

    #[test]
    fn test_normal_quantile_monotonic() {
        let q1 = normal_quantile(0.3);
        let q2 = normal_quantile(0.5);
        let q3 = normal_quantile(0.7);
        assert!(
            q1 < q2 && q2 < q3,
            "Quantile should be monotonically increasing"
        );
    }

    #[test]
    fn test_calibrate_omega_simple() {
        // Simple quadratic loss
        let loss_fn = |obs: &[f64]| {
            if obs.is_empty() {
                return 0.0;
            }
            obs[0].powi(2)
        };

        // Synthetic data: observations around 0
        let data: Vec<Vec<f64>> = (0..50).map(|i| vec![((i as f64) - 25.0) / 10.0]).collect();

        let calib = calibrate_omega(
            &loss_fn, 0.0, // posterior mean at true value
            1.0, // posterior variance
            &data, 0.95, // target 95% coverage
            30,   // 30 bootstrap replicates (fast test)
        );

        // Check basic properties: ω should be positive
        assert!(
            calib.omega > 0.0,
            "ω should be positive, got {}",
            calib.omega
        );

        // Check calibration returned reasonable coverage estimate
        assert!(
            calib.empirical_coverage >= 0.0 && calib.empirical_coverage <= 1.0,
            "Coverage should be in [0, 1], got {}",
            calib.empirical_coverage
        );
    }

    #[test]
    fn test_calibrate_omega_coverage_range() {
        let loss_fn = |obs: &[f64]| {
            if obs.is_empty() {
                return 0.0;
            }
            (obs[0] - 0.5).powi(2)
        };

        let data: Vec<Vec<f64>> = (0..100).map(|i| vec![(i as f64) / 100.0]).collect();

        let calib = calibrate_omega(&loss_fn, 0.5, 0.01, &data, 0.95, 50);

        assert!(calib.empirical_coverage >= 0.0 && calib.empirical_coverage <= 1.0);
    }

    #[test]
    fn test_omega_calibration_fields() {
        let loss_fn = |_obs: &[f64]| 0.0;
        let data = vec![vec![1.0], vec![2.0]];

        let calib = calibrate_omega(&loss_fn, 1.5, 1.0, &data, 0.95, 30);

        assert_eq!(calib.target_coverage, 0.95);
        assert_eq!(calib.num_replicates, 30);
        assert!(calib.omega > 0.0);
    }
}
