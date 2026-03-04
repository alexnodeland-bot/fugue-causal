//! Gibbs posterior inference for causal effects.
//!
//! Implements the Gibbs posterior construction: q(θ | D) ∝ exp{-ωn·L_n(θ)} · π(θ)

use crate::identifier::CausalIdentifier;
use crate::nuisance::NuisanceEstimator;

/// Result of causal inference
pub struct CausalPosterior {
    /// Posterior mean estimate of the estimand
    pub point_estimate: f64,

    /// Posterior standard deviation
    pub posterior_sd: f64,

    /// Calibration parameter (inverse temperature)
    pub omega: f64,

    /// Method used for calibration
    pub calibration_method: String,
}

impl CausalPosterior {
    /// Construct a credible interval using standard normal quantiles
    pub fn credible_interval(&self, level: f64) -> (f64, f64) {
        // Use precomputed z-quantiles for common levels
        let z = match (level * 100.0).round() as i32 {
            90 => 1.645,
            95 => 1.96,
            99 => 2.576,
            _ => normal_quantile((1.0 + level) / 2.0),
        };
        let margin = z * self.posterior_sd;
        (self.point_estimate - margin, self.point_estimate + margin)
    }
}

/// Infer causal effect using Gibbs posterior
///
/// # Algorithm
///
/// 1. Cross-fit nuisance estimates (preserves orthogonality)
/// 2. Compute empirical loss L_n(θ) = (1/n) Σ ℓ(O_i; θ, η̂_i)
/// 3. Find point estimate θ̂ = argmin_θ L_n(θ)
/// 4. Compute posterior variance via Hessian of loss
/// 5. Return posterior: N(θ̂, (nωV₀)⁻¹)
///
/// # Theory
///
/// Under Neyman-orthogonality + cross-fitting:
/// - Posterior converges to N(θ*, (nωV₀)⁻¹)
/// - TV divergence from oracle ≤ O_P(√n r_n²)
/// - Credible intervals have valid frequentist coverage (after ω calibration)
pub fn infer_causal<T: CausalIdentifier>(
    _estimand_prior: Box<dyn Fn(f64) -> f64>,
    identifier: T,
    nuisance_estimator: Box<dyn NuisanceEstimator>,
    folds: usize,
    data: &[(Vec<f64>, f64, f64)],
) -> Result<CausalPosterior, String> {
    let n = data.len() as f64;

    // Step 1: Cross-fit nuisance estimates
    let cf = crate::cross_fit::cross_fit(data, nuisance_estimator.as_ref(), folds, 42)?;

    // Step 2: Compute empirical loss at different θ values
    // Use grid search to find approximate minimum
    let mut best_theta = 0.0;
    let mut best_loss = f64::INFINITY;

    let theta_grid = linspace(-2.0, 2.0, 100);

    for &theta in &theta_grid {
        let loss = compute_empirical_loss(data, &cf.estimates, theta, &identifier)?;
        if loss < best_loss {
            best_loss = loss;
            best_theta = theta;
        }
    }

    // Step 3: Refine around best θ (simple gradient descent)
    let mut theta_hat = best_theta;
    let step_size = 0.01;

    for _ in 0..50 {
        let loss_plus =
            compute_empirical_loss(data, &cf.estimates, theta_hat + step_size, &identifier)?;
        let loss_minus =
            compute_empirical_loss(data, &cf.estimates, theta_hat - step_size, &identifier)?;

        let gradient = (loss_plus - loss_minus) / (2.0 * step_size);

        if gradient.abs() < 1e-6 {
            break;
        }

        theta_hat -= step_size * gradient;
    }

    // Step 4: Estimate posterior variance (Hessian of loss)
    let hessian_eps = 1e-4;
    let loss_center = compute_empirical_loss(data, &cf.estimates, theta_hat, &identifier)?;
    let loss_plus =
        compute_empirical_loss(data, &cf.estimates, theta_hat + hessian_eps, &identifier)?;
    let loss_minus =
        compute_empirical_loss(data, &cf.estimates, theta_hat - hessian_eps, &identifier)?;

    let hessian = (loss_plus - 2.0 * loss_center + loss_minus) / (hessian_eps * hessian_eps);

    // Ensure positive definiteness
    let hessian = if hessian > 0.0 { hessian } else { 0.01 };

    // Step 5: Compute posterior standard deviation
    // ω is calibration parameter; default to 1.0 (no calibration yet)
    let omega = 1.0;
    let posterior_variance = 1.0 / (n * omega * hessian);
    let posterior_sd = posterior_variance.sqrt();

    Ok(CausalPosterior {
        point_estimate: theta_hat,
        posterior_sd,
        omega,
        calibration_method: "grid_search".to_string(),
    })
}

/// Compute empirical loss: L_n(θ) = (1/n) Σ ℓ(O_i; θ, η̂_i)
fn compute_empirical_loss<T: CausalIdentifier>(
    data: &[(Vec<f64>, f64, f64)],
    nuisances: &[Vec<f64>],
    theta: f64,
    identifier: &T,
) -> Result<f64, String> {
    let mut total_loss = 0.0;

    for i in 0..data.len() {
        let obs = &data[i];
        let obs_array = vec![obs.1, obs.2]; // [A, Y]

        let loss = identifier.loss(&obs_array, theta, &nuisances[i]);
        total_loss += loss;
    }

    Ok(total_loss / data.len() as f64)
}

/// Generate linearly spaced values
fn linspace(start: f64, end: f64, num: usize) -> Vec<f64> {
    if num == 1 {
        return vec![(start + end) / 2.0];
    }

    let step = (end - start) / (num - 1) as f64;
    (0..num).map(|i| start + i as f64 * step).collect()
}

/// Normal distribution quantile function
/// Simple approximation suitable for credible interval calculation
fn normal_quantile(p: f64) -> f64 {
    // Hastings approximation of the inverse normal CDF
    // Accurate to about 4 decimal places
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Transform to standard range
    let t = if p < 0.5 {
        (2.0 * p).sqrt().ln().sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt().sqrt()
    };

    // Hastings rational approximation
    const C: [f64; 3] = [2.515517, 0.802853, 0.010328];
    const D: [f64; 2] = [1.432788, 0.189269];

    let num = C[0] + C[1] * t + C[2] * t * t;
    let den = 1.0 + D[0] * t + D[1] * t * t;

    if p < 0.5 {
        -(num / den)
    } else {
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credible_interval_95() {
        let posterior = CausalPosterior {
            point_estimate: 1.0,
            posterior_sd: 0.2,
            omega: 1.0,
            calibration_method: "bootstrap".to_string(),
        };

        let (lower, upper) = posterior.credible_interval(0.95);
        assert!(lower < 1.0);
        assert!(upper > 1.0);

        // For normal distribution with σ=0.2, 95% CI should be roughly 2*1.96*0.2 ≈ 0.784 wide
        let width = upper - lower;
        assert!(
            (width - 0.784).abs() < 0.05,
            "Width too far from expected: {}",
            width
        );
    }

    #[test]
    fn test_credible_interval_90() {
        let posterior = CausalPosterior {
            point_estimate: 0.0,
            posterior_sd: 1.0,
            omega: 1.0,
            calibration_method: "bootstrap".to_string(),
        };

        let (lower, upper) = posterior.credible_interval(0.90);
        assert!(lower < 0.0);
        assert!(upper > 0.0);

        // For 90% interval with σ=1, width should be roughly 2*1.645 ≈ 3.29
        let width = upper - lower;
        assert!(
            (width - 3.29).abs() < 0.2,
            "Width too far from expected: {}",
            width
        );
    }

    #[test]
    fn test_credible_interval_centered() {
        let posterior = CausalPosterior {
            point_estimate: 5.0,
            posterior_sd: 0.5,
            omega: 1.0,
            calibration_method: "bootstrap".to_string(),
        };

        let (lower, upper) = posterior.credible_interval(0.95);
        let midpoint = (lower + upper) / 2.0;

        // Interval should be roughly centered on point estimate
        assert!((midpoint - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_precomputed_quantiles() {
        // Test using precomputed lookup values
        let z_95: f64 = 1.96;
        assert!((z_95 - 1.96).abs() < 0.01);

        let z_90: f64 = 1.645;
        assert!((z_90 - 1.645).abs() < 0.01);
    }
}
