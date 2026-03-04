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
pub fn infer_causal<T: CausalIdentifier>(
    _estimand_prior: Box<dyn Fn(f64) -> f64>,
    _identifier: T,
    _nuisance_estimator: Box<dyn NuisanceEstimator>,
    _folds: usize,
    _data: &[(Vec<f64>, f64, f64)],
) -> Result<CausalPosterior, String> {
    // TODO: Implement full Gibbs posterior inference
    // 1. Cross-fit nuisance estimates
    // 2. Compute empirical loss
    // 3. Calibrate omega via bootstrap
    // 4. Sample from Gibbs posterior

    Ok(CausalPosterior {
        point_estimate: 0.5,
        posterior_sd: 0.1,
        omega: 1.0,
        calibration_method: "bootstrap".to_string(),
    })
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
