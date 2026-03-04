//! Nuisance estimation strategies.
//!
//! Nuisances are auxiliary parameters (propensity scores, outcome regressions)
//! required by causal identifiers.

/// Trait for estimating nuisance components
pub trait NuisanceEstimator: Send + Sync {
    /// Estimate nuisances from cross-fitting fold
    ///
    /// # Arguments
    ///
    /// * `training_data` - (X, A, Y) tuples for training
    /// * `validation_data` - (X, A, Y) tuples for validation
    ///
    /// # Returns
    ///
    /// Vec of nuisance estimates on validation data
    fn estimate_fold(
        &self,
        training_data: &[(Vec<f64>, f64, f64)],
        validation_data: &[(Vec<f64>, f64, f64)],
    ) -> Result<Vec<Vec<f64>>, String>;

    fn name(&self) -> &str;
}

/// Simple plugin estimator for linear confounders
///
/// Computes propensity scores and outcome models via simple sample statistics.
/// Suitable for demonstration and when confounders are well-balanced.
pub struct PluginEstimator;

impl NuisanceEstimator for PluginEstimator {
    fn estimate_fold(
        &self,
        training_data: &[(Vec<f64>, f64, f64)],
        validation_data: &[(Vec<f64>, f64, f64)],
    ) -> Result<Vec<Vec<f64>>, String> {
        // Compute sample propensity from training data
        let p_treated = training_data
            .iter()
            .filter(|(_, a, _)| *a > 0.5)
            .count() as f64
            / training_data.len() as f64;

        // Bound propensity away from 0 and 1
        let propensity = (p_treated).max(0.01).min(0.99);

        // Compute conditional outcome means
        let (sum_y_treated, count_treated, sum_y_control, count_control) = training_data
            .iter()
            .fold((0.0, 0, 0.0, 0), |(st, ct, sc, cc), (_, a, y)| {
                if *a > 0.5 {
                    (st + y, ct + 1, sc, cc)
                } else {
                    (st, ct, sc + y, cc + 1)
                }
            });

        let m_1 = if count_treated > 0 {
            sum_y_treated / count_treated as f64
        } else {
            0.0
        };

        let m_0 = if count_control > 0 {
            sum_y_control / count_control as f64
        } else {
            0.0
        };

        // Return estimates for validation data
        // Format: [propensity, m_1, m_0] for DR/AIPW or similar identifiers
        Ok(validation_data
            .iter()
            .map(|_| vec![propensity, m_1, m_0])
            .collect())
    }

    fn name(&self) -> &str {
        "PluginEstimator"
    }
}
