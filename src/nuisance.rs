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

/// Simple plugin estimator (placeholder)
pub struct SimplePluginEstimator;

impl NuisanceEstimator for SimplePluginEstimator {
    fn estimate_fold(
        &self,
        _training_data: &[(Vec<f64>, f64, f64)],
        validation_data: &[(Vec<f64>, f64, f64)],
    ) -> Result<Vec<Vec<f64>>, String> {
        // Return dummy estimates for now
        Ok(validation_data
            .iter()
            .map(|_| vec![0.5, 0.0, 0.0])
            .collect())
    }

    fn name(&self) -> &str {
        "SimplePluginEstimator"
    }
}
