//! Integration with fugue probabilistic programming library.
//!
//! This module enables use of fugue traces as causal priors and conditions
//! inference via effect handlers that apply causal loss functions.

use crate::identifier::CausalIdentifier;
use crate::nuisance::NuisanceEstimator;
use crate::posterior::CausalPosterior;

/// Causal effect handler for fugue traces.
///
/// This struct wraps a causal identifier and loss function to be used
/// as an effect handler in fugue probabilistic programs. It allows
/// automatic causal conditioning of probabilistic traces.
pub struct CausalEffectHandler<T: CausalIdentifier> {
    /// The causal identifier (RA, IPW, DR, R-learner)
    identifier: T,
}

impl<T: CausalIdentifier> CausalEffectHandler<T> {
    /// Create a new causal effect handler with the given identifier.
    pub fn new(identifier: T) -> Self {
        CausalEffectHandler { identifier }
    }

    /// Apply causal loss to a trace observation.
    ///
    /// This is the core integration point: given an observation from
    /// a fugue trace, compute the causal loss under this identifier.
    pub fn loss_from_trace(
        &self,
        observation: &(Vec<f64>, f64, f64),
        nuisances: &[f64],
        theta: f64,
    ) -> f64 {
        // Flatten observation to standard format: [X_1, ..., X_p, A, Y]
        let mut obs_vec = observation.0.clone();
        obs_vec.push(observation.1);  // treatment A
        obs_vec.push(observation.2);  // outcome Y
        self.identifier.loss(&obs_vec, theta, nuisances)
    }
}

/// Helper trait for converting fugue trace outputs to causal observations.
///
/// Implement this trait for your trace output type to enable automatic
/// conversion to (covariate, treatment, outcome) tuples.
pub trait TraceObservation {
    /// Extract covariate vector from trace output.
    fn extract_covariates(&self) -> Vec<f64>;

    /// Extract binary treatment indicator.
    fn extract_treatment(&self) -> f64;

    /// Extract continuous outcome.
    fn extract_outcome(&self) -> f64;

    /// Convert to standard observation tuple.
    fn as_observation(&self) -> (Vec<f64>, f64, f64) {
        (
            self.extract_covariates(),
            self.extract_treatment(),
            self.extract_outcome(),
        )
    }
}

/// Batch convert fugue traces to causal observations.
///
/// Takes a vector of trace outputs and converts them to the standard
/// (covariate, treatment, outcome) format for causal inference.
pub fn traces_to_observations<T: TraceObservation>(
    traces: Vec<T>,
) -> Vec<(Vec<f64>, f64, f64)> {
    traces.into_iter().map(|t| t.as_observation()).collect()
}

/// Run causal inference directly on fugue traces.
///
/// This function bridges the gap: it takes raw trace outputs, converts
/// them to observations, and runs the full causal inference pipeline.
///
/// # Arguments
///
/// * `traces` - Vector of traced outputs, each convertible to observations
/// * `identifier` - Causal identifier (RA, IPW, DR, R-learner)
/// * `nuisance_estimator` - Nuisance component estimator
/// * `folds` - K-fold cross-fitting parameter
///
/// # Returns
///
/// Posterior distribution over the causal effect.
///
/// # Example
///
/// ```ignore
/// use fugue_causal::{fugue_integration, prior_ate, DoublyRobust, PluginEstimator};
///
/// // After running fugue probabilistic program:
/// let traces: Vec<MyTraceOutput> = run_program();
///
/// // Convert and infer directly
/// let posterior = fugue_integration::infer_from_traces(
///     traces,
///     DoublyRobust,
///     Box::new(PluginEstimator),
///     5,
/// )?;
/// ```
pub fn infer_from_traces<T: TraceObservation>(
    traces: Vec<T>,
    identifier: impl CausalIdentifier,
    nuisance_estimator: Box<dyn NuisanceEstimator>,
    folds: usize,
) -> Result<CausalPosterior, String> {
    use crate::{infer_causal, prior_ate};

    let observations = traces_to_observations(traces);
    infer_causal(prior_ate(), identifier, nuisance_estimator, folds, &observations)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyTrace {
        covariates: Vec<f64>,
        treatment: f64,
        outcome: f64,
    }

    impl TraceObservation for DummyTrace {
        fn extract_covariates(&self) -> Vec<f64> {
            self.covariates.clone()
        }

        fn extract_treatment(&self) -> f64 {
            self.treatment
        }

        fn extract_outcome(&self) -> f64 {
            self.outcome
        }
    }

    #[test]
    fn test_trace_to_observation_conversion() {
        let trace = DummyTrace {
            covariates: vec![1.0, 2.0],
            treatment: 1.0,
            outcome: 5.5,
        };

        let obs = trace.as_observation();
        assert_eq!(obs.0, vec![1.0, 2.0]);
        assert_eq!(obs.1, 1.0);
        assert_eq!(obs.2, 5.5);
    }

    #[test]
    fn test_batch_convert_traces() {
        let traces = vec![
            DummyTrace {
                covariates: vec![0.5],
                treatment: 1.0,
                outcome: 2.0,
            },
            DummyTrace {
                covariates: vec![0.3],
                treatment: 0.0,
                outcome: 1.0,
            },
        ];

        let observations = traces_to_observations(traces);
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].0, vec![0.5]);
        assert_eq!(observations[1].1, 0.0);
    }
}
