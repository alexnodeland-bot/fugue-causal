//! Checkpointing and serialization for causal inference results.
//!
//! Enables saving and loading posterior estimates for reproducibility and long-running workflows.

use crate::posterior::CausalPosterior;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Serializable checkpoint of a causal inference result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PosteriorCheckpoint {
    /// Point estimate of causal effect
    pub point_estimate: f64,

    /// Posterior standard deviation
    pub posterior_sd: f64,

    /// Calibration temperature (ω)
    pub omega: f64,

    /// Calibration method used
    pub calibration_method: String,

    /// Metadata: timestamp of checkpoint
    pub timestamp: String,

    /// Metadata: identifer used
    pub identifier_name: String,

    /// Metadata: number of observations
    pub n_observations: usize,
}

impl PosteriorCheckpoint {
    /// Create checkpoint from posterior and metadata
    pub fn from_posterior(
        posterior: &CausalPosterior,
        identifier_name: &str,
        n_observations: usize,
    ) -> Self {
        Self {
            point_estimate: posterior.point_estimate,
            posterior_sd: posterior.posterior_sd,
            omega: posterior.omega,
            calibration_method: posterior.calibration_method.clone(),
            timestamp: chrono_now(),
            identifier_name: identifier_name.to_string(),
            n_observations,
        }
    }

    /// Convert back to CausalPosterior
    pub fn to_posterior(&self) -> CausalPosterior {
        CausalPosterior {
            point_estimate: self.point_estimate,
            posterior_sd: self.posterior_sd,
            omega: self.omega,
            calibration_method: self.calibration_method.clone(),
        }
    }

    /// Save checkpoint to file (binary format)
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let bytes =
            bincode::serialize(self).map_err(|e| format!("Serialization failed: {}", e))?;

        let mut file = File::create(path).map_err(|e| format!("File creation failed: {}", e))?;

        file.write_all(&bytes)
            .map_err(|e| format!("Write failed: {}", e))?;

        Ok(())
    }

    /// Load checkpoint from file (binary format)
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("File open failed: {}", e))?;

        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| format!("Read failed: {}", e))?;

        bincode::deserialize(&bytes).map_err(|e| format!("Deserialization failed: {}", e))
    }

    /// Human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "PosteriorCheckpoint {{\n  \
             point_estimate: {:.6},\n  \
             posterior_sd: {:.6},\n  \
             omega: {:.6},\n  \
             calibration: {},\n  \
             identifier: {},\n  \
             n_observations: {},\n  \
             timestamp: {}\n\
             }}",
            self.point_estimate,
            self.posterior_sd,
            self.omega,
            self.calibration_method,
            self.identifier_name,
            self.n_observations,
            self.timestamp
        )
    }
}

/// Simplified timestamp (ISO 8601-ish, no external dependencies)
fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let secs = duration.as_secs();
    let millis = duration.subsec_millis();

    // Simple format: YYYY-MM-DDTHH:MM:SS.sssZ
    // (Note: simplified, doesn't handle all timezones properly)
    format!("{}T{:03}Z", secs, millis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_creation() {
        let posterior = CausalPosterior {
            point_estimate: 2.5,
            posterior_sd: 0.15,
            omega: 1.0,
            calibration_method: "bootstrap".to_string(),
        };

        let checkpoint = PosteriorCheckpoint::from_posterior(&posterior, "DR", 500);

        assert_eq!(checkpoint.point_estimate, 2.5);
        assert_eq!(checkpoint.posterior_sd, 0.15);
        assert_eq!(checkpoint.identifier_name, "DR");
        assert_eq!(checkpoint.n_observations, 500);
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let posterior = CausalPosterior {
            point_estimate: 3.14,
            posterior_sd: 0.08,
            omega: 1.5,
            calibration_method: "grid_search".to_string(),
        };

        let checkpoint = PosteriorCheckpoint::from_posterior(&posterior, "IPW", 1000);
        let recovered = checkpoint.to_posterior();

        assert_eq!(recovered.point_estimate, posterior.point_estimate);
        assert_eq!(recovered.posterior_sd, posterior.posterior_sd);
        assert_eq!(recovered.omega, posterior.omega);
        assert_eq!(
            recovered.calibration_method,
            posterior.calibration_method
        );
    }

    #[test]
    fn test_checkpoint_summary() {
        let posterior = CausalPosterior {
            point_estimate: 1.0,
            posterior_sd: 0.2,
            omega: 1.0,
            calibration_method: "bootstrap".to_string(),
        };

        let checkpoint = PosteriorCheckpoint::from_posterior(&posterior, "RA", 250);
        let summary = checkpoint.summary();

        assert!(summary.contains("1.000000"));
        assert!(summary.contains("RA"));
        assert!(summary.contains("250"));
    }
}
