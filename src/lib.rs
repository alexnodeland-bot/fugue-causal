//! # fugue-causal: Bayesian Causal Inference via Generalized Bayes
//!
//! Extends the fugue probabilistic programming library with loss-based causal inference
//! using generalized (Gibbs) posteriors.
//!
//! **Source Paper:** Javurek, E., et al. (2026). "Generalized Bayes for Causal Inference."
//! ArXiv:2603.03035v1. <https://arxiv.org/abs/2603.03035>
//!
//! ## Core Idea
//!
//! Instead of specifying full data-generating models P(X, A, Y | ξ) and placing priors on
//! nuisance components, we:
//!
//! 1. Place priors directly on causal estimands θ (e.g., ATE, CATE)
//! 2. Update via identification-driven loss functions (not likelihoods)
//! 3. Use Gibbs posteriors: q(θ | D) ∝ exp{−ωn·L_n(θ)} · π(θ)
//!
//! **Key advantage:** Formal robustness. Neyman-orthogonal losses (like DR/AIPW) give
//! second-order nuisance robustness: TV divergence = O_P(√n · r_n²), not O_P(√n · r_n).
//!
//! ## Example
//!
//! ```ignore
//! use fugue_causal::*;
//!
//! let problem = CausalProblem {
//!     estimand_prior: prior_ate(),
//!     identifier: DoublyRobust,
//!     nuisance_estimator: Box::new(CausalForest::new()),
//!     folds: 5,
//! };
//!
//! let posterior = infer_causal(problem, observational_data)?;
//! ```

pub mod bootstrap;
pub mod cross_fit;
pub mod estimand;
pub mod identifier;
pub mod nuisance;
pub mod posterior;

pub use bootstrap::{calibrate_omega, OmegaCalibration};
pub use cross_fit::{cross_fit, CrossFittedNuisances};
pub use estimand::Estimand;
pub use identifier::{CausalIdentifier, DoublyRobust, Orthogonality, RLearner};
pub use nuisance::{NuisanceEstimator, PluginEstimator};
pub use posterior::{infer_causal, CausalPosterior};

// Re-exports
pub use estimand::{prior_ate, TreatmentSpec};

/// A causal inference problem specification.
pub struct CausalProblem<T: CausalIdentifier> {
    /// Prior distribution over the causal estimand: log π(θ)
    pub estimand_prior: Box<dyn Fn(f64) -> f64>,
    /// Strategy for identifying causal effects from observational data
    pub identifier: T,
    /// Estimator for nuisance components (propensity, outcomes, etc.)
    pub nuisance_estimator: Box<dyn NuisanceEstimator>,
    /// Number of folds for cross-fitting
    pub folds: usize,
}
