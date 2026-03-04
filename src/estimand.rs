//! Causal estimand specifications.
//!
//! An estimand is the causal quantity we want to estimate: ATE, CATE, ATT, etc.

/// Specification of a treatment variable
#[derive(Clone, Debug)]
pub struct TreatmentSpec {
    pub name: String,
    pub treatment_value: f64,
    pub control_value: f64,
}

/// A causal estimand (quantity of interest)
#[derive(Clone, Debug)]
pub enum Estimand {
    /// Average Treatment Effect: E[Y(1) - Y(0)]
    ATE(TreatmentSpec),

    /// Conditional ATE: E[Y(1) - Y(0) | X ∈ strata]
    CATE {
        treatment: TreatmentSpec,
        conditioning_vars: Vec<String>,
    },

    /// Average Treatment Effect on the Treated: E[Y(1) - Y(0) | A = 1]
    ATT(TreatmentSpec),

    /// Average Treatment Effect on the Untreated: E[Y(1) - Y(0) | A = 0]
    ATU(TreatmentSpec),

    /// Heterogeneous Treatment Effects: θ(x) = E[Y(1) - Y(0) | X = x]
    HTE(TreatmentSpec),

    /// Custom estimand defined by user function
    Custom(String),
}

/// Create a standard ATE prior (e.g., N(0, 1) in standardized units)
pub fn prior_ate() -> Box<dyn Fn(f64) -> f64> {
    Box::new(|_theta: f64| {
        // log π(θ) for a standard normal prior
        -0.5 // Constant factor for log N(0, 1)
    })
}

/// Create a domain-informed prior on ATE
pub fn prior_ate_informed(mean: f64, variance: f64) -> Box<dyn Fn(f64) -> f64> {
    Box::new(move |theta: f64| {
        // log π(θ) for N(mean, variance)
        let residual = theta - mean;
        -0.5 * (residual * residual / variance) - 0.5 * (variance.ln())
    })
}
