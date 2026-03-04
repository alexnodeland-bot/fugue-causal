//! Causal identification strategies and their properties.
//!
//! An identifier is a strategy for recovering causal effects from observational data.
//! Each identifier specifies nuisances (e.g., propensity score, outcome regression),
//! a per-observation loss function, and orthogonality properties.

use crate::estimand::Estimand;

/// Orthogonality of a causal identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orthogonality {
    /// Neyman-orthogonal: D_η D_θ L = 0
    /// Nuisance error propagates as O_P(√n r_n²)
    Neyman,
    
    /// Partially orthogonal (e.g., one-way)
    /// Nuisance error propagates as O_P(√n r_n r_n')
    Partial,
    
    /// Non-orthogonal
    /// Nuisance error propagates as O_P(√n r_n)
    None,
}

/// Convergence rate requirement for nuisances
#[derive(Clone, Copy, Debug)]
pub enum NuisanceRate {
    /// Parametric rate: n^{-1/2}
    Parametric,
    
    /// Slow rate: o(n^{-1/4})
    Slow,
    
    /// Faster rate: o(n^{-1/2})
    Fast,
}

/// Trait for causal identification strategies
pub trait CausalIdentifier: Send + Sync {
    /// Per-observation loss function: ℓ^S(O; θ, η)
    /// Returns scalar loss for given observation, estimand value, and nuisance estimates
    fn loss(&self, observation: &[f64], estimand_value: f64, nuisances: &[f64]) -> f64;
    
    /// Nuisance components required by this strategy
    fn nuisance_names(&self) -> Vec<&'static str>;
    
    /// Orthogonality property of this identifier
    fn orthogonality(&self) -> Orthogonality;
    
    /// Convergence rate requirement for nuisances
    fn nuisance_rate_requirement(&self) -> NuisanceRate;
    
    /// Human-readable name
    fn name(&self) -> &'static str;
}

/// Regression Adjustment
pub struct RegressionAdjustment;

impl CausalIdentifier for RegressionAdjustment {
    fn loss(&self, observation: &[f64], estimand_value: f64, nuisances: &[f64]) -> f64 {
        // Pseudo-outcome: m_1(X) - m_0(X)
        // Loss: (pseudo_outcome - θ)²
        let pseudo_outcome = nuisances[0] - nuisances[1];
        (pseudo_outcome - estimand_value).powi(2)
    }
    
    fn nuisance_names(&self) -> Vec<&'static str> {
        vec!["m_1", "m_0"]  // outcome models
    }
    
    fn orthogonality(&self) -> Orthogonality {
        Orthogonality::None
    }
    
    fn nuisance_rate_requirement(&self) -> NuisanceRate {
        NuisanceRate::Parametric
    }
    
    fn name(&self) -> &'static str {
        "Regression Adjustment"
    }
}

/// Inverse Probability Weighting
pub struct InverseProbabilityWeighting;

impl CausalIdentifier for InverseProbabilityWeighting {
    fn loss(&self, observation: &[f64], estimand_value: f64, nuisances: &[f64]) -> f64 {
        // Pseudo-outcome: A·Y/e(X) - (1-A)·Y/(1-e(X))
        let a = observation[0];
        let y = observation[1];
        let e = nuisances[0];
        
        let pseudo_outcome = (a * y) / e - ((1.0 - a) * y) / (1.0 - e);
        (pseudo_outcome - estimand_value).powi(2)
    }
    
    fn nuisance_names(&self) -> Vec<&'static str> {
        vec!["e"]  // propensity score
    }
    
    fn orthogonality(&self) -> Orthogonality {
        Orthogonality::None
    }
    
    fn nuisance_rate_requirement(&self) -> NuisanceRate {
        NuisanceRate::Parametric
    }
    
    fn name(&self) -> &'static str {
        "Inverse Probability Weighting"
    }
}

/// Doubly Robust / AIPW (Augmented IPW)
pub struct DoublyRobust;

impl CausalIdentifier for DoublyRobust {
    fn loss(&self, observation: &[f64], estimand_value: f64, nuisances: &[f64]) -> f64 {
        // Pseudo-outcome:
        // (A/e - (1-A)/(1-e)) · (Y - m_A) + m_1 - m_0
        let a = observation[0];
        let y = observation[1];
        let e = nuisances[0];
        let m_1 = nuisances[1];
        let m_0 = nuisances[2];
        let m_a = if a > 0.5 { m_1 } else { m_0 };
        
        let ipw_term = (a / e) - ((1.0 - a) / (1.0 - e));
        let pseudo_outcome = ipw_term * (y - m_a) + m_1 - m_0;
        
        (pseudo_outcome - estimand_value).powi(2)
    }
    
    fn nuisance_names(&self) -> Vec<&'static str> {
        vec!["e", "m_1", "m_0"]  // propensity + outcome models
    }
    
    fn orthogonality(&self) -> Orthogonality {
        Orthogonality::Neyman
    }
    
    fn nuisance_rate_requirement(&self) -> NuisanceRate {
        NuisanceRate::Slow
    }
    
    fn name(&self) -> &'static str {
        "Doubly Robust (AIPW)"
    }
}
