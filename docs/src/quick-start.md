# Quick Start

Get fugue-causal running in 5 minutes.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
fugue-causal = "0.1"
```

## Basic Example: Estimating ATE

```rust
use fugue_causal::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Your observational data: (covariates, treatment, outcome)
    let data = vec![
        (vec![1.0, 2.0], 1.0, 5.5),  // X, A, Y
        (vec![0.5, 1.5], 0.0, 3.2),
        (vec![2.0, 3.0], 1.0, 7.1),
        // ... more observations
    ];
    
    // Define the causal problem
    let problem = CausalProblem {
        estimand_prior: prior_ate(),
        identifier: DoublyRobust,  // Neyman-orthogonal
        nuisance_estimator: Box::new(SimplePluginEstimator),
        folds: 5,  // Cross-fitting
    };
    
    // Infer causal effect
    let posterior = infer_causal(
        problem.estimand_prior,
        problem.identifier,
        problem.nuisance_estimator,
        problem.folds,
        &data,
    )?;
    
    // Get results with uncertainty quantification
    println!("Point estimate: {:.3}", posterior.point_estimate);
    println!("Posterior SD: {:.3}", posterior.posterior_sd);
    
    let (lower, upper) = posterior.credible_interval(0.95);
    println!("95% credible interval: [{:.3}, {:.3}]", lower, upper);
    
    Ok(())
}
```

## What's Happening?

1. **Estimand** — ATE (Average Treatment Effect): E[Y(1) - Y(0)]
2. **Identifier** — Doubly Robust (DR/AIPW): Neyman-orthogonal, robust to nuisance error
3. **Prior** — Standard normal N(0, 1) on the ATE
4. **Cross-fitting** — Data split into 5 folds for nuisance estimation (preserves orthogonality)
5. **Posterior** — Gibbs posterior: q(θ | D) ∝ exp{-ωn·L_n(θ)} · π(θ)

## Key Concepts

### Neyman-Orthogonality

The DR identifier satisfies *Neyman-orthogonality*, which means:
- Nuisance estimation error doesn't directly affect the first-order optimality condition
- Convergence rate is O_P(√n r²_n), not O_P(√n r_n)
- You can use slower, more flexible nuisance estimators and still get valid inference

### Credible Intervals

The credible interval has **frequentist coverage** via bootstrap calibration of ω.
- A 95% credible set contains the true parameter ≈95% of the time (repeated sampling)
- This is unusual—Bayesian intervals usually don't have frequentist coverage!
- Here it works because we calibrate ω to enforce it

## Next Steps

- **[Estimands](./framework/estimands.md)** — Learn about ATE, CATE, ATT, custom estimands
- **[Identifiers](./framework/identifiers.md)** — Understand RA, IPW, DR, R-learner
- **[Use Cases](./use-cases.md)** — Synthesis, quantum, causal discovery applications

## Troubleshooting

**Q: How do I choose an identifier?**  
A: Use DR/AIPW (DoublyRobust) by default—it's Neyman-orthogonal and flexible.

**Q: What if my nuisance estimator is slow?**  
A: Orthogonal identifiers (DR) can handle nuisance rates up to O(n^{-1/4}). Non-orthogonal (RA, IPW) require O(n^{-1/2}).

**Q: Can I use my own identification strategy?**  
A: Yes! Implement the `CausalIdentifier` trait and define your loss function.

**Q: How do I estimate nuisances?**  
A: Implement `NuisanceEstimator`. Use any ML model: random forests, XGBoost, neural nets, etc.
