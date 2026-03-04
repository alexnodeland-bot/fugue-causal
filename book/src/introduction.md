# Fugue-Causal: Introduction

## What is Fugue-Causal?

**Fugue-Causal** is a Rust library for **Bayesian causal inference via generalized Bayes** (Gibbs posteriors). It provides loss-based causal identification with Neyman-orthogonal estimators and formal uncertainty quantification, designed for composability and extensibility in production systems.

**Future integration** with the [fugue](https://github.com/alexnodeland/fugue) probabilistic programming library is planned for v1.1+ (probabilistic traces as causal priors).

## The Problem

Standard Bayesian causal inference requires:
1. **Full generative models**: Specify P(X, A, Y | ξ) completely
2. **Nuisance priors**: Place priors on propensity scores, outcome models, etc.
3. **Full posterior inference**: Marginalize over all nuisances to get causal effect posterior

**Issues:**
- Model misspecification → invalid inferences
- Prior elicitation is indirect and hard to validate
- Computationally expensive

## The Solution: Generalized Bayes

Instead:
1. **Place priors directly on causal estimands** θ (e.g., ATE, CATE)
2. **Update via identification-driven loss** (not likelihood)
3. **Use Gibbs posteriors**: q(θ | D) ∝ exp{−ωn·L_n(θ)} · π(θ)

**Benefits:**
- ✅ Avoids full generative model specification
- ✅ Formal orthogonality → nuisance robustness
- ✅ Convergence guarantees via Theorem 5.1 (Javurek et al. 2026)
- ✅ Composable with any loss function (RA, IPW, DR, R-learner, etc.)

## Key Features

### Identifiers (Strategies)
- **Regression Adjustment (RA)**: m̂₁(X) - m̂₀(X)
- **Inverse Probability Weighting (IPW)**: (A·Y/ê - (1-A)·Y/(1-ê))
- **Doubly Robust (DR/AIPW)**: Combines RA + IPW (orthogonal)
- **R-Learner**: Residual-on-residual (always orthogonal)

### Estimands
- **ATE**: Average Treatment Effect
- **CATE**: Conditional ATE (stratified)
- **ATT/ATU**: Effects on treated/untreated
- **HTE**: Heterogeneous treatment effects
- **Custom**: User-defined estimands

### Guarantees
- **Neyman-Orthogonality**: Second-order nuisance robustness
- **Cross-Fitting**: Eliminates empirical process bias
- **Credible Intervals**: Valid frequentist coverage (via bootstrap calibration)

## Quick Example

```rust
use fugue_causal::*;

// Generate synthetic data
let observations = vec![
    vec![x1, a1, y1],  // [covariate, treatment, outcome]
    // ... more observations
];

// Run inference
let posterior = infer_causal(
    prior_ate(),           // Prior on ATE
    DoublyRobust,          // Orthogonal identifier
    Box::new(PluginEstimator),  // Nuisance estimator
    5,                     // K-fold cross-fitting
    &observations,
)?;

// Results
println!("ATE: {:.4}", posterior.point_estimate);
println!("Posterior Std: {:.4}", posterior.posterior_sd);
```

## Paper & Theory

**Source**: Javurek, E., et al. (2026). "Generalized Bayes for Causal Inference."  
ArXiv: [2603.03035v1](https://arxiv.org/abs/2603.03035)

**Theorem 5.1** (Orthogonality Robustness):
Under Neyman-orthogonality + cross-fitting:
```
TV(q_{n,fe}^S, q_{n,or}^S) = O_P(√n · r_n²)
```
where r_n is nuisance estimation error. **Key:** Error propagates as r_n², not r_n!

## Use Cases

### 1. Production ML Systems
- **Monitoring**: Detect causal shifts in recommenders
- **A/B Testing**: Valid uncertainty quantification
- **Policy Learning**: Heterogeneous policy evaluation

### 2. Scientific Research
- **Economics**: Labor market causal effects
- **Health**: Drug efficacy with uncertainty
- **Climate**: Policy impact analysis

## Next Steps

- **[Quick Start](quick-start.md)**: 5-minute setup
- **[Theory](theory.md)**: Deep dive into orthogonality & convergence
- **[Tutorials](tutorials.md)**: Step-by-step examples
- **[Examples](examples.md)**: ATE, CATE

## Modules

- `estimand`: Causal quantities (ATE, CATE, HTE, etc.)
- `identifier`: Identification strategies (RA, IPW, DR, R-learner)
- `nuisance`: Nuisance component estimation
- `posterior`: Gibbs posterior inference + credible intervals
- `cross_fit`: K-fold cross-fitting for orthogonality
- `bootstrap`: ω calibration for frequentist coverage

## Citation

```bibtex
@article{javurek2026generalizedbayes,
  title={Generalized Bayes for Causal Inference},
  author={Javurek, E. and others},
  journal={arXiv preprint},
  year={2026},
  eprint={2603.03035}
}
```
