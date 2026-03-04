# fugue-causal

**Bayesian Causal Inference via Generalized Bayes**

A Rust library for loss-based causal inference using generalized (Gibbs) posteriors, **integrated with the [fugue](https://github.com/alexnodeland/fugue) probabilistic programming library**. Features Neyman-orthogonal identifiers and formal uncertainty quantification.

## Quick Start

```rust
use fugue_causal::*;

// Estimate average treatment effect (ATE)
let posterior = infer_causal(
    prior_ate(),                      // Prior on causal effect
    DoublyRobust,                     // Identification strategy (orthogonal)
    Box::new(PluginEstimator),        // Nuisance estimator
    5,                                // K-fold cross-fitting
    &observations,                    // Data: Vec<(covariate, treatment, outcome)>
)?;

println!("ATE: {:.4} ± {:.4}", posterior.point_estimate, posterior.posterior_sd);
```

## Why This Matters

**Standard Bayesian causal inference is brittle:**
- Requires specifying full data-generating models (P(X,A,Y|ξ))
- High-dimensional nuisance priors are hard to elicit
- Vulnerable to model misspecification

**Generalized Bayes is cleaner:**
- Skip the likelihood entirely
- Place priors directly on causal estimands (ATE, CATE, etc.)
- Update via identification-driven loss functions
- Formal robustness: Neyman-orthogonal losses give O_P(√n r²_n) nuisance robustness (vs. O_P(√n r_n))

## Key Features

- **Composable identifiers**: RA, IPW, DR/AIPW, R-learner, custom
- **Formal uncertainty quantification**: Gibbs posteriors with bootstrap calibration
- **Cross-fitting preservation**: Maintains orthogonality for valid inference
- **Estimand-native priors**: Place beliefs directly on treatment effects, not nuisances
- **Generics over estimands**: ATE, CATE, ATT, ATU, HTE, custom

## Theory

See [SPEC.md](./SPEC.md) for the full framework, theorems, and integration design.

**Core Theorem (5.1):** Under Neyman-orthogonality + cross-fitting:
```
TV(feasible_posterior, oracle_posterior) = O_P(√n · r_n²)
```
where r_n is nuisance estimation error. If r_n = o(n^{-1/4}), posteriors are asymptotically indistinguishable.

## Integration with Fugue

**Fugue Integration (v1.0+):**
- Use fugue probabilistic programs as causal data sources
- Convert trace outputs to causal observations via `TraceObservation` trait
- Run causal inference directly on probabilistic traces with `infer_from_traces()`
- Effect handlers for conditioning traces on causal losses

**Design Philosophy:**
- Causal identifiers + loss functions work *independent* of fugue (no hard dependency)
- Fugue integration is *additive*: use fugue traces for data, or provide observations directly
- Extensible via traits: `CausalIdentifier`, `NuisanceEstimator`, `TraceObservation`

## Composability & Future Extensions

**v1.1+ (Q2 2026):**
- **Causal-Evolutionary Search** via [fugue-evo](https://github.com/alexnodeland/fugue-evo): GA search over identifier + prior + estimator combinations
- Continuous treatments, multiple outcomes, meta-learner ensemble
- GPU acceleration (CUDA)
- Domain-specific applications (synthesis, quantum, ML pipelines) as separate projects using fugue-causal as a library

## Paper

**Source:** [Generalized Bayes for Causal Inference](https://arxiv.org/abs/2603.03035v1)  
Javurek, E., et al. (2026). ArXiv:2603.03035v1  

## Status

**v1.0.0 Production Release**: All core features complete. 40 tests passing. Full documentation. Ready for use in production causal inference pipelines. Designed for extensibility — implement custom identifiers and nuisance estimators via traits.

---

**Library ecosystem:** [fugue](https://github.com/alexnodeland/fugue) (probabilistic programming), [fugue-evo](https://github.com/alexnodeland/fugue-evo) (genetic algorithms)
