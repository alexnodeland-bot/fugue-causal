# Introduction: Causal Inference for Fugue Probabilistic Programs

## What is Fugue-Causal?

**Fugue-causal is an extension library for [fugue](https://github.com/alexnodeland/fugue).** It is NOT standalone.

It adds loss-based Bayesian causal inference directly to probabilistic traces, enabling formal causal reasoning on programs that generate observational data.

**Core insight:** Fugue traces ARE causal data sources. Instead of writing separate causal inference pipelines, condition traces via causal loss functions using effect handlers. The posterior is your causal estimate.

## The Problem

Standard Bayesian causal inference on observational data requires:

1. **Full generative model**: Specify P(X, A, Y | ξ) completely
2. **Nuisance priors**: Place priors on propensity scores, outcome models, confounders
3. **Full inference**: Compute posterior over ξ, then marginalize to get causal effect posterior

**Issues:**
- Model misspecification → invalid inferences
- Prior elicitation on nuisances is indirect and hard to validate
- Coupling between causal parameters and nuisances makes inference fragile

## The Solution: Generalized Bayes on Traces

Instead of building a full generative model:

1. **Use fugue traces as data** — Your probabilistic program IS your data generation
2. **Place priors directly on causal estimands** θ (e.g., ATE, CATE)
3. **Update via identification-driven loss** (not likelihood)
4. **Use Gibbs posteriors**: q(θ | D) ∝ exp{−ωn·L_n(θ)} · π(θ)

**Benefits:**
- ✅ Traces encapsulate data generation — no need for explicit models
- ✅ Formal orthogonality → nuisance robustness
- ✅ Convergence guarantees via Theorem 5.1 (Javurek et al. 2026)
- ✅ Composable with any loss function (RA, IPW, DR, R-learner, etc.)

## Key Features

### Identifiers (Strategies for Identifying Causal Effects)
- **Regression Adjustment (RA)**: m̂₁(X) - m̂₀(X)
- **Inverse Probability Weighting (IPW)**: (A·Y/ê - (1-A)·Y/(1-ê))
- **Doubly Robust (DR/AIPW)**: Combines RA + IPW (orthogonal, robust)
- **R-Learner**: Residual-on-residual (always orthogonal)

### Estimands (Causal Quantities)
- **ATE**: Average Treatment Effect
- **CATE**: Conditional ATE (stratified)
- **ATT/ATU**: Effects on treated/untreated
- **HTE**: Heterogeneous treatment effects
- **Custom**: User-defined estimands

### Guarantees
- **Neyman-Orthogonality**: Second-order nuisance robustness
- **Cross-Fitting**: Eliminates empirical process bias
- **Credible Intervals**: Valid frequentist coverage (bootstrap calibrated)
- **Checkpointing**: Serialize/load posteriors for reproducibility

## Quick Example

```rust
use fugue_causal::*;
use fugue_causal::fugue_integration::{TraceObservation, infer_from_traces};

// Implement TraceObservation for your trace output type
impl TraceObservation for MyTrace {
    fn extract_covariates(&self) -> Vec<f64> {
        // Return confounders, demographics, etc.
        vec![self.age, self.income, /* ... */]
    }
    fn extract_treatment(&self) -> f64 {
        // Return 0.0 or 1.0
        if self.received_intervention { 1.0 } else { 0.0 }
    }
    fn extract_outcome(&self) -> f64 {
        // Return continuous outcome
        self.health_score
    }
}

// Run your fugue probabilistic program
let traces = my_program();

// Infer causal effects
let posterior = infer_from_traces(
    traces,                           // Traces from your program
    DoublyRobust,                     // Orthogonal identifier
    Box::new(PluginEstimator),        // Nuisance estimator
    5,                                // K-fold cross-fitting
)?;

println!("Effect: {:.4} ± {:.4}", posterior.point_estimate, posterior.posterior_sd);
```

## Why This Matters

### For Probabilistic Programming
Fugue programs generate observational data. Fugue-causal turns that data into formal causal estimates with uncertainty quantification—no separate inference pipeline needed.

### For Causal Inference
Standard causal inference tools require you to implement nuisance estimation, cross-fitting, and loss evaluation manually. Fugue-causal integrates all of this, with formal guarantees baked in.

### For Science
The posterior comes with formal convergence guarantees under Neyman-orthogonality. If your identifier is orthogonal (like DR or R-learner), nuisance estimation error can't invalidate your inference.

## Theory

**Source Paper**: Javurek, E., et al. (2026). "Generalized Bayes for Causal Inference."  
ArXiv: [2603.03035v1](https://arxiv.org/abs/2603.03035)

**Theorem 5.1** (Posterior Stability Under Orthogonality):
Under Neyman-orthogonality + cross-fitting:
```
TV(q_n,feasible, q_n,oracle) = O_P(√n · r_n²)
```
where r_n is nuisance estimation error.

**Key insight:** Error propagates as r_n², not r_n. This second-order robustness is what makes formal causal inference possible.

## Integration with Fugue

Fugue traces are the data source. Implement `TraceObservation` for your trace output type and call `infer_from_traces()`:

```rust
use fugue_causal::fugue_integration::TraceObservation;

impl TraceObservation for MyTrace {
    fn extract_covariates(&self) -> Vec<f64> { /* ... */ }
    fn extract_treatment(&self) -> f64 { /* ... */ }
    fn extract_outcome(&self) -> f64 { /* ... */ }
}

// Then:
let posterior = infer_from_traces(traces, identifier, estimator, folds)?;
```

No modifications to your fugue program needed. No explicit data formats. Just implement one trait.

## Modules

- `estimand`: Causal quantities (ATE, CATE, HTE, etc.)
- `identifier`: Identification strategies (RA, IPW, DR, R-learner)
- `nuisance`: Nuisance component estimation
- `posterior`: Gibbs posterior inference + credible intervals
- `cross_fit`: K-fold cross-fitting for orthogonality
- `bootstrap`: ω calibration for frequentist coverage
- `checkpoint`: Serialization for reproducibility
- `fugue_integration`: Bridge between fugue traces and causal inference

## Use Cases

### 1. Production ML Systems
- **Monitoring**: Detect causal shifts in recommenders
- **A/B Testing**: Valid uncertainty quantification
- **Policy Learning**: Heterogeneous policy evaluation

### 2. Scientific Research
- **Economics**: Labor market causal effects with formal inference
- **Health**: Drug efficacy with uncertainty
- **Climate**: Policy impact analysis

### 3. Generative Systems
- Use causal inference to validate or tune generative models
- Identify which parameters/features causally affect outcomes

## Next Steps

- **[Quick Start](quick-start.md)**: 5-minute setup
- **[Theory](theory.md)**: Deep dive into orthogonality & convergence
- **[Tutorials](tutorials.md)**: Step-by-step examples
- **[Examples](examples.md)**: ATE, CATE
- **[API Reference](api-reference.md)**: All public APIs

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
