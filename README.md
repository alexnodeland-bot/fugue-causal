# fugue-causal

**Bayesian Causal Inference Extension for Fugue**

An extension library for the [fugue](https://github.com/alexnodeland/fugue) probabilistic programming library. Adds loss-based causal inference using generalized (Gibbs) posteriors directly on probabilistic traces.

## What is fugue-causal?

**fugue-causal is not standalone.** It's built on top of fugue and extends its trace system with:

1. **Causal Identifiers** — Loss-based identification strategies (RA, IPW, DR/AIPW, R-learner)
2. **Gibbs Posteriors** — Bayesian inference on causal estimands via exp{-ωn·L_n(θ)}·π(θ)
3. **Cross-Fitting** — Neyman-orthogonal parameter estimation for formal robustness
4. **Uncertainty Quantification** — Credible intervals with bootstrap calibration

The core architecture treats **fugue traces as causal data sources**. Effect handlers apply causal losses, enabling direct inference on trace outputs without ever specifying full generative models.

## Quick Start

Use fugue to generate observational data. Implement `TraceObservation` for your trace type. Run causal inference:

```rust
use fugue_causal::*;
use fugue_causal::fugue_integration::{TraceObservation, infer_from_traces};

// Your trace output type
impl TraceObservation for MyTrace {
    fn extract_covariates(&self) -> Vec<f64> { /* ... */ }
    fn extract_treatment(&self) -> f64 { /* ... */ }
    fn extract_outcome(&self) -> f64 { /* ... */ }
}

// Run fugue program → collect traces → infer causally
let traces = my_probabilistic_program();
let posterior = infer_from_traces(
    traces,
    DoublyRobust,
    Box::new(PluginEstimator),
    5,  // K-fold
)?;

println!("ATE: {:.4} ± {:.4}", posterior.point_estimate, posterior.posterior_sd);
```

## Why This Matters

**Standard Bayesian causal inference is fragile:**
- Requires full data-generating models: P(X, A, Y | ξ)
- Nuisance priors are indirect and hard to elicit
- Vulnerable to model misspecification

**Generalized Bayes is cleaner:**
- Skip the likelihood entirely
- Place priors directly on causal estimands (ATE, CATE, etc.)
- Update via identification-driven loss functions
- Formal robustness: Neyman-orthogonal losses give O_P(√n r_n²) nuisance robustness (vs. O_P(√n r_n))

## Core Features

### Identifiers (4 built-in + custom)
- **Regression Adjustment (RA)**: m̂₁(X) - m̂₀(X)
- **Inverse Probability Weighting (IPW)**: A·Y/ê - (1-A)·Y/(1-ê)
- **Doubly Robust/AIPW**: Combines RA + IPW (Neyman-orthogonal)
- **R-Learner**: Residual-on-residual (always orthogonal)

### Estimands
- **ATE**: Average Treatment Effect
- **CATE**: Conditional ATE (stratified by covariate)
- **ATT/ATU**: Effects on treated/untreated
- **HTE**: Heterogeneous treatment effects
- **Custom**: User-defined estimands

### Guarantees
- **Neyman-Orthogonality**: Second-order nuisance robustness
- **Cross-Fitting**: Eliminates empirical process bias
- **Credible Intervals**: Valid frequentist coverage (bootstrap calibrated)
- **Checkpointing**: Serialize posteriors for reproducibility

## Theory

**Source:** Javurek, E., et al. (2026). "Generalized Bayes for Causal Inference."  
ArXiv: [2603.03035v1](https://arxiv.org/abs/2603.03035)

**Theorem 5.1** (Posterior Stability Under Orthogonality):
```
TV(q_n,fe, q_n,or) = O_P(√n · r_n²)
```
where r_n is nuisance estimation error. Under Neyman-orthogonality + cross-fitting, 
feasible posteriors are asymptotically indistinguishable from oracle posteriors.

## Status

**v1.0.0 Production Release**
- ✅ Core causal inference engine (40 tests passing)
- ✅ Full CI/CD (GitHub Actions)
- ✅ mdBook documentation (8 chapters, 50KB)
- ✅ API stability guarantee (v1.0 → v1.x backward compatible)
- ✅ Fugue integration layer (TraceObservation trait, infer_from_traces)

**v1.1+ (Q2 2026):**
- Continuous treatments
- Multiple outcomes
- Meta-learner ensemble
- GPU acceleration (CUDA)

## Installation

Requires [fugue](https://github.com/alexnodeland/fugue) as a dependency:

```toml
[dependencies]
fugue = { path = "../fugue" }  # Once available as registry package
fugue-causal = "1.0.0"
```

## Examples

```bash
# Direct causal inference (no traces needed)
cargo run --example ate_basic

# Heterogeneous treatment effects
cargo run --example cate_heterogeneous
```

See [book/src/examples.md](book/src/examples.md) for full walkthroughs.

## Documentation

- **[Introduction](book/src/introduction.md)**: Problem statement, solution, integration with fugue
- **[Theory](book/src/theory.md)**: Deep dive on orthogonality, convergence rates
- **[Quick Start](book/src/quick-start.md)**: 5-minute setup guide
- **[Tutorials](book/src/tutorials.md)**: Step-by-step walkthroughs
- **[API Reference](book/src/api-reference.md)**: All public functions and traits
- **[Examples](book/src/examples.md)**: Working code samples
- **[FAQ](book/src/faq.md)**: Troubleshooting and common questions

Build locally: `mdbook serve book/`

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

## Ecosystem

- **[fugue](https://github.com/alexnodeland/fugue)** — Probabilistic programming library (core dependency)
- **[fugue-evo](https://github.com/alexnodeland/fugue-evo)** — Evolutionary algorithms for causal estimator search
- **[quiver](https://github.com/alexnodeland/quiver)** — Audio synthesis library (domain application)

---

**License:** MIT  
**Repository:** https://github.com/alexnodeland/fugue-causal
