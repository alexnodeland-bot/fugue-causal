# fugue-causal

**Bayesian Causal Inference via Generalized Bayes**

Extends the [fugue](https://github.com/alexnodeland/fugue) probabilistic programming library with loss-based causal inference using generalized (Gibbs) posteriors.

## Quick Start

```rust
use fugue_causal::*;

let problem = CausalProblem {
    estimand_prior: prior_ate(),
    identifier: DoublyRobust,
    nuisance_estimator: Box::new(CausalForest::new()),
    folds: 5,
};

let posterior = infer_causal(problem, observational_data)?;
let credible_interval = posterior.credible_interval(0.95)?;
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

## Integration with fugue-evo

**Causal-Evolutionary Search**: Use fugue-evo genetic algorithms to search over causal inference strategies — find which identifier + prior + nuisance estimator works best for your problem.

Domain-specific applications (synthesis, quantum circuits, etc.) belong in separate projects that use fugue-causal as a library.

## Paper

**Source:** [Generalized Bayes for Causal Inference](https://arxiv.org/abs/2603.03035v1)  
Javurek, E., et al. (2026). ArXiv:2603.03035v1  

## Status

**Specification Phase:** SPEC.md complete. Implementation roadmap in place. Ready for core framework + integration work.

---

**Library ecosystem:** [fugue](https://github.com/alexnodeland/fugue) (probabilistic programming), [fugue-evo](https://github.com/alexnodeland/fugue-evo) (genetic algorithms)
