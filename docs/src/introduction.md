# Introduction

## What is fugue-causal?

**fugue-causal** is a Rust library for Bayesian causal inference using *generalized Bayes* (Gibbs posteriors). It bridges causal machine learning and probabilistic programming, providing a formal framework for uncertainty quantification in causal effect estimation.

**Source:** [Generalized Bayes for Causal Inference](https://arxiv.org/abs/2603.03035) (Javurek et al., 2026)

## The Problem

Standard Bayesian causal inference is brittle:

1. **Requires full data-generating models** — You must specify P(X, A, Y | ξ) completely
2. **High-dimensional nuisance priors** — Priors on propensity scores and outcome regressions are indirect and hard to elicit
3. **Feedback coupling** — Joint likelihoods couple nuisance posteriors, undermining robustness
4. **Indirect priors on causal effects** — Prior on the treatment effect θ is determined indirectly through the model specification

**Result:** Posterior is vulnerable to model misspecification, even when the causal identification is valid.

## The Solution

**Skip the likelihood.** Place priors directly on causal estimands θ and update via *identification-driven loss functions*.

- **No data model** — You don't specify P(X, A, Y | ξ)
- **Direct estimand priors** — π(θ) speaks directly about treatment effects
- **Composable loss-based identification** — Swap between identification strategies (RA, IPW, DR, R-learner, etc.)
- **Formal robustness** — Neyman-orthogonal losses provide second-order robustness to nuisance estimation error

## Key Advantages

| Feature | Standard Bayes | Generalized Bayes |
|---------|---|---|
| Data model required | ✓ Complex | ✗ Not needed |
| Prior on estimand | Indirect | Direct & interpretable |
| Orthogonal identifiers | No (O_P(√n r_n)) | Yes (O_P(√n r²_n)) |
| Composable identifiers | ✗ No | ✓ Yes |
| Nuisance robustness | Fragile | Proven formally |

## When to Use fugue-causal

✓ You want formal uncertainty quantification for causal effects  
✓ You want to compose different identification strategies  
✓ You want robustness to nuisance estimation error  
✓ You're using causal machine learning pipelines (double ML, causal forests, etc.)  
✓ You care about production-ready causal inference with formal guarantees  

✗ You have strong priors on the full data-generating model  
✗ You're doing pure causal discovery (structure learning)  
✗ Your problem is simple enough for frequentist point estimates  

## Architecture

fugue-causal is built on three layers:

1. **Estimands** — What causal quantity are you interested in? (ATE, CATE, ATT, etc.)
2. **Identifiers** — How do you recover the estimand from observational data? (RA, IPW, DR, R-learner, etc.)
3. **Posterior** — Gibbs posterior construction with formal uncertainty quantification

All layers are composable: swap estimands, swap identifiers, swap priors—same framework.

## Next Steps

- **[Quick Start](./quick-start.md)** — Get running in 5 minutes
- **[Core Concepts](./concepts.md)** — Understand the theory
- **[Use Cases](./use-cases.md)** — See what you can build
