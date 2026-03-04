# CLAUDE.md

Development guidance for Claude Code (claude.ai/code) when working on fugue-causal.

## Build & Test Commands

```bash
# Build
cargo build

# Build release
cargo build --release

# Run all tests
cargo test

# Run a specific test
cargo test test_name

# Run tests in a module
cargo test module_name::

# Run tests with output
cargo test -- --nocapture

# Property-based tests
cargo test --test property_tests

# Clippy linting
cargo clippy --all-targets --all-features -- -D warnings

# Format code
cargo fmt

# Check documentation
cargo doc --no-deps --all-features

# Full CI pipeline
make ci

# Watch for changes (requires cargo-watch)
make watch

# Serve docs locally
make doc-serve
```

## Architecture

**fugue-causal** is a Bayesian causal inference library for Rust using generalized Bayes (Gibbs posteriors).

**Core paper:** Javurek et al. (2026). "Generalized Bayes for Causal Inference." ArXiv:2603.03035v1

### Core Abstractions

1. **`CausalIdentifier` trait** (src/identifier.rs)
   - Defines how to identify causal effects from observational data
   - Implements loss-based estimation strategies (RA, IPW, DR, R-learner, etc.)
   - Metadata: orthogonality class, nuisance rate requirements

2. **`Estimand` enum** (src/estimand.rs)
   - What causal quantity are you targeting? (ATE, CATE, ATT, ATU, HTE, custom)
   - Prior specification on the estimand (not on data-generating model)

3. **`NuisanceEstimator` trait** (src/nuisance.rs)
   - Estimates auxiliary components needed by identifiers
   - Pluggable: can use any ML model (random forest, XGBoost, neural net, etc.)
   - Cross-fitting support for orthogonality preservation

4. **`CausalPosterior` struct** (src/posterior.rs)
   - Result of Gibbs posterior inference
   - Point estimate + posterior SD + ω calibration
   - Credible interval computation with frequentist coverage

### Module Organization

```
src/
├── lib.rs           # Library root, re-exports
├── estimand.rs      # Causal estimands (ATE, CATE, ATT, etc.)
├── identifier.rs    # Identification strategies (RA, IPW, DR)
├── nuisance.rs      # Nuisance estimation interface
├── posterior.rs      # Gibbs posterior construction & inference
├── cross_fit.rs     # Cross-fitting orchestration (Phase 1)
├── bootstrap.rs     # ω calibration via bootstrap (Phase 2)
└── examples.rs      # Runnable examples (Phase 3)
```

### Type Patterns

Identifiers use trait objects for composability:

```rust
pub trait CausalIdentifier: Send + Sync {
    fn loss(&self, observation: &[f64], estimand_value: f64, nuisances: &[f64]) -> f64;
    fn nuisance_names(&self) -> Vec<&'static str>;
    fn orthogonality(&self) -> Orthogonality;
    fn nuisance_rate_requirement(&self) -> NuisanceRate;
    fn name(&self) -> &'static str;
}
```

Posteriors are generic over the inference method:

```rust
pub fn infer_causal<T: CausalIdentifier>(
    estimand_prior: Box<dyn Fn(f64) -> f64>,
    identifier: T,
    nuisance_estimator: Box<dyn NuisanceEstimator>,
    folds: usize,
    data: &[(Vec<f64>, f64, f64)],
) -> Result<CausalPosterior, String>
```

## Development Phases

### Phase 1: Core Framework (Weeks 1-2) ← START HERE
- [ ] Cross-fitting orchestration
- [ ] Empirical loss computation
- [ ] Gibbs posterior basic construction
- [ ] Integration tests (end-to-end inference)
- [ ] Move from scaffolds to real logic

### Phase 2: Robustness & Calibration (Weeks 3-4)
- [ ] Bootstrap calibration for ω
- [ ] Heterogeneous effects (CATE as function)
- [ ] Meta-learner ensemble (combine identifiers)
- [ ] Convergence diagnostics
- [ ] Property-based tests

### Phase 3: Production Features (Weeks 5-6)
- [ ] Checkpointing (pause/resume inference)
- [ ] Parallel cross-fitting (rayon)
- [ ] Feature flags (std, parallel, checkpoint)
- [ ] Examples: ATE, CATE, synthesis, quantum
- [ ] Performance benchmarks

### Phase 4: Polish & Release (Weeks 7+)
- [ ] API stability audit
- [ ] Documentation tutorials
- [ ] Edge case hardening
- [ ] WASM crate (optional, like fugue-evo-wasm)
- [ ] v1.0 release

## Key Design Decisions

1. **Loss-based, not likelihood-based** — Generalized Bayes avoids specifying P(X,A,Y|ξ). Instead, place priors on estimands and update via loss.

2. **Cross-fitting by default** — Preserves Neyman-orthogonality for robust inference.

3. **Pluggable estimators** — Any ML model can estimate nuisances; framework is agnostic.

4. **Composable identifiers** — Swap RA ↔ IPW ↔ DR; same inference pipeline.

5. **Bootstrap calibration** — ω tuned for frequentist coverage, enabling valid uncertainty quantification.

## Testing Strategy

- **Unit tests** — Each identifier's loss function, orthogonality checks
- **Integration tests** — End-to-end inference on synthetic data
- **Property-based tests** — Correctness over large parameter spaces
- **Benchmarks** — Performance on varying data sizes

Example synthetic data generation (use in tests):
```rust
fn synthetic_ate_data(n: usize, true_ate: f64) -> Vec<(Vec<f64>, f64, f64)> {
    // Generate X ~ N(0,1), A ~ Bernoulli(0.5), Y = 2X + true_ate*A + N(0,1)
    // Known true causal effect for validation
}
```

## Important Files

- **SPEC.md** — Full technical specification (theory, algorithms, design)
- **README.md** — Quick start + overview
- **Makefile** — Build targets (make ci, make quick, make watch)
- **.github/workflows/ci.yml** — GitHub Actions pipeline

## Common Pitfalls

1. **Forgetting cross-fitting** — Violates orthogonality; posteriors will be biased.
2. **Wrong nuisance rate** — Orthogonal losses need O(n^{-1/4}); non-orthogonal need O(n^{-1/2}).
3. **ω = 1 without calibration** — Will give wrong coverage; always calibrate bootstrap.
4. **Ignoring overlap** — If P(A|X) near 0 or 1, IPW blows up; use DR instead.

## References

- Paper: https://arxiv.org/abs/2603.03035v1
- Kennedy (2023) on CATE: https://projecteuclid.org/articles/electronic-journal-of-statistics/Towards-optimal-doubly-robust-estimation-of-heterogeneous-treatment-effects/10.1214/23-EJS1161
- Chernozhukov et al. (2018) on double ML: https://arxiv.org/abs/1707.01217
