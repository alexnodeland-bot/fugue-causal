# fugue-causal v1.0 Roadmap

**Goal:** Production-quality Bayesian causal inference library on par with fugue-evo.  
**Target Date:** 2 weeks (6 development days)  
**Methodology:** Ralph Loops + iterative development

---

## Phase 1: Core Framework (Days 1-2)

### Objectives
- Implement full Gibbs posterior inference (not stubs)
- Cross-fitting orchestration with validation fold separation
- Empirical loss computation from data
- Integration tests proving end-to-end correctness

### Tasks
1. **Cross-Fitting Module** (`src/cross_fit.rs`)
   - Stratified K-fold split
   - Train/validation fold separation
   - Nuisance estimation on training folds only
   - Loss evaluation on validation folds

2. **Empirical Loss Computation** (expand `src/posterior.rs`)
   - Aggregate per-observation losses
   - Support for ATE (scalar) and CATE (function-valued)
   - Handle edge cases (propensity near 0/1, missing data)

3. **Gibbs Posterior Construction**
   - exp{-ωn·L_n(θ)} · π(θ) sampling/integration
   - Support both point estimate and full posterior
   - Variational approximation (optional)

4. **Integration Tests**
   - Synthetic ATE data with known ground truth
   - Verify posterior mean converges to truth
   - Verify credible intervals have correct coverage

### Success Criteria
- All 14 existing unit tests still pass
- 5+ integration tests for ATE/CATE inference
- End-to-end inference on synthetic data works
- No panics or unwrap() calls

---

## Phase 2: Robustness & Meta-Learners (Days 3-4)

### Objectives
- Bootstrap calibration for ω (frequentist validity)
- Heterogeneous treatment effects (CATE)
- Meta-learner ensemble (combine identifiers)
- Advanced identifiers (R-learner, DML variants)

### Tasks
1. **Bootstrap Calibration** (`src/bootstrap.rs`)
   - Empirical coverage of credible intervals
   - Tune ω for (1-α) coverage
   - Syring & Martin (2019) algorithm

2. **CATE Infrastructure**
   - Function-valued estimands
   - Projection to finite-dimensional summaries
   - Conditional inference by covariate strata

3. **Meta-Learner Ensemble**
   - Combine multiple identifiers with weights
   - Ensemble loss function
   - Cross-validation for weight selection

4. **Additional Identifiers**
   - R-learner (robust, optimal coverage)
   - DML-IPW (IPW with debiasing)
   - Custom identifier interface

### Success Criteria
- Bootstrap calibration test passes
- CATE estimation on synthetic data
- Ensemble shows improved performance
- 20+ tests total

---

## Phase 3: Examples & Documentation (Days 5)

### Objectives
- Runnable examples showcasing real use cases
- Tutorial-grade documentation
- Synthesis parameter sensitivity analysis
- Quantum circuit causal analysis

### Tasks
1. **Examples** (`examples/`)
   - `ate_basic.rs` — Simple ATE estimation
   - `cate_heterogeneous.rs` — Heterogeneous effects
   - `synthesis_parameters.rs` — Quiver audio parameter sensitivity
   - `quantum_importance.rs` — Gate importance in quantum circuits

2. **mdbook Documentation**
   - Expand introduction.md with problem motivation
   - Write theory chapter (orthogonality, cross-fitting, bootstrap)
   - Add advanced topics (ensemble, CATE, custom identifiers)
   - FAQ + troubleshooting

3. **Tutorial Examples**
   - Jupyter-style walkthrough (in markdown)
   - Step-by-step ATE estimation
   - Interpreting credible intervals
   - Choosing identifiers + priors

### Success Criteria
- 4+ runnable examples
- make doc-serve builds cleanly
- All examples compile and run
- Tutorial covers 80% of use cases

---

## Phase 4: Optimization & Release (Day 6)

### Objectives
- Performance optimization
- Feature flags for optional functionality
- Feature parity with fugue/fugue-evo
- Publish v1.0.0

### Tasks
1. **Performance**
   - Parallel cross-fitting with rayon (optional feature)
   - Vectorized loss computation where possible
   - Benchmarks for varying data sizes

2. **Feature Flags**
   - `parallel` — Use rayon for cross-fitting
   - `checkpoint` — Save/resume inference state
   - `std` — Standard library (for no_std safety)

3. **Polish**
   - API stability review (no breaking changes after 1.0)
   - Edge case hardening (overlaps, separation, singular matrices)
   - Error messages + diagnostics
   - License + CONTRIBUTING.md

4. **Release**
   - Tag v1.0.0
   - Update Cargo.toml version
   - Publish to crates.io
   - GitHub release notes

### Success Criteria
- All tests pass under all feature combinations
- Benchmarks show reasonable performance
- Documentation complete
- v1.0.0 published and indexed

---

## Module Implementation Checklist

### Phase 1 (PRIORITY)
- [ ] `src/cross_fit.rs` — K-fold cross-fitting
- [ ] `src/posterior.rs` — Gibbs posterior + real inference
- [ ] Integration tests — End-to-end correctness
- [ ] Fix: `fn infer_causal()` to actually return a real posterior

### Phase 2
- [ ] `src/bootstrap.rs` — ω calibration
- [ ] `src/estimand.rs` — Extend with CATE support
- [ ] `src/identifier.rs` — Add R-learner, DML variants
- [ ] Property-based tests (proptest)

### Phase 3
- [ ] `examples/*.rs` — 4+ runnable examples
- [ ] `docs/src/*.md` — Theory + tutorials
- [ ] CLI tools (optional, like fugue-evo might have)

### Phase 4
- [ ] Cargo.toml features
- [ ] Benchmarks (`benches/`)
- [ ] CONTRIBUTING.md
- [ ] Release checklist

---

## Known Technical Debt

Current state (v0.1.0-pre):
- `infer_causal()` returns dummy posterior (TODO)
- No cross-fitting implementation
- No bootstrap calibration
- Estimands are enum only (no extensibility)
- No parallel processing

All addressed by end of Phase 2.

---

## Success Metrics for v1.0

- ✓ Specification matches implementation (SPEC.md = reality)
- ✓ All theoretical guarantees are validated (Theorem 5.1, convergence)
- ✓ 30+ unit tests + 10+ integration tests
- ✓ 100% public API documented
- ✓ 4+ complete examples with explanations
- ✓ Performance benchmarked and reasonable
- ✓ Feature parity with fugue-evo in structure/quality
- ✓ No panics, all errors handled gracefully
- ✓ GitHub CI all green (check, fmt, clippy, test, doc)
- ✓ Published to crates.io with badge

---

## Ralph Loops Integration

This roadmap follows the Ralph Loops 3-phase methodology:

1. **Phase 1: Requirements** → V1_0_ROADMAP.md (this file) + SPEC.md
2. **Phase 2: Planning** → Task breakdown above + prioritized checklist
3. **Phase 3: Building** → Iterative implementation with fresh context per iteration

Each Phase 1 task becomes a Claude Code build session with:
- Fresh context window
- Focused scope (one module/feature)
- Tests + docs included
- Integration back to main

Expected: 6 focused, tight iterations → v1.0 ready.
