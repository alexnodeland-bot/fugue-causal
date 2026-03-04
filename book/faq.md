# FAQ: Frequently Asked Questions

## Conceptual

### Q: Why "Generalized Bayes" instead of standard Bayesian causal inference?

**A**: Standard Bayes requires specifying the full data-generating process P(X,A,Y|ξ), placing priors on nuisances, and marginalizing. This is:
- Indirect (priors on propensity scores are hard to elicit)
- Fragile (vulnerable to model misspecification)
- Computationally expensive

Generalized Bayes places the prior **directly on causal effects** θ and updates via loss functions that encode identification. This:
- ✅ Avoids full generative models
- ✅ Decouples causal parameters from nuisances
- ✅ Provides formal robustness guarantees (Theorem 5.1)

See [Theory](theory.md) for details.

### Q: What does "orthogonal" mean?

**A**: A loss function ℓ is **Neyman-orthogonal** if the first-order interaction between θ and η vanishes:

```
D_η D_θ E[ℓ(θ*, η₀)] = 0
```

**Why it matters**: Nuisance estimation errors don't blow up the posterior.
- Non-orthogonal (RA, IPW alone): Error propagates as O_P(√n · r_n)
- Orthogonal (DR/AIPW, R-learner): Error propagates as O_P(√n · r_n²)

For slow rates r_n = o(n^{-1/4}), orthogonality saves you from needing super-efficient nuisance estimation.

### Q: What's the difference between ATE, CATE, and HTE?

**A**:
- **ATE** (Average TE): Single number E[Y(1) - Y(0)]
  - Use: Overall policy evaluation
- **CATE** (Conditional ATE): Effect stratified by covariate E[Y(1)-Y(0)|X ∈ strata]
  - Use: Targeted analysis, group comparisons
- **HTE** (Heterogeneous TE): Function θ(x) that varies with x
  - Use: Individual-level effects, policy optimization

In fugue-causal:
- **ATE**: default with `prior_ate()` on single number
- **CATE**: stratify data, run `infer_causal` on each stratum
- **HTE**: use R-Learner with full data (estimates function via residualization)

---

## Technical

### Q: How do I know which identifier to use?

**A**: Choose based on your assumptions:

| Identifier | Assumption | Robustness | When to Use |
|-----------|-----------|-----------|-----------|
| **RA** | Outcomes well-modeled | Lower | Few confounders, strong overlap |
| **IPW** | Propensity well-modeled | Lower | Selection is clear, propensity simple |
| **DR/AIPW** | Either outcome OR propensity | Higher | **Standard choice** (if unsure, use this) |
| **R-Learner** | Either outcome OR propensity | Higher | Heterogeneous effects, modern ML |

**Rule of thumb**: Start with **DoublyRobust** (orthogonal, robust, industry standard).

### Q: What's cross-fitting and why is it mandatory?

**A**: Cross-fitting ensures **orthogonality in finite samples**.

**Without cross-fitting** (same data for fitting + evaluation):
- Empirical process bias doesn't vanish
- Orthogonality is violated
- Convergence guarantees fail

**With K-fold cross-fitting**:
1. Split data: D = D_train ∪ D_valid
2. Fit nuisances on D_train, evaluate loss on D_valid
3. Pool K-fold results
4. Validation losses are i.i.d. conditional on training → orthogonality preserved ✓

**In fugue-causal**: Always enabled by default via `num_folds` parameter.

### Q: What's ω (omega) and why calibrate it?

**A**: The Gibbs posterior q(θ | D) ∝ exp{−ωn·L_n(θ)} · π(θ) has "temperature" ω.

- ω controls posterior spread: variance ∝ 1/ω
- **Uncalibrated ω = 1**: Posterior may have invalid coverage
- **Calibrated ω**: Credible intervals achieve target frequentist coverage

**Calibration algorithm**: Bootstrap resampling tunes ω so P(θ* ∈ CI) ≈ (1−α).

**In fugue-causal**:
```rust
use fugue_causal::calibrate_omega;

let calib = calibrate_omega(&loss_fn, mean, var, &data, 0.95, 100)?;
println!("Calibrated ω: {}", calib.omega);
```

### Q: Do I need a lot of data?

**A**: Depends on identifier + nuisance complexity:

| Scenario | Minimum n |
|----------|-----------|
| ATE, 1-2 confounders, DR | 200-500 |
| CATE, 5-10 strata, DR | 500-2000 |
| HTE, flexible nuisance models | 2000+ |

**General rule**: Need enough data for:
1. Overlap in treatment/control across covariates
2. Nuisance estimator to converge
3. K-fold cross-fitting (typical: 50-100 obs per fold)

### Q: How do I pick K (number of folds)?

**A**:
- **K=5**: Default, balanced between precision and computation
- **K=10**: Higher precision, more computation
- **K=n**: Leave-one-out, most precise but expensive for large n

**Guideline**: Use K such that each fold has 50-200 observations.

---

## Practical

### Q: I have a continuous treatment (not binary). Can I use this?

**A**: Not yet. Fugue-causal assumes **binary treatment** A ∈ {0, 1}.

**For continuous treatments**, you'd need:
- Different loss functions (e.g., for average dose-response)
- Generalized propensity models
- More assumptions

Future versions may support this. For now, **bin your treatment** or use alternative tools (e.g., econometric methods for continuous treatments).

### Q: Can I use this with my ML model (XGBoost, neural nets)?

**A**: Yes! Implement the `NuisanceEstimator` trait:

```rust
use fugue_causal::NuisanceEstimator;

struct MyEstimator {
    // Your XGBoost model, neural net, etc.
}

impl NuisanceEstimator for MyEstimator {
    fn estimate_fold(
        &self,
        training: &[(Vec<f64>, f64, f64)],
        validation: &[(Vec<f64>, f64, f64)],
    ) -> Result<Vec<Vec<f64>>> {
        // Fit on training fold
        // Return nuisance estimates for validation fold
        // For DoublyRobust: [propensity, outcome_1, outcome_0]
    }
}

let posterior = infer_causal(
    prior_ate(),
    DoublyRobust,
    Box::new(MyEstimator::new()),
    5,
    &observations,
)?;
```

### Q: My credible intervals are huge. Why?

**A**: Large posterior SD can indicate:

1. **High outcome variance**: Noisy outcome Y → larger uncertainty (expected)
2. **Poor overlap**: Propensity scores near 0/1 → high IPW variance
3. **Weak treatment signal**: Small true effect → hard to detect
4. **Insufficient data**: Small n → higher uncertainty

**Diagnostics**:
```rust
// 1. Check propensity overlap
// Are both A=0 and A=1 present in all X regions?

// 2. Check outcome variance
let outcome_var = observations
    .iter()
    .map(|o| o[2])  // Y
    .fold((0.0, 0), |acc, y| (acc.0 + y, acc.1 + 1));

// 3. Check effect size
// True effect must be large enough relative to noise
```

### Q: Can I combine multiple identifiers?

**A**: Not directly, but you can:

1. **Run all identifiers separately**, compare results
2. **Ensemble**: Average posteriors (requires weights)
3. **Robustness check**: Do results agree across identifiers?

If DR/AIPW and R-learner agree → robust to nuisance misspecification ✓

### Q: How do I validate my results?

**A**: Use:

1. **Simulation**: Generate synthetic data with known true effect, check coverage
2. **Sensitivity analysis**: Vary key assumptions, re-estimate
3. **Identifier comparison**: RA vs IPW vs DR vs R-learner
4. **Subgroup analysis**: Effect stable across subsamples?
5. **Placebo tests**: Run on dummy outcomes (should be ~0)

Example:
```rust
// Placebo test: true effect should be 0
let placebo_observations: Vec<Vec<f64>> = observations
    .iter()
    .map(|o| vec![o[0], o[1], generate_random_outcome()]) // Replace Y
    .collect();

let placebo_posterior = infer_causal(..., &placebo_observations)?;
println!("Placebo effect: {:.4}", placebo_posterior.point_estimate);
// Should be close to 0
```

---

## Troubleshooting

### Q: I get "Error: Singular matrix" when running inference

**A**: Likely cause: **Perfect overlap violation** or **collinearity** in covariates.

**Fix**:
1. Check for constant columns (covariates that don't vary)
2. Remove highly correlated features
3. Ensure both A=0 and A=1 exist in your data
4. Try a regularized nuisance estimator (ridge regression, boosting)

### Q: "NuisanceEstimator returned wrong dimensions"

**A**: Check that your estimator returns the right number of nuisance estimates:

| Identifier | Nuisances | Count | Order |
|-----------|-----------|-------|-------|
| RA | m_0, m_1 | 2 | [outcome_0, outcome_1] |
| IPW | e | 1 | [propensity] |
| DR/AIPW | e, m_0, m_1 | 3 | [propensity, outcome_0, outcome_1] |
| R-Learner | e, m | 2 | [propensity, outcome] |

### Q: Posterior SD is NaN or Inf

**A**: Likely **Hessian computation failed** (loss not smooth at optimum).

**Fix**:
1. Add small regularization to nuisances (L2 penalty)
2. Use bounded outcome Y ∈ [a, b]
3. Ensure propensity scores are bounded away from 0/1 (truncate: e ∈ [0.05, 0.95])

---

## Integration

### Q: How does this work with fugue (the PPL)?

**A**: Fugue traces ARE estimand samplers. Effect handlers condition traces on loss functions.

**Planned integration**: `fugue` traces + `fugue-causal` losses → automatic causal inference in probabilistic programs.

### Q: How do I use this with quiver (synthesis) or QCSim (quantum)?

**A**: Treat parameters as treatments, output as outcome:

```rust
// Synthesis example: Does filter cutoff affect perceived brightness?
// X = base filter setting, A = adjusted cutoff (low/high)
// Y = listener preference score

// Quantum example: Does gate fidelity affect circuit success?
// X = circuit depth, A = perfect vs noisy gate
// Y = success probability
```

See [Examples](examples.md) for synthesis and quantum use cases.

---

## Contributing

### Q: I found a bug / have a feature request

**A**: Open an issue or PR on [GitHub](https://github.com/alexnodeland-bot/fugue-causal):
- **Bug reports**: Include minimal reproducible example
- **Feature requests**: Describe use case and desired API
- **Papers/methods**: Link relevant citations

### Q: Can I extend this library?

**A**: Yes! Key extension points:
1. New identifiers: Implement `CausalIdentifier` trait
2. New estimands: Add variants to `Estimand` enum
3. Advanced inference: Custom posterior samplers
4. Integration: Wrap external ML libraries as `NuisanceEstimator`

See module docs for trait requirements.

---

## References

- [Theory](theory.md): Deep dive on orthogonality, convergence
- [Tutorials](tutorials.md): Step-by-step guides
- [Examples](examples.md): Working code
- Paper: [ArXiv:2603.03035](https://arxiv.org/abs/2603.03035)
