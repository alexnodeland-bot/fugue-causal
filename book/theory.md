# Theory: Bayesian Causal Inference via Generalized Bayes

## Generalized Bayes

### Standard Bayesian Inference (Issues)

In standard Bayesian inference on causal effects, we:
1. Specify full data-generating model: P(X, A, Y | ξ) where ξ includes nuisances
2. Place prior π(ξ)
3. Compute posterior P(ξ | D) ∝ P(D | ξ)π(ξ)
4. Marginalize: P(θ | D) = ∫ P(θ | ξ) P(ξ | D) dξ

**Problems:**
- Model misspecification
- Indirect prior elicitation
- Coupling between causal parameter and nuisance priors

### Generalized Bayes (Clean Solution)

Instead, we:
1. **Place prior directly on causal estimand**: π(θ)
2. **Define loss via identification**: L_n(θ) = Σᵢ ℓ(Oᵢ; θ, η̂)
3. **Form Gibbs posterior**: q(θ | D) ∝ exp{−ωn·L_n(θ)} · π(θ)

**Advantages:**
- ✅ Avoids full generative model
- ✅ Loss functions encode identification strategy
- ✅ Nuisance estimation is separated (via cross-fitting)
- ✅ Formal robustness guarantees

### Gibbs Posterior Properties

The posterior q(θ | D) ∝ exp{−ωn·L_n(θ)} · π(θ):
- **Shape**: Proportional to exp(−ωn·L_n)
- **Temperature ω**: Controls posterior spread (inverse precision)
  - ω → ∞: Posterior → point mass at θ̂
  - ω → 0: Posterior → prior π(θ)
  - ω = 1: Balanced (default)
- **Calibration**: Bootstrap tunes ω for frequentist coverage

---

## Orthogonality & Nuisance Robustness

### The Neyman Orthogonality Condition

A loss function ℓ^S(O; θ, η) is **Neyman-orthogonal** at (θ*, η₀) if:

```
D_η D_θ E[ℓ^S(O; θ*, η)] |_{θ*, η_0} = 0
```

**In words**: The first-order interaction between the causal parameter θ and nuisance η vanishes.

### Why Orthogonality Matters

**Theorem 5.1 (Javurek et al. 2026)**

If loss ℓ is Neyman-orthogonal at (θ*, η₀) + cross-fitting:
```
TV(q_{n,fe}^S, q_{n,or}^S) = O_P(√n · r_n²)
```

where:
- q_{n,or} = oracle posterior (true nuisances)
- q_{n,fe} = feasible posterior (estimated nuisances)
- r_n = nuisance estimation error (e.g., o(n^{-1/4}) for slow rate)

**Key insight**: Error propagates as r_n², not r_n!

### Convergence Rates

| Orthogonality | Loss Function | Error Rate |
|---------------|---------------|-----------|
| Neyman | RA, IPW mixed → **DR/AIPW** | r_n² (slow) |
| Partial | Some terms orthogonal | r_n · r_n' (mixed) |
| None | **RA, IPW alone** | r_n (fast) |

**For DR/AIPW**: If r_n = o(n^{-1/4}), then error = O_P(n^{-1/2})

---

## Cross-Fitting

### The Problem Without Cross-Fitting

If we fit nuisances and evaluate loss on the same data:
- Empirical process bias terms don't vanish
- Orthogonality is violated in finite samples
- Convergence guarantees fail

### The Solution: K-Fold Cross-Fitting

**Algorithm**:
1. Partition data: D = ∪ₖ D_train^(k) ∪ D_valid^(k)
2. For each fold k:
   - Fit nuisances η̂^(k) on D_train^(k)
   - Evaluate loss ℓ_n^(k) on D_valid^(k)
3. Pool losses: L_n = (1/K) Σₖ L_n^(k)

**Key property**: Validation losses are i.i.d. conditional on training samples → orthogonality preserved

### Implementation

```rust
use fugue_causal::cross_fit;

let nuisance_fits = cross_fit(
    5,                    // K folds
    &observations,
    |train_fold, val_fold| {
        // Fit on train_fold
        // Return nuisance estimates for validation
        nuisance_estimator.estimate_fold(train_fold, val_fold)
    }
)?;
```

---

## Identifiers

### Regression Adjustment (RA)

**Loss**: (m̂₁(X) - m̂₀(X) - θ)²

**Nuisances**: Outcome models m₀, m₁

**Orthogonality**: ❌ None
- Requires r_n = o(n^{-1/2})
- Less robust to misspecification

**Use case**: Observational studies with strong overlap, well-specified outcomes

### Inverse Probability Weighting (IPW)

**Loss**: (A·Y/ê - (1-A)·Y/(1-ê) - θ)²

**Nuisances**: Propensity score ê(X) = P(A=1|X)

**Orthogonality**: ❌ None
- Requires r_n = o(n^{-1/2})
- High variance if propensity near 0/1

**Use case**: Strong selection on observables, weak overlap tolerance

### Doubly Robust / AIPW

**Loss**:
```
((A/ê - (1-A)/(1-ê)) · (Y - m_A) + m₁ - m₀ - θ)²
```

**Nuisances**: Propensity ê, outcomes m₀, m₁

**Orthogonality**: ✅ Neyman-orthogonal
- Requires r_n = o(n^{-1/4})
- "Doubly robust": works if ê correct OR outcomes correct
- Lower variance than IPW

**Use case**: Standard choice for observational causal inference

### R-Learner

**Loss**:
```
(Ã · θ - Ỹ)² where Ã = A - ê, Ỹ = Y - m
```

**Nuisances**: Propensity ê, outcome model m

**Orthogonality**: ✅ Always Neyman-orthogonal
- Requires r_n = o(n^{-1/4})
- Simple residualization
- Interpretable

**Use case**: Heterogeneous treatment effects (HTE), modern ML pipelines

---

## Causal Estimands

### ATE (Average Treatment Effect)

```
τ = E[Y(1) - Y(0)]
```

**Interpretation**: Average effect if entire population treated vs untreated

**Prior**: N(0, 1) (standard) or domain-informed

### CATE (Conditional ATE)

```
τ(x) = E[Y(1) - Y(0) | X = x]
```

**Interpretation**: Effect conditional on covariate value

**Use case**: Stratified analysis, detecting heterogeneity

### ATT / ATU

```
ATT = E[Y(1) - Y(0) | A = 1]
ATU = E[Y(1) - Y(0) | A = 0]
```

**Use case**: Policy targeting (always-takers vs never-takers)

### HTE (Heterogeneous Treatment Effects)

Function-valued estimand: θ(x) varies with covariates

**Identification**: Requires overlap + stronger assumptions

**Tools**: Causal forests, X-learners, R-learners

---

## Frequentist Coverage & Bootstrap Calibration

### Credible Intervals from Gibbs Posteriors

A (1−α) credible interval [θ̂ − z_{α/2} · √V̂/ω, θ̂ + z_{α/2} · √V̂/ω] has coverage:
```
P(θ* ∈ CI) ≈ (1 − α) · f(ω, r_n, ...)
```

Without ω calibration, coverage may be invalid.

### Bootstrap Calibration (Syring & Martin 2019)

**Algorithm**:
1. For each bootstrap replicate b = 1, ..., B:
   - Resample with replacement
   - Fit nuisances on bootstrap sample
   - Compute loss on full data
2. Find ω* such that:
   ```
   P(θ* ∈ CI(ω*)) ≈ (1 − α) over replicates
   ```

**Result**: Posterior intervals achieve target coverage

### Implementation

```rust
use fugue_causal::calibrate_omega;

let calibration = calibrate_omega(
    &loss_fn,
    posterior_mean,
    posterior_variance,
    &data,
    0.95,         // 95% target coverage
    100,          // bootstrap replicates
)?;

println!("Calibrated ω: {}", calibration.omega);
println!("Empirical coverage: {}", calibration.empirical_coverage);
```

---

## References

1. **Javurek, E.**, et al. (2026). "Generalized Bayes for Causal Inference." ArXiv:2603.03035v1.
2. **Chernozhukov, V.**, Newey, W. K., & Robins, J. M. (2018). "Double/Debiased Machine Learning for Treatment and Causal Parameters." *The Econometrics Journal*, 21(1).
3. **Syring, N.**, & Martin, R. (2019). "Calibrating General Posterior Credible Regions." *Bayesian Analysis*, 14(3).
4. **Kennedy, E. H.** (2020). "Optimal Doubly Robust Estimation of Heterogeneous Treatment Effects." *arXiv*.
