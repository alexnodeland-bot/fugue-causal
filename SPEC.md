# fugue-causal: Bayesian Causal Inference via Generalized Bayes

**Status:** Specification Phase  
**Source Paper:** [Generalized Bayes for Causal Inference](https://arxiv.org/abs/2603.03035) (Javurek et al., 2603.03035v1, submitted March 3, 2026)  
**Integration:** Core extension to the [fugue](https://github.com/alexnodeland/fugue) probabilistic programming library

---

## 1. Problem Statement

Standard Bayesian causal inference is brittle:

1. **Full model specification required**: Must parameterize P(X, A, Y | ξ_x, ξ_a, ξ_y)—the entire data-generating process.
2. **High-dimensional nuisance priors**: Priors on propensity scores e(x) and outcome regressions m(x) are indirect, hard to elicit, and coupled.
3. **Feedback coupling**: Joint likelihood couples nuisance posteriors; outcome information feeds back into propensity estimation, breaking robustness.
4. **Indirect prior on causal estimands**: The mapping π(ξ) → π(θ) is non-transparent; prior elicitation on the causal effect θ itself is nearly impossible.

**Result:** Posterior is vulnerable to model misspecification and complex prior assumptions, even when the underlying causal identification is solid.

### The Generalized Bayes Solution

**Skip the likelihood.** Place priors directly on causal estimands θ ∈ Θ (e.g., ATE, CATE) and update via identification-driven loss functions instead of data likelihoods.

- **No data model**: No P(X, A, Y | ξ) required.
- **Direct estimand priors**: π(θ) speaks directly about treatment effects, not nuisances.
- **Composable loss-based identification**: Different identification strategies (RA, IPW, DR, R-learner, etc.) yield different losses; same framework applies to all.
- **Formal robustness**: Neyman-orthogonal losses (like DR/AIPW) provide second-order robustness to nuisance estimation error.

---

## 2. Core Architecture

### 2.1 Causal Identification Strategies

A **causal identification strategy** S is a tuple:
- **Nuisances**: η^S ∈ H^S (e.g., propensity score e(x), outcome regression m(x))
- **Per-observation loss**: ℓ^S(O; θ, η^S) such that E[ℓ^S(O; θ, η^S_0)] = ℓ(θ; η_0)
- **Identification property**: argmin_θ E[ℓ^S(O; θ, η_0)] = θ* (the true causal estimand)

#### Standard Strategies (Scalar ATE)

**Regression Adjustment (RA):**
```
Pseudo-outcome: Ŷ^RA(O; m̂) = m̂₁(X) − m̂₀(X)
Per-obs loss: ℓ^RA(O; θ, m̂) = (Ŷ^RA − θ)²
```
- Identifies ATE under unconfoundedness + overlap
- **Not orthogonal**: requires nuisance rate r_n = o(n^{-1/2})

**Inverse Probability Weighting (IPW):**
```
Pseudo-outcome: Ŷ^IPW(O; ê) = (AY)/ê(X) − ((1−A)Y)/(1−ê(X))
Per-obs loss: ℓ^IPW(O; θ, ê) = (Ŷ^IPW − θ)²
```
- Identifies ATE under unconfoundedness + overlap
- **Not orthogonal**: requires nuisance rate r_n = o(n^{-1/2})

**Doubly Robust / AIPW:**
```
Pseudo-outcome: Ŷ^DR(O; ê, m̂) = (A/ê(X) − (1−A)/(1−ê(X)))(Y − m̂_A(X)) + m̂₁(X) − m̂₀(X)
Per-obs loss: ℓ^DR(O; θ, ê, m̂) = (Ŷ^DR − θ)²
```
- Identifies ATE under unconfoundedness + overlap
- **Neyman-orthogonal**: requires only nuisance rate r_n = o(n^{-1/4})
- Optimal when both nuisances are equally important

#### Heterogeneous Treatment Effects (CATE)

For function-valued estimands θ(x) = E[Y(1) − Y(0) | X = x]:

```
Pseudo-outcome regression: ℓ^S(O; θ, η̂^S) = (Ŷ^S(O; η̂^S) − θ(X))²
```

Where Ŷ^S is the same pseudo-outcome as above; the regression of Ŷ^S on X recovers θ(x).

### 2.2 Neyman-Orthogonality

A loss ℓ^S is **Neyman-orthogonal** at (θ*, η^S_0) if:

```
D_η D_θ E[ℓ^S(O; θ, η)] |_{θ=θ*, η=η_0} = 0
```

**In words:** The first-order interaction between the causal parameter and nuisance vanishes at the truth. Small changes in η don't affect the gradient in θ.

**Consequence:** The leading bias term from nuisance estimation error cancels in the first-order optimality condition. Convergence rate improves from O_P(√n r_n) to O_P(√n r²_n).

**Examples:**
- DR/AIPW: ✓ Orthogonal
- RA: ✗ Non-orthogonal (requires accurate m_a)
- IPW: ✗ Non-orthogonal (requires accurate e)

### 2.3 The Gibbs Posterior

Given:
- **Prior**: π(θ) ∈ P(Θ)
- **Loss function**: L_n^S(θ; η̂^S) = (1/n) Σ_i ℓ^S(O_i; θ, η̂^S)
- **Calibration parameter**: ω > 0

The **Gibbs posterior** (generalized Bayesian posterior) is:

```
q_n^S(θ | D_n) ∝ exp{−ω n L_n^S(θ; η̂^S)} · π(θ)
```

**Variational form:**
```
q_n^S = argmin_{q ∈ P(Θ)} {ω n E_θ∼q[L_n^S(θ; η̂^S)] + KL(q ∥ π)}
```

**Properties:**
- Recovers standard Bayesian posterior when loss is negative log-likelihood
- Variance (credible interval width) ∝ ω^{−1}
- ω tuned via bootstrap calibration for frequentist coverage

### 2.4 Cross-Fitting (Essential)

To preserve orthogonality:

1. **Split data** {1, ..., n} into K folds {I_k}
2. **Fit nuisances** η̂^{S,(-k)} on {O_i : i ∉ I_k} for each k
3. **Form cross-fitted loss**:
   ```
   L_n^S(θ; η̂^S) = (1/n) Σ_{k=1}^K Σ_{i ∈ I_k} ℓ^S(O_i; θ, η̂^{S,(-k)})
   ```
4. **Construct posterior** q_n^S using cross-fitted loss

**Why:** Conditional on training folds, validation losses are i.i.d., enabling standard LLN/CLT and eliminating empirical-process bias that would violate orthogonality.

---

## 3. Key Theoretical Results

### Theorem 5.1: Posterior Stability Under Orthogonal Losses

**Setup:**
- Θ ⊂ ℝ^d (finite-dimensional estimand)
- θ* true estimand, η_0^S true nuisances
- V_0 := ∇²_θ E[ℓ^S(O; θ, η_0)] |_{θ=θ*} positive definite (Hessian)

**Assumptions:**
- **(A1) BvM:** Oracle posterior q_{n,or}^S(θ | D_n; η_0^S) ≈ N(θ̂_{or}, (nω V_0)^{−1})
- **(A2) Orthogonality:** D_η D_θ E[ℓ^S] |_{θ*, η_0} = 0
- **(A3) Cross-fitting + nuisance rate:** ‖η̂^S − η_0^S‖ = r_n → 0

**Conclusion:**
```
TV(q_{n,fe}^S, q_{n,or}^S) = O_P(√n · r_n²) + o_P(1)
```

If r_n = o(n^{−1/4}), then feasible posterior ≈ oracle posterior in total variation.

**Intuition:** Orthogonality front-loads nuisance robustness. The squared rate r²_n (not r_n) means you can use slower nuisance estimators and still get valid uncertainty quantification.

### Theorem A.2: Infinite-Dimensional Extension

For Θ a separable Hilbert space and T(θ) a fixed finite-dimensional projection (e.g., CATE at covariate points):

```
TV(q_{n,fe}^T, q_{n,or}^T) = O_P(√n · r_n²) + o_P(1)
```

Stability extends to infinite-dimensional objects through finite-dimensional projections.

### Calibration of ω (Frequentist Validity)

For (1−α) credible interval to have (1−α) frequentist coverage:

**Bootstrap calibration** (Syring & Martin, 2019):
1. Compute q_n(θ | D_n) with ω_0 (e.g., ω_0 = 1)
2. Draw bootstrap samples from empirical distribution
3. Compute coverage of (1−α) credible set on bootstrap samples
4. Adjust ω until coverage ≈ 1−α

Result: Uncertainty quantification is valid even when nuisance estimators are slow (e.g., converge at n^{−1/3} rates instead of n^{−1/2}).

---

## 4. Integration with Fugue

### 4.1 Trait Design

```rust
// Strategy identifier and orthogonality characterization
pub trait CausalIdentifier {
    /// Per-observation loss: ℓ^S(O; θ, η)
    fn loss(&self, observation: &Observation, estimand: &Estimand, nuisances: &Nuisances) -> f64;
    
    /// Nuisance components this strategy requires
    fn nuisance_schema(&self) -> NuisanceSchema;
    
    /// Orthogonality property
    fn orthogonality(&self) -> Orthogonality; // Neyman, Partial, None
    
    /// Convergence rate for nuisances
    fn nuisance_rate_requirement(&self) -> NuisanceRate; // n^-1/2, n^-1/4, etc.
}

// Causal problem definition
pub struct CausalProblem<T: CausalIdentifier> {
    pub estimand_prior: Distribution,
    pub identifier: T,
    pub nuisance_estimator: Box<dyn NuisanceEstimator>,
    pub folds: usize, // For cross-fitting
}

// Posterior inference
pub struct CausalPosterior {
    pub posterior: Distribution,
    pub omega: f64,
    pub calibration_method: CalibrationMethod,
}
```

### 4.2 Effect Handler Pattern

Fugue's traces are estimand samplers. Conditioning by loss is a natural effect:

```rust
// In fugue-causal skill
impl<T: CausalIdentifier> EffectHandler for CausalConditioner<T> {
    fn on_sample(&mut self, trace: &Trace, value: &Value) -> Weight {
        let estimand = extract_estimand(trace);
        let nuisances = self.nuisance_estimates.clone();
        let loss = self.identifier.loss(&self.observation, &estimand, &nuisances);
        Weight((-self.omega * loss).exp()) // Gibbs reweighting
    }
}
```

### 4.3 Composable Estimator Library

```rust
// Concrete implementations
struct RegressionAdjustment;
impl CausalIdentifier for RegressionAdjustment { ... }

struct InverseProbabilityWeighting;
impl CausalIdentifier for InverseProbabilityWeighting { ... }

struct DoublyRobust; // DR/AIPW
impl CausalIdentifier for DoublyRobust { ... }

struct RLearner;
impl CausalIdentifier for RLearner { ... }

struct MetaLearnerEnsemble {
    learners: Vec<Box<dyn CausalIdentifier>>,
}
impl CausalIdentifier for MetaLearnerEnsemble { ... }
```

### 4.4 Estimand Types

```rust
pub enum Estimand {
    ATE(TreatmentSpec),                    // Average Treatment Effect
    CATE {                                  // Conditional ATE
        treatment: TreatmentSpec,
        conditioning_vars: Vec<String>,
    },
    ATT,                                    // Average Treatment on Treated
    ATU,                                    // Average Treatment on Untreated
    HTE,                                    // Heterogeneous Treatment Effects (function)
    Custom(Box<dyn Fn(&Trace) -> f64>),    // User-defined
}
```

---

## 5. Unlocked Use Cases

### 5.1 Composable Causal Pipelines

```rust
let problem = CausalProblem {
    estimand_prior: prior_ate(),
    identifier: DoublyRobust,
    nuisance_estimator: Box::new(CausalForest::new()),
    folds: 5,
};

let posterior = infer_causal(problem, data)?;
// Get credible intervals with formal uncertainty quantification
```

Swap identifier (RA → DR → R-learner), swap nuisance estimator (random forest → XGBoost → neural net), swap prior (domain knowledge). Same framework, composable.

### 5.2 Causal-Evolutionary Search (with fugue-evo)

GA searches over causal inference strategies:
- **Genome** = (choice of identifier, prior specification, nuisance hyperparameters)
- **Fitness** = calibration quality on held-out validation data
- **Selection pressure** = Bayesian updating (via Gibbs posterior reweighting)

Result: Automatically find which causal inference method works best for your problem class.

### 5.3 Proof-Carrying Code

If you use orthogonal identifiers + cross-fitting + sufficient nuisance rates, you get formal convergence guarantees:

```rust
let problem = CausalProblem {
    identifier: DoublyRobust, // ✓ Orthogonal
    folds: 5,                 // ✓ Cross-fitted
    ..
};

// Compiler can verify: "This posterior is robust to O_P(√n r²_n) nuisance error"
let posterior = infer_causal(problem, data)?;
```

---

## 6. Implementation Roadmap

### Phase 1: Core Framework (Weeks 1–2)
- [ ] `CausalIdentifier` trait + schema
- [ ] Gibbs posterior construction with variational form
- [ ] DR/AIPW + RA + IPW implementations
- [ ] Cross-fitting orchestration
- [ ] Basic bootstrap calibration

### Phase 2: Integration (Weeks 3–4)
- [ ] Effect handler for loss-based conditioning
- [ ] Estimand enum + prior specifications
- [ ] Metadata tracking (orthogonality, nuisance rates)
- [ ] Documentation + examples

### Phase 3: Extensions (Weeks 5–6)
- [ ] Meta-learner ensemble (combine multiple identifiers)
- [ ] Heterogeneous effects (CATE as functional estimand)
- [ ] Quantum/audio domain adapters
- [ ] Causal discovery integration (optional: structure learning)

### Phase 4: Validation & Applications (Weeks 7+)
- [ ] Replicate paper's numerical experiments
- [ ] Pair with fugue-evo for strategy search
- [ ] Domain-specific applications in separate projects

---

## 7. References

**Paper:**
- Javurek, E., et al. (2026). "Generalized Bayes for Causal Inference." *arXiv:2603.03035v1*. https://arxiv.org/abs/2603.03035

**Foundational:**
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Chernozhukov, V., Newey, W. K., & Robins, J. (2018). "Double/debiased machine learning for treatment and structural parameters." *Econometric Reviews*, 38(4), 322–345.
- Kennedy, E. H. (2023). "Towards optimal doubly robust estimation of heterogeneous treatment effects." *Electronic Journal of Statistics*, 17(2), 3008–3049.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

**Probabilistic Programming:**
- Tran, D., et al. (2017). "Deep Probabilistic Programming." *ICLR 2017*. (Pyro inspiration)
- Gram-Hansen, B., et al. (2017). "Inference in probabilistic programs with unknown parameters." *AAAI Spring Symposium*.

---

## 8. Authors & Attribution

**Specification:** Alex Nodeland  
**Source Framework:** Javurek, Emil, et al. (ArXiv 2603.03035v1)  
**Integration Design:** Chief Desmond (OpenClaw)

---

*Last updated: 2026-03-04 | Status: Specification Phase*
