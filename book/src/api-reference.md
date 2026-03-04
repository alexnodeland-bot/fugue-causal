# API Reference

## Core Functions

### `infer_causal`

```rust
pub fn infer_causal<T: CausalIdentifier>(
    prior: Box<dyn Fn(f64) -> f64>,
    identifier: T,
    nuisance_estimator: Box<dyn NuisanceEstimator>,
    folds: usize,
    observations: &[Vec<f64>],
) -> Result<CausalPosterior>
```

**Description**: Perform Bayesian causal inference via generalized Bayes.

**Arguments**:
- `prior`: Prior distribution on estimand (log-density function)
- `identifier`: Causal identification strategy (RA, IPW, DR, R-learner)
- `nuisance_estimator`: Estimator for confounders/propensity
- `folds`: K for K-fold cross-fitting (typical: 5-10)
- `observations`: Data as [covariate, treatment, outcome] per row

**Returns**: `CausalPosterior` struct with point estimate, posterior SD, etc.

**Errors**: `Result::Err` if:
- Insufficient observations
- Singular matrices (overlap violation)
- NuisanceEstimator fails

---

## Priors

### `prior_ate`

```rust
pub fn prior_ate() -> Box<dyn Fn(f64) -> f64>
```

**Description**: Standard normal prior N(0, 1) on ATE.

**Returns**: Closure computing log-density log π(θ).

### `prior_ate_informed`

```rust
pub fn prior_ate_informed(mean: f64, variance: f64) -> Box<dyn Fn(f64) -> f64>
```

**Description**: Domain-informed prior N(mean, variance) on ATE.

**Arguments**:
- `mean`: Prior expectation of causal effect
- `variance`: Prior uncertainty

**Returns**: Closure computing log π(θ) for N(mean, variance).

---

## Identifiers

### `DoublyRobust`

```rust
pub struct DoublyRobust;

impl CausalIdentifier for DoublyRobust {
    fn loss(&self, observation: &[f64], estimand_value: f64, nuisances: &[f64]) -> f64
    fn orthogonality(&self) -> Orthogonality { Orthogonality::Neyman }
    fn nuisance_rate_requirement(&self) -> NuisanceRate { NuisanceRate::Slow }
    fn name(&self) -> &'static str { "Doubly Robust (AIPW)" }
}
```

**Nuisances required**: [propensity, outcome_1, outcome_0]

**Orthogonality**: Neyman-orthogonal ✓

**Robustness**: Works if propensity OR outcomes correct

**Use**: General observational studies (recommended)

### `RLearner`

```rust
pub struct RLearner;

impl CausalIdentifier for RLearner {
    fn loss(&self, observation: &[f64], estimand_value: f64, nuisances: &[f64]) -> f64
    fn orthogonality(&self) -> Orthogonality { Orthogonality::Neyman }
    fn nuisance_rate_requirement(&self) -> NuisanceRate { NuisanceRate::Slow }
    fn name(&self) -> &'static str { "R-Learner (Orthogonal ML)" }
}
```

**Nuisances required**: [propensity, outcome]

**Orthogonality**: Always Neyman-orthogonal ✓

**Robustness**: Works if propensity OR outcome correct

**Use**: Heterogeneous treatment effects, modern ML pipelines

### `InverseProbabilityWeighting`

```rust
pub struct InverseProbabilityWeighting;

impl CausalIdentifier for InverseProbabilityWeighting {
    fn orthogonality(&self) -> Orthogonality { Orthogonality::None }
    fn nuisance_rate_requirement(&self) -> NuisanceRate { NuisanceRate::Parametric }
}
```

**Nuisances required**: [propensity]

**Orthogonality**: Non-orthogonal ✗

**Constraint**: Requires r_n = o(n^{-1/2})

**Use**: Clear selection mechanism, strong propensity model

### `RegressionAdjustment`

```rust
pub struct RegressionAdjustment;

impl CausalIdentifier for RegressionAdjustment {
    fn orthogonality(&self) -> Orthogonality { Orthogonality::None }
    fn nuisance_rate_requirement(&self) -> NuisanceRate { NuisanceRate::Parametric }
}
```

**Nuisances required**: [outcome_1, outcome_0]

**Orthogonality**: Non-orthogonal ✗

**Constraint**: Requires r_n = o(n^{-1/2})

**Use**: Well-specified outcome models, parametric approach

---

## Posterior & Results

### `CausalPosterior`

```rust
pub struct CausalPosterior {
    pub point_estimate: f64,
    pub posterior_sd: f64,
    pub omega: f64,
    pub calibration_method: String,
}
```

**Fields**:
- `point_estimate`: MAP/posterior mean θ̂
- `posterior_sd`: Standard deviation of posterior
- `omega`: Temperature parameter (inverse precision)
- `calibration_method`: "bootstrap" or "default"

**Usage**:
```rust
let ci_lower = posterior.point_estimate - 1.96 * posterior.posterior_sd;
let ci_upper = posterior.point_estimate + 1.96 * posterior.posterior_sd;
println!("95% CI: [{:.4}, {:.4}]", ci_lower, ci_upper);
```

---

## Bootstrap Calibration

### `calibrate_omega`

```rust
pub fn calibrate_omega(
    loss_fn: &dyn Fn(&[f64]) -> f64,
    posterior_mean: f64,
    posterior_variance: f64,
    data: &[Vec<f64>],
    target_coverage: f64,
    n_replicates: usize,
) -> OmegaCalibration
```

**Description**: Calibrate ω for frequentist coverage via bootstrap.

**Arguments**:
- `loss_fn`: Per-observation loss function
- `posterior_mean`: Initial θ̂
- `posterior_variance`: Initial variance estimate
- `data`: Observations
- `target_coverage`: Target coverage (e.g., 0.95)
- `n_replicates`: Bootstrap replicates (50-200 typical)

**Returns**: `OmegaCalibration` with calibrated ω and empirical coverage

### `OmegaCalibration`

```rust
pub struct OmegaCalibration {
    pub omega: f64,
    pub empirical_coverage: f64,
    pub num_replicates: usize,
    pub target_coverage: f64,
}
```

---

## Cross-Fitting

### `cross_fit`

```rust
pub fn cross_fit<F, T>(
    folds: usize,
    observations: &[Vec<f64>],
    estimator: F,
) -> Result<CrossFittedNuisances>
where
    F: Fn(&[Vec<f64>], &[Vec<f64>]) -> Result<Vec<Vec<f64>>>,
    T: Clone,
```

**Description**: Perform K-fold cross-fitting for nuisance estimation.

**Arguments**:
- `folds`: K for K-fold splitting
- `observations`: Full dataset
- `estimator`: Closure fitting on train, returning estimates for validation

**Returns**: `CrossFittedNuisances` with estimates for each observation

---

## Traits

### `CausalIdentifier`

```rust
pub trait CausalIdentifier: Send + Sync {
    fn loss(&self, observation: &[f64], estimand_value: f64, nuisances: &[f64]) -> f64;
    fn nuisance_names(&self) -> Vec<&'static str>;
    fn orthogonality(&self) -> Orthogonality;
    fn nuisance_rate_requirement(&self) -> NuisanceRate;
    fn name(&self) -> &'static str;
}
```

**Implement this** to add new identifiers.

### `NuisanceEstimator`

```rust
pub trait NuisanceEstimator: Send {
    fn estimate_fold(
        &self,
        training: &[(Vec<f64>, f64, f64)],
        validation: &[(Vec<f64>, f64, f64)],
    ) -> Result<Vec<Vec<f64>>>;
}
```

**Implement this** to integrate custom ML models (XGBoost, neural nets, etc.).

---

## Enums

### `Orthogonality`

```rust
pub enum Orthogonality {
    Neyman,    // First-order interaction vanishes
    Partial,   // Some orthogonality
    None,      // Non-orthogonal
}
```

**Convergence**:
- Neyman: O_P(√n r_n²)
- Partial: O_P(√n r_n r_n')
- None: O_P(√n r_n)

### `NuisanceRate`

```rust
pub enum NuisanceRate {
    Parametric,  // n^{-1/2}
    Slow,        // o(n^{-1/4})
    Fast,        // o(n^{-1/2})
}
```

---

## Estimands

### `Estimand`

```rust
pub enum Estimand {
    ATE(TreatmentSpec),
    CATE { treatment: TreatmentSpec, conditioning_vars: Vec<String> },
    ATT(TreatmentSpec),
    ATU(TreatmentSpec),
    HTE(TreatmentSpec),
    Custom(String),
}
```

**Future support**: Currently focused on ATE; CATE/HTE support expanding.

---

## Built-in Estimators

### `PluginEstimator`

```rust
pub struct PluginEstimator;

impl NuisanceEstimator for PluginEstimator {
    fn estimate_fold(&self, training: &[...], validation: &[...]) -> Result<...>
}
```

**Description**: Simple linear/logistic regression plugin (for demonstration).

**Use**: Testing, quick prototypes. Replace with ML model in production.

---

## Helper Functions

### `normal_quantile`

```rust
pub fn normal_quantile(p: f64) -> f64
```

**Description**: Compute Φ^{-1}(p) for standard normal.

**Returns**: z-value such that P(Z ≤ z) = p.

**Examples**:
```rust
normal_quantile(0.025)   // -1.96
normal_quantile(0.5)     // 0.0
normal_quantile(0.975)   // 1.96
```

---

## Error Handling

All functions return `Result<T>` where errors include:

- **InsufficientObservations**: n < min required
- **IdentificationFailed**: Overlap or singularity issue
- **NuisanceEstimationError**: Custom estimator failure
- **PosteriorComputationError**: Optimization or numerical failure

**Pattern**:
```rust
match infer_causal(...) {
    Ok(posterior) => { /* use posterior */ }
    Err(e) => eprintln!("Inference failed: {}", e),
}
```

---

## Complete Example

```rust
use fugue_causal::*;

fn main() -> Result<()> {
    // Data
    let obs = vec![vec![0.5, 1.0, 3.2], /* ... */];

    // Inference
    let posterior = infer_causal(
        prior_ate(),
        DoublyRobust,
        Box::new(PluginEstimator),
        5,
        &obs,
    )?;

    // Results
    let z = normal_quantile(0.975);
    let ci = (
        posterior.point_estimate - z * posterior.posterior_sd,
        posterior.point_estimate + z * posterior.posterior_sd,
    );

    println!("ATE: {:.4} [{:.4}, {:.4}]", posterior.point_estimate, ci.0, ci.1);
    Ok(())
}
```
