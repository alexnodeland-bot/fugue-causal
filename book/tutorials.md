# Tutorials: Step-by-Step Guides

## Basic ATE Inference

### Scenario
You have observational data and want to estimate the average causal effect of a treatment on an outcome, with valid uncertainty quantification.

### Data Format

Arrange your data as observations with [covariate, treatment, outcome]:

```rust
let observations: Vec<Vec<f64>> = vec![
    vec![x1, a1, y1],  // Person 1: covariate, treatment (0/1), outcome
    vec![x2, a2, y2],  // Person 2
    // ... more observations
];
```

- **Covariate X**: Confounder (e.g., age, prior knowledge)
- **Treatment A**: Binary (0 = control, 1 = treated)
- **Outcome Y**: Continuous (e.g., test score, recovery time)

### Step 1: Choose Prior

Place a prior directly on the causal effect:

```rust
use fugue_causal::prior_ate;

// Standard normal prior N(0, 1)
let prior = prior_ate();

// Or domain-informed prior N(μ, σ²)
let prior = fugue_causal::prior_ate_informed(
    5.0,    // Expected effect
    1.0,    // Uncertainty (variance)
);
```

### Step 2: Select Identifier

Choose a causal identification strategy:

```rust
use fugue_causal::{DoublyRobust, RLearner};

// For general observational studies: Doubly Robust (orthogonal, robust)
let identifier = DoublyRobust;

// For heterogeneous effects: R-Learner (always orthogonal)
let identifier = RLearner;
```

### Step 3: Specify Nuisance Estimator

Provide a strategy for estimating confounders (propensity scores, outcomes):

```rust
use fugue_causal::PluginEstimator;

// Simple linear regression plugin (for demonstration)
let nuisance_estimator = Box::new(PluginEstimator);

// In production, use:
// - CausalForest (for complex nonlinear relationships)
// - GradientBoosting (XGBoost, LightGBM)
// - NeuralNet (with proper regularization)
```

### Step 4: Run Inference

```rust
use fugue_causal::infer_causal;

let posterior = infer_causal(
    prior,
    identifier,
    nuisance_estimator,
    5,  // K-fold cross-fitting (default: 5-10)
    &observations,
)?;
```

### Step 5: Interpret Results

```rust
println!("Point Estimate: {:.4}", posterior.point_estimate);
println!("Posterior Std:  {:.4}", posterior.posterior_sd);

// 95% Credible Interval
let z_95 = 1.96;
let ci_lower = posterior.point_estimate - z_95 * posterior.posterior_sd;
let ci_upper = posterior.point_estimate + z_95 * posterior.posterior_sd;
println!("95% CI: [{:.4}, {:.4}]", ci_lower, ci_upper);

// Calibration info
println!("ω (temperature): {:.4}", posterior.omega);
println!("Method: {}", posterior.calibration_method);
```

### Complete Example

```rust
use fugue_causal::*;

fn main() {
    // Simulate data: X ~ N(0,1), A ~ Bernoulli(0.5), Y = X + 2*A + noise
    let n = 500;
    let observations: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let seed = (i as f64 * 12.9898).sin() * 43758.5453;
            let x = (seed.fract() - 0.5) * 2.0;
            let a = if seed.fract() > 0.5 { 1.0 } else { 0.0 };
            let y = x + 2.0 * a + (seed.sin() * 0.5);
            vec![x, a, y]
        })
        .collect();

    // Inference
    let posterior = infer_causal(
        prior_ate(),
        DoublyRobust,
        Box::new(PluginEstimator),
        5,
        &observations,
    ).expect("Inference failed");

    println!("ATE: {:.4}", posterior.point_estimate);
    println!("SE: {:.4}", posterior.posterior_sd);
}
```

---

## Heterogeneous Treatment Effects

### Scenario
Treatment effects vary across individuals. You want to estimate effects in different groups.

### Key Question
**Does the treatment effect depend on covariates?**

Example: Drug effectiveness varies by age, gender, or disease severity.

### Approach: Stratified Analysis

#### Option 1: Partition by Covariate

```rust
use fugue_causal::*;

// Divide data into age strata
let age_groups = vec![(20, 35), (35, 50), (50, 65)];
let mut results = Vec::new();

for (age_min, age_max) in age_groups {
    // Filter observations in age range
    let stratum: Vec<Vec<f64>> = observations
        .iter()
        .filter(|obs| obs[0] >= age_min as f64 && obs[0] < age_max as f64)
        .cloned()
        .collect();

    // Run inference on stratum
    let posterior = infer_causal(
        prior_ate(),
        DoublyRobust,
        Box::new(PluginEstimator),
        5,
        &stratum,
    )?;

    results.push((age_min, age_max, posterior.point_estimate));
}

// Display results
for (min, max, effect) in results {
    println!("Age [{}, {}): ATE = {:.4}", min, max, effect);
}
```

#### Option 2: Use R-Learner (Residualization)

R-Learner is designed for heterogeneous effects:

```rust
let posterior = infer_causal(
    prior_ate(),
    RLearner,      // Optimized for HTE
    Box::new(PluginEstimator),
    5,
    &observations,
)?;

// θ(X) = residual treatment coefficient
// Varies with X in the loss surface
```

### Interpretation

1. **Similar effects across strata** → Effect is homogeneous
2. **Different effects** → Heterogeneity detected
3. **Credible intervals overlap** → Cannot confidently distinguish
4. **Non-overlapping CIs** → Robust evidence of heterogeneity

---

## Parameter Sensitivity Analysis

### Scenario
You have a system with many parameters and want to know which ones most affect the outcome. Treat each parameter variation as a causal treatment.

### Approach: CATE for Parameters

Treat each parameter as a "treatment" and estimate its causal effect:

```rust
use fugue_causal::*;

let parameters = vec!["CutoffFreq", "Resonance", "AttackTime"];
let mut importances = Vec::new();

for param_name in parameters {
    // Generate experiments varying parameter
    let observations: Vec<Vec<f64>> = (0..1000)
        .map(|i| {
            // X = parameter value (covariate)
            // A = low/high setting (treatment)
            // Y = output quality (outcome)
            let x = /* parameter value */;
            let a = /* low (0) vs high (1) */ if i % 2 == 0 { 0.0 } else { 1.0 };
            let y = /* simulate output */ ;
            vec![x, a, y]
        })
        .collect();

    // Inference
    let posterior = infer_causal(
        prior_ate(),
        DoublyRobust,
        Box::new(PluginEstimator),
        5,
        &observations,
    )?;

    importances.push((
        param_name,
        posterior.point_estimate,
        posterior.posterior_sd,
    ));
}

// Rank by effect size
importances.sort_by(|a, b| {
    b.1.abs().partial_cmp(&a.1.abs()).unwrap()
});

for (name, effect, sd) in importances {
    println!("{}: {:.4} ± {:.4}", name, effect, sd);
}
```

### Interpretation

- **Large effect + narrow CI** → Parameter is important & well-estimated
- **Small effect** → Parameter has little impact
- **Large SD** → Insufficient data or noisy outcome

---

## Best Practices

### 1. Data Checks

Before inference:
```rust
// Overlap check: Do both A=0 and A=1 appear in all covariate regions?
// Overlap is critical for identification

// Covariate balance: Check if confounder assumptions reasonable
// Compare covariate distributions A=0 vs A=1
```

### 2. Identifier Selection

| Scenario | Identifier |
|----------|-----------|
| Few confounders, strong overlap | IPW (faster) |
| Outcome well-modeled | RA |
| Both good, or unsure | **DR (recommended)** |
| Heterogeneous effects | **R-Learner** |

### 3. Nuisance Estimator

```rust
// For small n (<500): Simple linear/logistic
let estimator = Box::new(PluginEstimator);

// For medium n (500-5000): Gradient boosting
// (integrate XGBoost or LightGBM)

// For large n (>5000): Neural networks or causal forests
```

### 4. Fold Selection

```rust
// K-fold cross-fitting: typically 5-10 folds
// - K=5: default, good balance
// - K=10: higher precision, more computation
// - K=n: leave-one-out (expensive)

let num_folds = 5;  // recommended
```

### 5. Credible Interval Interpretation

```rust
// 95% CI from Gibbs posterior (calibrated ω)
// → 95% of true values, over repeated experiments
// → Valid frequentist coverage (via bootstrap)

let z_95 = 1.96;
let ci = [
    posterior.point_estimate - z_95 * posterior.posterior_sd,
    posterior.point_estimate + z_95 * posterior.posterior_sd,
];
```

---

## Common Pitfalls

### ❌ Ignoring Overlap

**Problem**: If treatment assignment is deterministic in some regions (ê near 0 or 1), IPW/DR fails.

**Fix**: Check overlap before inference, trim extreme propensity scores, or use causal forests.

### ❌ Using Biased Nuisance Estimators

**Problem**: If propensity/outcome model is misspecified, inference is invalid (even with orthogonality).

**Fix**: Use cross-validation + flexible models (boosting, splines). Check overlap explicitly.

### ❌ Not Cross-Fitting

**Problem**: Fitting nuisances and evaluating loss on same data → bias.

**Fix**: Always use K-fold (default in `infer_causal`).

### ❌ Forgetting Uncertainty Quantification

**Problem**: Point estimate only, no confidence intervals → false precision.

**Fix**: Always report posterior SD and credible intervals.

---

## Next Steps

- Read [Theory](theory.md) for formal guarantees
- Explore [Examples](examples.md) for more complex scenarios
- Check [API Reference](api-reference.md) for complete function signatures
