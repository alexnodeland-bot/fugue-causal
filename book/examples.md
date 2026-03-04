# Examples: Complete Working Code

## 1. Basic ATE Inference

**File**: `examples/ate_basic.rs`

**What**: Estimate average treatment effect with synthetic data and validation against ground truth.

**Run**: `cargo run --example ate_basic`

**Key concepts**:
- Synthetic data generation
- Doubly Robust identifier selection
- Credible intervals and coverage checking
- Convergence diagnostics

---

## 2. Heterogeneous Treatment Effects (CATE)

**File**: `examples/cate_heterogeneous.rs`

**What**: Detect treatment effect heterogeneity across covariate strata.

**Run**: `cargo run --example cate_heterogeneous`

**Key concepts**:
- Stratified analysis
- Effect variation with covariates
- R-Learner identifier
- Comparison across strata

---

## 3. Synthesis Parameter Sensitivity

**File**: `examples/synthesis_parameters.rs`

**What**: Identify which audio synthesis parameters most affect perceived quality (integration with quiver).

**Run**: `cargo run --example synthesis_parameters`

**Key concepts**:
- CATE for parameter importance ranking
- Effect size vs uncertainty tradeoff
- Integration with generative systems
- Parameter optimization via causal inference

---

## 4. Quantum Gate Importance

**File**: `examples/quantum_importance.rs`

**What**: Determine which quantum gates most impact circuit success probability (integration with QCSim).

**Run**: `cargo run --example quantum_importance`

**Key concepts**:
- Causal analysis of circuit design
- Gate fidelity as treatment effect
- Depth-dependent heterogeneity
- Integration with quantum simulation

---

## How to Adapt Examples

### Example Template

```rust
use fugue_causal::*;

fn main() {
    // 1. Generate or load observations
    let observations: Vec<Vec<f64>> = vec![
        vec![x1, a1, y1],
        vec![x2, a2, y2],
        // ...
    ];

    // 2. Run inference
    let posterior = infer_causal(
        prior_ate(),              // Or prior_ate_informed(mean, var)
        DoublyRobust,             // Or RLearner, IPW, RA
        Box::new(PluginEstimator), // Or custom NuisanceEstimator
        5,                        // K-fold
        &observations,
    ).expect("Inference failed");

    // 3. Extract and display results
    println!("Estimate: {:.4}", posterior.point_estimate);
    println!("Std Dev: {:.4}", posterior.posterior_sd);
}
```

### Modify for Your Data

1. **Replace data generation** with your actual data loading:
   ```rust
   let observations = load_csv("data.csv")?;
   ```

2. **Choose appropriate identifier** based on your domain:
   ```rust
   DoublyRobust     // Most robust, default
   RLearner         // For heterogeneous effects
   IPW              // If propensity well-modeled
   ```

3. **Implement custom NuisanceEstimator** if needed:
   ```rust
   struct MyEstimator { /* ... */ }
   impl NuisanceEstimator for MyEstimator { /* ... */ }
   ```

4. **Adjust prior** for domain knowledge:
   ```rust
   prior_ate_informed(
       5.0,   // Expected effect (e.g., 5% improvement)
       1.0,   // Uncertainty (variance)
   )
   ```

---

## Output Interpretation

All examples print results in this format:

```
Point Estimate:    2.5000
Posterior Std Dev: 0.2500
95% Credible Interval: [2.0039, 2.9961]
Contains Truth:    ✓
```

**Meaning**:
- **Point Estimate**: Best estimate of causal effect
- **Posterior Std Dev**: Uncertainty (squared = variance)
- **95% CI**: Range containing true effect with 95% probability
- **Contains Truth**: ✓ if CI covers known true effect (for synthetic data)

---

## Extending the Examples

### Add More Identifiers

```rust
use fugue_causal::RLearner;

// Run with R-Learner and compare
let posterior_rl = infer_causal(
    prior_ate(),
    RLearner,
    Box::new(PluginEstimator),
    5,
    &observations,
)?;

// Do results agree across identifiers?
println!("DR/AIPW: {:.4}", posterior_dr.point_estimate);
println!("R-Learner: {:.4}", posterior_rl.point_estimate);
```

### Add Bootstrap Calibration

```rust
use fugue_causal::calibrate_omega;

let calibration = calibrate_omega(
    &|obs: &[f64]| {
        // Loss function for this observation
        let loss = /* ... */;
        loss
    },
    posterior.point_estimate,
    posterior.posterior_sd * posterior.posterior_sd,
    &observations,
    0.95,  // 95% target coverage
    100,   // Bootstrap replicates
)?;

println!("Calibrated ω: {}", calibration.omega);
```

### Sensitivity Analysis

```rust
// Vary K-fold and compare stability
for k in &[3, 5, 10] {
    let posterior = infer_causal(
        prior_ate(),
        DoublyRobust,
        Box::new(PluginEstimator),
        *k,
        &observations,
    )?;
    println!("K={}: ATE = {:.4} ± {:.4}", k, posterior.point_estimate, posterior.posterior_sd);
}
```

---

## Performance Benchmarks

Typical runtime on modern laptop:

| n (observations) | Identifier | Time |
|-----------------|-----------|------|
| 500 | DoublyRobust | < 100ms |
| 1000 | DoublyRobust | < 200ms |
| 5000 | DoublyRobust | < 1s |
| 10000 | DoublyRobust | < 2s |

Phase 4 (parallel processing) will enable faster execution for large n.

---

## Next Steps

- Modify an example for your own data
- Implement a custom `NuisanceEstimator` (e.g., XGBoost)
- Check [Tutorials](tutorials.md) for detailed guidance
- Review [Theory](theory.md) for formal guarantees
