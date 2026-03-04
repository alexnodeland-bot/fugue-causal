# Quick Start: 5 Minutes

## Installation

Add to `Cargo.toml`:

```toml
[dependencies]
fugue-causal = "0.2.0"
```

## Your First Inference

```rust
use fugue_causal::{infer_causal, prior_ate, DoublyRobust, PluginEstimator};

fn main() {
    // Step 1: Prepare data [covariate, treatment, outcome]
    let observations = vec![
        vec![0.5, 1.0, 5.2],   // Person 1: covariate=0.5, treated=1, outcome=5.2
        vec![-0.3, 0.0, 1.8],  // Person 2: covariate=-0.3, control=0, outcome=1.8
        // ... more observations
    ];

    // Step 2: Run causal inference
    let posterior = infer_causal(
        prior_ate(),                      // Prior on causal effect
        DoublyRobust,                     // Identification strategy
        Box::new(PluginEstimator),       // Nuisance estimator
        5,                                // K-fold cross-fitting
        &observations,
    ).expect("Inference failed");

    // Step 3: Interpret results
    println!("Average Treatment Effect: {:.4}", posterior.point_estimate);
    println!("Posterior Std Dev: {:.4}", posterior.posterior_sd);
    
    // 95% Credible Interval
    let z_95 = 1.96;
    let ci_lower = posterior.point_estimate - z_95 * posterior.posterior_sd;
    let ci_upper = posterior.point_estimate + z_95 * posterior.posterior_sd;
    println!("95% CI: [{:.4}, {:.4}]", ci_lower, ci_upper);
}
```

## Data Requirements

Your observations must be:
- **Format**: Vec<Vec<f64>> where each inner vec = [X, A, Y]
  - **X**: Covariate(s) as f64
  - **A**: Binary treatment 0.0 (control) or 1.0 (treated)
  - **Y**: Continuous outcome
- **Size**: At least 100-200 observations (more is better)
- **Overlap**: Both A=0 and A=1 must appear across covariate ranges

## Identifiers at a Glance

```rust
use fugue_causal::{DoublyRobust, RLearner, InverseProbabilityWeighting, RegressionAdjustment};

// General observational studies (most robust)
DoublyRobust

// Heterogeneous treatment effects
RLearner

// When propensity is well-modeled
InverseProbabilityWeighting

// When outcomes are well-modeled
RegressionAdjustment
```

## Run an Example

Clone the repo and run:

```bash
cargo run --example ate_basic
cargo run --example cate_heterogeneous
```

## Next Steps

- **Tutorials**: See [Tutorials](tutorials.md) for step-by-step guides
- **Theory**: Deep dive in [Theory](theory.md)
- **Full Examples**: Check [Examples](examples.md)
- **API**: Browse [API Reference](api-reference.md)

## Common Pitfalls

❌ **Forgetting overlap**: Ensure treatment variation across covariates  
✅ **Fix**: Check propensity scores, trim if ê too close to 0/1

❌ **Ignoring uncertainty**: Only reporting point estimates  
✅ **Fix**: Always include posterior_sd and credible intervals

❌ **Mixing cross-fitting data**: Fitting and evaluating on same data  
✅ **Fix**: Use K-fold (enabled by default)

---

**Ready to dive in? Pick a [Tutorial](tutorials.md)!**
