//! Basic ATE (Average Treatment Effect) inference example
//!
//! Demonstrates:
//! - Synthetic data generation with known true ATE
//! - Doubly Robust identifier selection
//! - Posterior inference with credible intervals
//! - Convergence to ground truth

use fugue_causal::{infer_causal, prior_ate, DoublyRobust, PluginEstimator};

fn main() {
    println!("=== Fugue-Causal: Basic ATE Inference ===\n");

    // Parameters
    let n = 500;
    let true_ate = 2.5;
    let num_folds = 5;

    // Generate synthetic data: X ~ N(0,1), A ~ Bernoulli(0.5), Y = X + A*ATE + noise
    let data: Vec<(f64, f64, f64)> = (0..n)
        .map(|i| {
            let seed = (i as f64 * 12.9898).sin() * 43758.5453;
            let x = (seed.fract() - 0.5) * 2.0; // Uniform(-1, 1) approximation
            let a = if seed.fract() > 0.5 { 1.0 } else { 0.0 };
            let y = x + a * true_ate + (seed.sin() * 0.5); // noise
            (x, a, y)
        })
        .collect();

    println!("Data Generation:");
    println!("  n = {}", n);
    println!("  true_ate = {}", true_ate);
    println!("  X ~ U(-1, 1), A ~ Bernoulli(0.5)");
    println!("  Y = X + A*ATE + noise(σ=0.5)\n");

    // Convert to observation format: (X, A, Y) tuples
    let observations: Vec<(Vec<f64>, f64, f64)> =
        data.iter().map(|(x, a, y)| (vec![*x], *a, *y)).collect();

    // Inference
    println!("Running Causal Inference:");
    println!("  Identifier: Doubly Robust (DR/AIPW)");
    println!("  Folds: {}", num_folds);
    println!("  Prior: N(0, 1)\n");

    match infer_causal(
        prior_ate(),
        DoublyRobust,
        Box::new(PluginEstimator),
        num_folds,
        &observations,
    ) {
        Ok(posterior) => {
            println!("Results:");
            println!("  Point Estimate:    {:.4}", posterior.point_estimate);
            println!("  Posterior Std Dev: {:.4}", posterior.posterior_sd);
            println!("  ω (inverse temp):  {:.4}", posterior.omega);
            println!("  Calibration:       {}\n", posterior.calibration_method);

            // 95% Credible Interval
            let z_95 = 1.96;
            let ci_lower = posterior.point_estimate - z_95 * posterior.posterior_sd;
            let ci_upper = posterior.point_estimate + z_95 * posterior.posterior_sd;
            println!(
                "  95% Credible Interval: [{:.4}, {:.4}]",
                ci_lower, ci_upper
            );

            // Check coverage
            let covers_truth = ci_lower <= true_ate && true_ate <= ci_upper;
            println!(
                "  Contains Truth:    {}",
                if covers_truth { "✓" } else { "✗" }
            );

            // Diagnostics
            let bias = posterior.point_estimate - true_ate;
            let rmse = bias.abs();
            println!("\n  Bias:    {:.4}", bias);
            println!("  RMSE:    {:.4}", rmse);
            println!(
                "  Coverage: {}",
                if covers_truth { "valid" } else { "invalid" }
            );
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    println!("\n=== End Example ===");
}
