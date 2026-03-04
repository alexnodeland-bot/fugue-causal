//! Heterogeneous Treatment Effects (CATE) example
//!
//! Demonstrates:
//! - Stratified inference across covariate groups
//! - Effect heterogeneity detection
//! - Confidence intervals per stratum

use fugue_causal::{
    infer_causal, prior_ate, RLearner, PluginEstimator,
};

fn main() {
    println!("=== Fugue-Causal: Heterogeneous Treatment Effects (CATE) ===\n");

    // Scenario: Treatment effect depends on baseline covariate X
    // E[Y(1) - Y(0) | X] = 1.0 + 2.0*X (effect increases with X)
    let n_total = 1000;
    let num_folds = 5;

    // Generate data stratified by X
    let strata_boundaries = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    let mut stratum_results = Vec::new();

    for (stratum_idx, window) in strata_boundaries.windows(2).enumerate() {
        let x_min = window[0];
        let x_max = window[1];
        let stratum_center = (x_min + x_max) / 2.0;
        let true_ate_stratum = 1.0 + 2.0 * stratum_center;

        let n_stratum = (n_total / strata_boundaries.len()).max(1);

        // Generate observations for this stratum
        let observations: Vec<(Vec<f64>, f64, f64)> = (0..n_stratum)
            .map(|i| {
                // Pseudo-random in stratum
                let seed = (i as f64 * stratum_idx as f64 * 17.3).sin() * 43758.5453;
                let x = x_min + (seed.fract()) * (x_max - x_min);
                let a = if seed.fract() > 0.5 { 1.0 } else { 0.0 };
                let y = x + a * true_ate_stratum + (seed.sin() * 0.2);
                (vec![x], a, y)
            })
            .collect();

        // Inference
        match infer_causal(
            prior_ate(),
            RLearner,
            Box::new(PluginEstimator),
            num_folds,
            &observations,
        ) {
            Ok(posterior) => {
                let z_95 = 1.96;
                let ci_lower = posterior.point_estimate - z_95 * posterior.posterior_sd;
                let ci_upper = posterior.point_estimate + z_95 * posterior.posterior_sd;

                let covers = ci_lower <= true_ate_stratum && true_ate_stratum <= ci_upper;

                stratum_results.push((
                    stratum_idx,
                    x_min,
                    x_max,
                    stratum_center,
                    true_ate_stratum,
                    posterior.point_estimate,
                    posterior.posterior_sd,
                    ci_lower,
                    ci_upper,
                    covers,
                ));
            }
            Err(e) => eprintln!("Error in stratum {}: {}", stratum_idx, e),
        }
    }

    println!("Stratified Causal Inference Results:");
    println!("Model: E[Y(1) - Y(0) | X] = 1.0 + 2.0*X\n");
    println!(
        "{:<8} {:<8} {:<10} {:<10} {:<10} {:<10}",
        "Stratum", "Center", "True ATE", "Est. ATE", "Std Dev", "95% CI"
    );
    println!("{}", "-".repeat(70));

    for (idx, x_min, x_max, center, truth, est, sd, ci_l, ci_u, covers) in stratum_results.iter()
    {
        let ci_str = format!("[{:.3}, {:.3}]", ci_l, ci_u);
        let mark = if *covers { "✓" } else { "✗" };
        println!(
            "{:<8} {:<8.3} {:<10.3} {:<10.3} {:<10.4} {:<10} {}",
            format!("[{}]", idx),
            center,
            truth,
            est,
            sd,
            ci_str,
            mark
        );
    }

    println!("\n=== Key Insight ===");
    println!("Treatment effect increases with baseline covariate X.");
    println!("Heterogeneity is captured across strata.");
    println!("\n=== End Example ===");
}
