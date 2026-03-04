//! Synthesis Parameter Sensitivity Analysis
//!
//! Demonstrates:
//! - CATE inference for identifying "which parameters matter"
//! - Integration with audio synthesis (conceptual)
//! - Parameter importance ranking via credible interval width

use fugue_causal::{
    infer_causal, prior_ate, DoublyRobust, PluginEstimator,
};

fn main() {
    println!("=== Fugue-Causal: Synthesis Parameter Sensitivity ===\n");

    // Scenario: Audio synthesis with 4 parameters
    // - Cutoff frequency (X1): affects brightness (high effect variability)
    // - Resonance (X2): affects timbre (moderate effect)
    // - Attack time (X3): affects onset (low effect)
    // - Sustain level (X4): affects sustain (low effect)
    //
    // Treatment: Apply a fixed synthesis pipeline with "high" vs "low" parameter settings
    // Outcome: Listener preference score (0-100)

    let parameters = vec!["Cutoff", "Resonance", "Attack", "Sustain"];
    let true_effects = vec![8.5, 4.2, 1.5, 0.8]; // True causal effects (synthetic)
    let n_per_param = 200;
    let num_folds = 5;

    println!("Scenario: Audio Synthesis Parameter Importance");
    println!("True Effects: {:?}\n", true_effects);

    let mut param_results = Vec::new();

    for (param_idx, param_name) in parameters.iter().enumerate() {
        let true_effect = true_effects[param_idx];

        // Generate synthetic data: listener preferences conditional on parameter
        // X ~ parameter value (0-10 scale), A ~ high/low setting, Y ~ preference score
        let observations: Vec<(Vec<f64>, f64, f64)> = (0..n_per_param)
            .map(|i| {
                let seed = (i as f64 * (param_idx as f64 + 1.0) * 23.11).sin() * 43758.5453;
                let x = (seed.fract() * 10.0); // Parameter range [0, 10]
                let a = if seed.fract() > 0.5 { 1.0 } else { 0.0 }; // Low/High setting
                // Outcome: base score + parameter effect + treatment effect
                let base = 50.0 + (x - 5.0) * 2.0;
                let y = base + a * true_effect + (seed.sin() * 3.0);
                (vec![x], a, y)
            })
            .collect();

        // Inference
        match infer_causal(
            prior_ate(),
            DoublyRobust,
            Box::new(PluginEstimator),
            num_folds,
            &observations,
        ) {
            Ok(posterior) => {
                let z_95 = 1.96;
                let ci_width = 2.0 * z_95 * posterior.posterior_sd;
                let ci_lower = posterior.point_estimate - z_95 * posterior.posterior_sd;
                let ci_upper = posterior.point_estimate + z_95 * posterior.posterior_sd;

                let covers = ci_lower <= true_effect && true_effect <= ci_upper;
                let relative_importance = true_effect / true_effects.iter().sum::<f64>();

                param_results.push((
                    param_name.to_string(),
                    true_effect,
                    posterior.point_estimate,
                    posterior.posterior_sd,
                    ci_width,
                    ci_lower,
                    ci_upper,
                    covers,
                    relative_importance,
                ));
            }
            Err(e) => eprintln!("Error for {}: {}", param_name, e),
        }
    }

    // Sort by importance (true effect size)
    param_results.sort_by(|a, b| b.8.partial_cmp(&a.8).unwrap());

    println!("Parameter Importance Ranking:");
    println!(
        "{:<12} {:<10} {:<10} {:<8} {:<8} {:<12} {:<8}",
        "Parameter", "True", "Estimate", "Std Dev", "CI Width", "95% CI", "Valid"
    );
    println!("{}", "-".repeat(85));

    for (name, truth, est, sd, ci_w, ci_l, ci_u, covers, imp) in param_results.iter() {
        let ci_str = format!("[{:.2}, {:.2}]", ci_l, ci_u);
        let mark = if *covers { "✓" } else { "✗" };
        let imp_str = format!("{:.1}%", imp * 100.0);
        println!(
            "{:<12} {:<10.2} {:<10.2} {:<8.3} {:<8.3} {:<12} {} ({})",
            name, truth, est, sd, ci_w, ci_str, mark, imp_str
        );
    }

    println!("\n=== Interpretation ===");
    println!("Parameter importance is determined by:");
    println!("1. Effect size (true causal effect)");
    println!("2. Uncertainty (posterior SD, CI width)");
    println!("\nParameters with larger effects and narrower CIs are more important");
    println!("for synthesis quality optimization.");
    println!("\n=== End Example ===");
}
