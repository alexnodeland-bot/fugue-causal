//! Quantum Gate Importance Inference
//!
//! Demonstrates:
//! - CATE for quantum circuit parameter analysis
//! - Gate fidelity as treatment effect
//! - Integration path with QCSim (conceptual)

use fugue_causal::{
    infer_causal, prior_ate, RLearner, PluginEstimator,
};

fn main() {
    println!("=== Fugue-Causal: Quantum Gate Importance ===\n");

    // Scenario: Quantum error correction circuit
    // - Different gate types: {CNOT, RX, RZ, H}
    // - Treatment: Perfect fidelity vs Noisy fidelity
    // - Outcome: Circuit success probability / error correction capacity
    // - Covariate X: Circuit depth (number of gates)

    let gates = vec!["CNOT", "RX", "RZ", "H"];
    let true_importances = vec![0.12, 0.08, 0.05, 0.03]; // Relative importance
    let n_per_gate = 300;
    let num_folds = 5;

    println!("Scenario: Quantum Error Correction Gate Importance");
    println!("Question: Which gate type most impacts success probability?\n");

    let mut gate_results = Vec::new();

    for (gate_idx, gate_name) in gates.iter().enumerate() {
        let true_importance = true_importances[gate_idx];

        // Generate synthetic data:
        // X ~ circuit depth, A ~ gate fidelity (0=noisy, 1=perfect), Y ~ success prob
        let observations: Vec<(Vec<f64>, f64, f64)> = (0..n_per_gate)
            .map(|i| {
                let seed = (i as f64 * (gate_idx as f64 + 1.0) * 31.41).sin() * 43758.5453;
                let x = (seed.fract() * 20.0); // Circuit depth [0, 20]
                let a = if seed.fract() > 0.5 { 1.0 } else { 0.0 }; // Fidelity level
                // Success probability: base decay with depth + gate importance + fidelity boost
                let depth_decay = (1.0 - x / 50.0).max(0.0);
                let base = 0.9 * depth_decay;
                let y = (base + a * true_importance).min(1.0).max(0.0);
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
                let se = posterior.posterior_sd;
                let ci_lower = posterior.point_estimate - z_95 * se;
                let ci_upper = posterior.point_estimate + z_95 * se;

                let covers = ci_lower <= true_importance && true_importance <= ci_upper;
                let relative_imp = true_importance / true_importances.iter().sum::<f64>();

                gate_results.push((
                    gate_name.to_string(),
                    true_importance,
                    posterior.point_estimate,
                    se,
                    ci_lower,
                    ci_upper,
                    covers,
                    relative_imp,
                ));
            }
            Err(e) => eprintln!("Error for gate {}: {}", gate_name, e),
        }
    }

    // Sort by importance
    gate_results.sort_by(|a, b| b.7.partial_cmp(&a.7).unwrap());

    println!("Gate Importance Ranking (Fidelity Effect on Success):");
    println!(
        "{:<8} {:<10} {:<10} {:<8} {:<18} {:<8} {:<10}",
        "Gate", "True Eff", "Est. Eff", "Std Dev", "95% CI", "Valid", "Relative"
    );
    println!("{}", "-".repeat(85));

    for (name, truth, est, se, ci_l, ci_u, covers, imp) in gate_results.iter() {
        let ci_str = format!("[{:.4}, {:.4}]", ci_l, ci_u);
        let mark = if *covers { "✓" } else { "✗" };
        let imp_str = format!("{:.1}%", imp * 100.0);
        println!(
            "{:<8} {:<10.4} {:<10.4} {:<8.4} {:<18} {} {:<10}",
            name, truth, est, se, ci_str, mark, imp_str
        );
    }

    println!("\n=== Interpretation ===");
    println!("Effect of improving gate fidelity on circuit success:");
    println!();
    for (name, _, est, _, _, _, _, imp) in gate_results.iter() {
        let stars = "*".repeat((imp * 50.0) as usize);
        println!("  {:<6} {:+.4} {} {:.0}%", name, est, stars, imp * 100.0);
    }

    println!("\n=== QCSim Integration ===");
    println!("These estimates feed into QCSim parameter search:");
    println!("1. High-importance gates → prioritize fidelity improvements");
    println!("2. Low-importance gates → acceptable error rates");
    println!("3. Depth interactions → detect critical gate combinations");
    println!("\n=== End Example ===");
}
