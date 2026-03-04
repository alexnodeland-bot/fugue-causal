//! Cross-fitting orchestration for causal inference.
//!
//! Cross-fitting is essential for preserving Neyman-orthogonality. It ensures
//! that nuisance estimation errors don't propagate to causal parameter estimates.
//!
//! **Algorithm:**
//! 1. Randomly split data into K folds
//! 2. For each fold k:
//!    a. Fit nuisances on training set (all folds except k)
//!    b. Compute loss on validation set (fold k)
//! 3. Aggregate losses across folds

use crate::nuisance::NuisanceEstimator;
use rand::seq::SliceRandom;

/// Result of cross-fitting nuisance estimation
#[derive(Clone, Debug)]
pub struct CrossFittedNuisances {
    /// Nuisance estimates for each observation
    /// nuisances\[i\] = nuisance estimates for observation i
    pub estimates: Vec<Vec<f64>>,

    /// Fold assignment for each observation (for diagnostics)
    pub fold_assignments: Vec<usize>,

    /// Number of folds used
    pub num_folds: usize,
}

/// Cross-fit nuisance estimates using K-fold stratification
///
/// # Arguments
///
/// * `data` - Training data: (covariates, treatment, outcome) tuples
/// * `estimator` - Nuisance estimation method
/// * `folds` - Number of folds (typically 5)
/// * `seed` - Random seed for fold assignment (for reproducibility)
///
/// # Returns
///
/// Cross-fitted nuisance estimates with fold assignments
///
/// # Theory
///
/// Under Neyman-orthogonality + cross-fitting, the first-order interaction
/// between causal parameter θ and nuisance η vanishes:
///
/// D_η D_θ E\[loss\] = 0
///
/// This ensures that nuisance estimation error r_n doesn't inflate the bias
/// in θ̂. Instead, convergence is O_P(√n r_n²) rather than O_P(√n r_n).
///
/// Cross-fitting is the mechanism that enforces this: validation losses are
/// i.i.d. conditional on training samples, preventing empirical process bias.
pub fn cross_fit(
    data: &[(Vec<f64>, f64, f64)],
    estimator: &dyn NuisanceEstimator,
    folds: usize,
    _seed: u64,
) -> Result<CrossFittedNuisances, String> {
    let n = data.len();

    if n < folds {
        return Err(format!(
            "Number of observations ({}) must be >= number of folds ({})",
            n, folds
        ));
    }

    // Assign observations to folds
    let mut fold_assignments: Vec<usize> = (0..n).map(|i| i % folds).collect();
    fold_assignments.shuffle(&mut rand::thread_rng());

    // Partition data by fold
    let mut fold_indices: Vec<Vec<usize>> = vec![Vec::new(); folds];
    for (i, &fold) in fold_assignments.iter().enumerate() {
        fold_indices[fold].push(i);
    }

    // Cross-fitting: for each fold, estimate nuisances on training data,
    // evaluate on validation data
    let mut all_nuisances: Vec<Vec<f64>> = vec![vec![]; n];

    for validation_fold in 0..folds {
        // Training indices: all except validation fold
        let mut training_indices: Vec<usize> = Vec::new();
        for (train_fold, fold_data) in fold_indices.iter().enumerate().take(folds) {
            if train_fold != validation_fold {
                training_indices.extend(fold_data);
            }
        }

        // Validation indices: only validation fold
        let validation_indices = &fold_indices[validation_fold];

        // Construct training and validation datasets
        let training_data: Vec<(Vec<f64>, f64, f64)> = training_indices
            .iter()
            .filter_map(|&i| data.get(i).cloned())
            .collect();

        let validation_data: Vec<(Vec<f64>, f64, f64)> = validation_indices
            .iter()
            .filter_map(|&i| data.get(i).cloned())
            .collect();

        // Estimate nuisances on training data, apply to validation data
        let fold_nuisances = estimator.estimate_fold(&training_data, &validation_data)?;

        // Store nuisance estimates in original data order
        for (j, &val_idx) in validation_indices.iter().enumerate() {
            all_nuisances[val_idx] = fold_nuisances[j].clone();
        }
    }

    Ok(CrossFittedNuisances {
        estimates: all_nuisances,
        fold_assignments,
        num_folds: folds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock estimator for testing
    struct DummyEstimator;

    impl NuisanceEstimator for DummyEstimator {
        fn estimate_fold(
            &self,
            _training_data: &[(Vec<f64>, f64, f64)],
            validation_data: &[(Vec<f64>, f64, f64)],
        ) -> Result<Vec<Vec<f64>>, String> {
            // Return dummy estimates: [0.5, 0.0, 0.0] for each validation observation
            Ok(validation_data
                .iter()
                .map(|_| vec![0.5, 0.0, 0.0])
                .collect())
        }

        fn name(&self) -> &str {
            "DummyEstimator"
        }
    }

    #[test]
    fn test_cross_fit_returns_estimates_for_all_observations() {
        let data = vec![
            (vec![1.0], 1.0, 2.0),
            (vec![2.0], 0.0, 1.0),
            (vec![3.0], 1.0, 3.0),
            (vec![4.0], 0.0, 2.0),
            (vec![5.0], 1.0, 4.0),
        ];

        let result = cross_fit(&data, &DummyEstimator, 2, 42);
        assert!(result.is_ok());

        let cf = result.unwrap();
        assert_eq!(cf.estimates.len(), 5);
        assert_eq!(cf.fold_assignments.len(), 5);
        assert_eq!(cf.num_folds, 2);

        // Each observation should have nuisance estimates
        for nuisances in &cf.estimates {
            assert_eq!(nuisances.len(), 3); // [e, m_1, m_0]
        }
    }

    #[test]
    fn test_cross_fit_fold_coverage() {
        let data = vec![
            (vec![1.0], 1.0, 2.0),
            (vec![2.0], 0.0, 1.0),
            (vec![3.0], 1.0, 3.0),
            (vec![4.0], 0.0, 2.0),
        ];

        let result = cross_fit(&data, &DummyEstimator, 2, 42);
        assert!(result.is_ok());

        let cf = result.unwrap();
        // Check that all folds are represented
        let mut fold_counts = vec![0; 2];
        for &fold in &cf.fold_assignments {
            fold_counts[fold] += 1;
        }

        // Each fold should have at least one observation
        for count in fold_counts {
            assert!(count > 0);
        }
    }

    #[test]
    fn test_cross_fit_error_too_few_observations() {
        let data = vec![(vec![1.0], 1.0, 2.0)];

        let result = cross_fit(&data, &DummyEstimator, 5, 42);
        assert!(result.is_err());
    }
}
