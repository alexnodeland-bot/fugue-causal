use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fugue_causal::{infer_causal, prior_ate, DoublyRobust, PluginEstimator, RLearner};

/// Generate synthetic data for benchmarking
fn generate_synthetic_data(n: usize) -> Vec<(Vec<f64>, f64, f64)> {
    (0..n)
        .map(|i| {
            let seed = (i as f64 * 12.9898).sin() * 43758.5453;
            let x = (seed.fract() - 0.5) * 2.0;
            let a = if seed.fract() > 0.5 { 1.0 } else { 0.0 };
            let y = x + 2.5 * a + (seed.sin() * 0.5);
            (vec![x], a, y)
        })
        .collect()
}

fn benchmark_dr_ate_500(c: &mut Criterion) {
    let data = black_box(generate_synthetic_data(500));

    c.bench_function("dr_ate_n500", |b| {
        b.iter(|| {
            infer_causal(
                prior_ate(),
                DoublyRobust,
                Box::new(PluginEstimator),
                5,
                &data,
            )
        })
    });
}

fn benchmark_dr_ate_1000(c: &mut Criterion) {
    let data = black_box(generate_synthetic_data(1000));

    c.bench_function("dr_ate_n1000", |b| {
        b.iter(|| {
            infer_causal(
                prior_ate(),
                DoublyRobust,
                Box::new(PluginEstimator),
                5,
                &data,
            )
        })
    });
}

fn benchmark_rlearner_ate_500(c: &mut Criterion) {
    let data = black_box(generate_synthetic_data(500));

    c.bench_function("rlearner_ate_n500", |b| {
        b.iter(|| infer_causal(prior_ate(), RLearner, Box::new(PluginEstimator), 5, &data))
    });
}

fn benchmark_rlearner_ate_1000(c: &mut Criterion) {
    let data = black_box(generate_synthetic_data(1000));

    c.bench_function("rlearner_ate_n1000", |b| {
        b.iter(|| infer_causal(prior_ate(), RLearner, Box::new(PluginEstimator), 5, &data))
    });
}

fn benchmark_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");

    for n in &[100, 500, 1000, 5000] {
        let data = black_box(generate_synthetic_data(*n));
        let bench_name = format!("dr_ate_n{}", n);

        group.bench_function(&bench_name, |b| {
            b.iter(|| {
                infer_causal(
                    prior_ate(),
                    DoublyRobust,
                    Box::new(PluginEstimator),
                    5,
                    &data,
                )
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_dr_ate_500,
    benchmark_dr_ate_1000,
    benchmark_rlearner_ate_500,
    benchmark_rlearner_ate_1000,
    benchmark_scaling
);
criterion_main!(benches);
