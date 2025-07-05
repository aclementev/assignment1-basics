use bpe::{pretokenize_naive_impl, pretokenize_parallel_impl};
use criterion::{criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    // Load the data for the benchmark
    let contents = std::fs::read_to_string("../data/TinyStoriesV2-GPT4-train-tiny.txt")
        .expect("the input file to be there");

    let mut group = c.benchmark_group("pretokenization");
    group.sample_size(50);

    group.bench_function("pretokenize naive", |b| {
        b.iter(|| {
            pretokenize_naive_impl(
                std::hint::black_box(&contents),
                std::hint::black_box(&["<|endoftext|>"]),
            )
        })
    });

    group.bench_function("pretokenize parallel", |b| {
        b.iter(|| {
            pretokenize_parallel_impl(
                std::hint::black_box(&contents),
                std::hint::black_box(&["<|endoftext|>"]),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
