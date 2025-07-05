use bpe::pretokenize_single;
use criterion::{criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    // Load the data for the benchmark
    let contents = std::fs::read_to_string("../data/TinyStoriesV2-GPT4-train-tiny.txt")
        .expect("the input file to be there");

    c.bench_function("pretokenize single", |b| {
        b.iter(|| {
            pretokenize_single(
                std::hint::black_box(&contents),
                std::hint::black_box(&["<|endoftext|>"]),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
