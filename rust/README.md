## Performance analysis

We use `criterion` for running statistically significant benchmarks. These 
are stored in the `benches` directory.

You can run them with:

```sh
cargo bench
```

We also use `cargo-flamegraph` for profiling, which can be used together with
`criterion` for running a function:

```sh
cargo flamegraph --bench benchmark -- --bench && open flamegraph.svg
```
