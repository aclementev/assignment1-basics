use itertools::Itertools;
use onig::{Regex, RegexOptions, Syntax};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use regex::escape;
use std::collections::HashMap;

const PAT: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Formats the sum of two numbers as string.
fn pretokenize_single<'a>(corpus: &'a str, special_tokens: &[&str]) -> HashMap<&'a str, usize> {
    // Split text by special tokens
    let pretok_re_str = Itertools::intersperse(
        // FIXME(alvaro): We use the `regex` library's escaper, which could not handle
        // all of the supported sequences, but should do most of them (oniguruma does not provide
        // one)
        special_tokens.into_iter().map(|tok| escape(tok)),
        "|".to_string(),
    )
    .collect::<String>();
    // We need to make sure to specify the syntax to be Python
    let pretok_re = Regex::with_options(
        &pretok_re_str,
        RegexOptions::REGEX_OPTION_NONE,
        Syntax::python(),
    )
    .expect("the special token regex to be valid");

    let mut freqs = HashMap::new();
    let re = Regex::with_options(PAT, RegexOptions::REGEX_OPTION_NONE, Syntax::python())
        .expect("the pretokenization pattern to be valid");

    for part in pretok_re.split(corpus) {
        println!("part: '{}'", &part);
        for capture in re.captures_iter(part).map(|c| c.at(0).unwrap()) {
            *freqs.entry(capture).or_insert(0) += 1
        }
    }

    return freqs;
}

/// Pretokenize a corpus string, returning dictionary with the input byte counts
#[pyfunction]
fn pretokenize_naive(
    corpus: &str,
    special_tokens: Vec<PyBackedStr>,
) -> PyResult<HashMap<&str, usize>> {
    // Unwrap the special tokens
    let special_tokens = &special_tokens.iter().map(|s| &**s).collect::<Vec<&str>>();
    let freqs = pretokenize_single(corpus, &special_tokens);
    Ok(freqs)
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_bpe")]
fn bpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pretokenize_naive, m)?)?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_pretokenize_single() {
        let res = pretokenize_single("foo bar <|endoftext|>baz", &["<|endoftext|>"]);
        let mut expected = HashMap::new();
        expected.insert("foo", 1);
        expected.insert("bar", 1);
        expected.insert("baz", 1);
        expected.insert(" ", 1);
        assert_eq!(res, expected);
    }
}
