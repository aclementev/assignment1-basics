import operator
import os
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from itertools import takewhile

import regex as re

from cs336_basics._bpe import pretokenize_naive as pretokenize_naive_rust

PAT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# For the list of available implementations, see below
DEFAULT_PRETOK_IMPL = "parallel"

type PretokStrategy = Callable[[bytes, list[bytes]], dict[tuple[bytes, ...], int]]


# from typing import Protocol, Self

# class Tokenizer(Protocol):
#     def get_vocab(self) -> dict[int, bytes]: ...
#     def get_merges(self) -> list[tuple[bytes, bytes]]: ...
#     def get_special_tokens(self) -> list[str]: ...
#     def encode(self, data: str) -> list[int]: ...
#     def decode(self, tokens: list[int]) -> str: ...
#
#
# class NaiveTokenizer:
#     def __init__(
#         self, vocab_size: int, special_tokens: list[str], vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]
#     ):
#         self.vocab_size = vocab_size
#         self.special_tokens = special_tokens
#         self.vocab = vocab
#         self.merges = merges
#
#     def get_vocab(self) -> dict[int, bytes]: ...
#     def get_merges(self) -> list[tuple[bytes, bytes]]: ...
#     def get_special_tokens(self) -> list[str]: ...
#     def encode(self, data: str) -> list[int]: ...
#     def decode(self, tokens: list[int]) -> str: ...
#
#     @classmethod
#     def train(cls, corpus: str, vocab_size: int, special_tokens: list[str]) -> Self:
#         vocab, merges = naive_bpe(corpus, vocab_size=vocab_size, special_tokens=special_tokens)
#         return cls(vocab_size, special_tokens, vocab, merges)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer. Returns the vocabulary and merges"""
    with open(input_path) as f:
        corpus = f.read()

    # Configure the pretokenization strategy
    pretok_strategy_name = os.environ.get("BPE_PRETOK_STRATEGY") or DEFAULT_PRETOK_IMPL
    pretok_strategy = PRETOK_IMPLS[pretok_strategy_name]

    vocab, merges = naive_bpe(
        corpus, vocab_size=vocab_size, special_tokens=special_tokens, pretok_strategy=pretok_strategy
    )
    return vocab, merges


def split_special_tokens(corpus: bytes, special_tokens: list[bytes]) -> Iterable[bytes]:
    """Count the occurrences of each of the strings in `special_tokens` inside `corpus` returning them
    in a frequency table, and returns a string which does not contain the `special_tokens`"""
    if not special_tokens:
        return [corpus]

    pattern = re.compile(b"|".join(re.escape(tok) for tok in special_tokens))
    parts = re.splititer(pattern, corpus)
    return parts


def pretokenized_counts(corpus: bytes, special_tokens: list[bytes]) -> dict[tuple[bytes, ...], int]:
    """Run a pre-tokenizer similar to the one used by GPT-2, returning pretokenized counts for efficiency"""
    text_parts = split_special_tokens(corpus, special_tokens)
    return _pretokenize_parts(text_parts)


def pretokenized_counts_rust(corpus: bytes, special_tokens: list[bytes]) -> dict[tuple[bytes, ...], int]:
    """Run a pre-tokenizer similar to the one used by GPT-2, returning pretokenized counts for efficiency.

    This one uses a pretokenization implemented in rust
    """
    return pretokenize_naive_rust(corpus.decode("utf-8"), [tok.decode("utf-8") for tok in special_tokens])


def naive_bpe(
    corpus: str, vocab_size: int, special_tokens: list[str], pretok_strategy: PretokStrategy | None = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges = []
    corpus_bytes = corpus.encode("utf-8")
    special_tok_bytes = [tok.encode("utf-8") for tok in special_tokens]

    # 1. Initialize vocabulary (bytes + special tokens)
    vocab_lst = [*special_tok_bytes, *(i.to_bytes() for i in range(256))]

    # 2. Pre-tokenize
    if not pretok_strategy:
        pretok_strategy = PRETOK_IMPLS[DEFAULT_PRETOK_IMPL]
    tok_counts = pretok_strategy(corpus_bytes, special_tok_bytes)

    # 3. BPE merges on pre-tokenized data until len(vocab) == vocab_size
    i = 1
    while len(vocab_lst) < vocab_size and len(tok_counts) > 1:
        # Find the most common pair of bytes
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for key, count in tok_counts.items():
            if len(key) == 1:
                continue
            # We cannot use the `update` method of a counter since it's possible that a
            # word has repeated byte pairs
            for pair in zip(key, key[1:]):
                pair_counts[pair] += count

        # Most common pairs
        # Max count
        _, top_count = pair_counts.most_common(1)[0]
        top_count_pairs = map(
            lambda item: item[0], takewhile(lambda item: item[1] == top_count, pair_counts.most_common())
        )
        # Take the lexicographically maximum pair
        merge_candidate = max(top_count_pairs)
        assert len(merge_candidate) == 2, "should be a pair"

        # Add it to the merges
        merges.append(merge_candidate)

        # Add it to the vocab
        new_tok = b"".join(merge_candidate)
        vocab_lst.append(new_tok)

        # Update the tok_counts
        new_tok_counts = tok_counts.copy()
        for key, count in tok_counts.items():
            if len(key) == 1:
                continue

            # Apply merges
            new_key = tuple(_merge_key_token(key, new_tok, merge_candidate))
            if new_key == key:
                continue

            # Replace the key
            del new_tok_counts[key]
            new_tok_counts[new_key] = count
        tok_counts = new_tok_counts
        i += 1

    vocab = dict(enumerate(vocab_lst))
    return vocab, merges


def _merge_key_token(key: tuple[bytes, ...], new_tok: bytes, merged: tuple[bytes, bytes]) -> Iterable[bytes]:
    """Update the key of a bpe fequency counter by merging the pairs that match the new token"""
    if len(key) == 1:
        yield next(iter(key))
        return

    key_iter = iter(key)
    left, right = next(key_iter, None), next(key_iter, None)
    while left is not None:
        if (left, right) == merged:
            yield new_tok
            left, right = next(key_iter, None), next(key_iter, None)
        else:
            yield left
            left, right = right, next(key_iter, None)


def _pretokenize_parts(parts: Iterable[bytes]) -> dict[tuple[bytes, ...], int]:
    """Process a sequence of text parts into"""
    pattern = re.compile(PAT)
    freqs: Counter[tuple[bytes, ...]] = Counter()
    for part in parts:
        scanner = pattern.finditer(part, concurrent=True)
        freqs += Counter(tuple(c.to_bytes() for c in m.group(0)) for m in scanner)

    return freqs


def pretokenized_counts_parallel(corpus: bytes, special_tokens: list[bytes]) -> dict[tuple[bytes, ...], int]:
    """Run a pre-tokenizer similar to the one used by GPT-2"""
    text_parts = split_special_tokens(corpus, special_tokens)

    # FIXME(alvaro): Use the number of CPU - 1?
    N = 8
    chunks = _split_chunks(text_parts, N)
    with ProcessPoolExecutor(max_workers=N) as executor:
        chunks_res = executor.map(_pretokenize_parts, chunks)

    # Collect the results from all the workers
    return dict(reduce(operator.add, chunks_res))


def _split_chunks(parts: Iterable[bytes], n: int) -> Iterable[Iterable[bytes]]:
    """Split an iterable of text parts (i.e. as split by special tokens) and group them
    into `n` chunks of size up to N (bytes) that can be processed concurrently
    """

    # FIXME(alvaro): Figure out a way to do this without materializing the whole text?
    part_lst = list(parts)
    total_size = sum(len(p) for p in part_lst)
    chunk_size = total_size // n

    chunk: list[bytes] = []
    run_size = 0
    for part in part_lst:
        chunk.append(part)
        run_size += len(part)
        if run_size >= chunk_size:
            yield chunk
            chunk = []
            run_size = 0

    # Return the accumulated chunk if there are no more parts
    if chunk:
        yield chunk


PRETOK_IMPLS = {
    "naive": pretokenized_counts,
    "parallel": pretokenized_counts_parallel,
    "rust-naive": pretokenized_counts_rust,
}
