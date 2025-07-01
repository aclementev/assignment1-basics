import os
from collections import Counter
from collections.abc import Iterable
from itertools import takewhile

import regex as re

PAT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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

    vocab, merges = naive_bpe(corpus, vocab_size=vocab_size, special_tokens=special_tokens)
    return vocab, merges


def preprocess_special_tokens(corpus: bytes, special_tokens: list[bytes]) -> Iterable[bytes]:
    """Count the occurrences of each of the strings in `special_tokens` inside `corpus` returning them
    in a frequency table, and returns a string which does not contain the `special_tokens`"""
    if not special_tokens:
        return [corpus]

    pattern = re.compile(b"|".join(re.escape(tok) for tok in special_tokens))
    parts = re.splititer(pattern, corpus)
    return parts


def pretokenized_counts(corpus: bytes, special_tokens: list[bytes]) -> dict[tuple[bytes, ...], int]:
    """Run a pre-tokenizer similar to the one used by GPT-2, returning pretokenized counts for efficiency"""

    # TODO(alvaro): Use re.finditer and count somehow without materializing all tokens
    # FIXME(alvaro): How do we handle the special tokens here? The pre-tokenization will split them
    # I think the correct way is to count them separately and then pre-tokenize without them
    text_parts = preprocess_special_tokens(corpus, special_tokens)
    pattern = re.compile(PAT)

    freqs: Counter[tuple[bytes, ...]] = Counter()
    for part in text_parts:
        # TODO(alvaro): There's a concurrent mode, maybe it works okay
        scanner = pattern.finditer(part, concurrent=True)
        freqs += Counter(tuple(c.to_bytes() for c in m.group(0)) for m in scanner)

    return freqs


def naive_bpe(
    corpus: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges = []
    corpus_bytes = corpus.encode("utf-8")
    special_tok_bytes = [tok.encode("utf-8") for tok in special_tokens]

    # 1. Initialize vocabulary (bytes + special tokens)
    vocab_lst = [*special_tok_bytes, *(i.to_bytes() for i in range(256))]

    # 2. Pre-tokenize
    tok_counts = pretokenized_counts(corpus_bytes, special_tok_bytes)

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
