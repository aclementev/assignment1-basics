from cs336_basics.bpe import naive_bpe, pretokenized_counts_rust


def main():
    special_tokens = ["<|endoftext|>"]
    with open("data/TinyStoriesV2-GPT4-train-tiny.txt") as f:
        corpus = f.read()

    naive_bpe(corpus, 1000, special_tokens, pretok_strategy=pretokenized_counts_rust)


if __name__ == "__main__":
    # To make multiprocessing happy
    main()
