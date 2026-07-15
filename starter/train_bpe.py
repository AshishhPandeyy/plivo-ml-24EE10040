"""Train a byte-level BPE on the provided corpus ONLY.

Base vocab = 256 raw bytes (guarantees lossless byte fallback for arbitrary
UTF-8), plus learned merges up to --vocab_size. Merges are saved to
bpe_merges.json next to tokenizer.py, which loads them at eval time.

    python train_bpe.py --data ../data/train_corpus.txt --vocab_size 1024
"""
import argparse
import collections
import json
import os
import re
import time

# split into space-prefixed word chunks so merges never cross word
# boundaries; keeps merge table small and encoding fast
PRETOK = re.compile(r" ?[^\s]+|\s+")


def get_word_counts(text):
    counts = collections.Counter(PRETOK.findall(text))
    return {tuple(w.encode("utf-8")): c for w, c in counts.items()}


def train(text, vocab_size):
    assert vocab_size > 256
    words = get_word_counts(text)          # tuple(byte ids) -> count
    # pair -> total count, and pair -> set of words containing it
    pair_counts = collections.Counter()
    pair_words = collections.defaultdict(set)
    for w, c in words.items():
        for a, b in zip(w, w[1:]):
            pair_counts[(a, b)] += c
            pair_words[(a, b)].add(w)

    merges = []
    next_id = 256
    while next_id < vocab_size:
        if not pair_counts:
            break
        pair = max(pair_counts, key=pair_counts.__getitem__)
        if pair_counts[pair] < 2:
            break
        merges.append(pair)
        affected = pair_words.pop(pair, set())
        del pair_counts[pair]
        for w in affected:
            c = words.pop(w, None)
            if c is None:
                continue
            # remove old pair contributions of this word
            for a, b in zip(w, w[1:]):
                p = (a, b)
                if p in pair_counts:
                    pair_counts[p] -= c
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                pair_words[p].discard(w)
            # apply the merge inside the word
            new_w, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == pair:
                    new_w.append(next_id)
                    i += 2
                else:
                    new_w.append(w[i])
                    i += 1
            new_w = tuple(new_w)
            words[new_w] = words.get(new_w, 0) + c
            for a, b in zip(new_w, new_w[1:]):
                pair_counts[(a, b)] += c
                pair_words[(a, b)].add(new_w)
        next_id += 1
    return merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/train_corpus.txt")
    ap.add_argument("--vocab_size", type=int, default=1024)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "bpe_merges.json")
    text = open(args.data, encoding="utf-8").read()
    t0 = time.time()
    merges = train(text, args.vocab_size)
    with open(out, "w") as f:
        json.dump({"vocab_size": 256 + len(merges),
                   "merges": [list(p) for p in merges]}, f)
    print(f"trained {len(merges)} merges -> vocab {256 + len(merges)} "
          f"({time.time() - t0:.0f}s), saved {out}")


if __name__ == "__main__":
    main()
