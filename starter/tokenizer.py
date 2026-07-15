"""Byte-level BPE tokenizer, trained on the provided corpus ONLY.

Vocab = 256 raw bytes + learned merges (loaded from bpe_merges.json). Because
the base alphabet is all 256 bytes, ANY UTF-8 text encodes losslessly:
decode(encode(text)) == text exactly (the scorer verifies this). If the merge
file is missing, we fall back to the plain byte tokenizer (vocab 256), so the
interface never breaks.

Interface (unchanged): load() -> obj with .encode(str)->list[int],
.decode(list[int])->str, .vocab_size.
"""
import json
import os
import re

_MERGES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "bpe_merges.json")
PRETOK = re.compile(r" ?[^\s]+|\s+")


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "byte"}, f)


class BPETokenizer:
    def __init__(self, merges):
        # merges: list of (a, b) byte/token id pairs, in learned order
        self.merges = [tuple(m) for m in merges]
        self.vocab_size = 256 + len(self.merges)
        # rank of each pair (lower = merge earlier)
        self.rank = {p: i for i, p in enumerate(self.merges)}
        # id -> raw bytes, for decoding
        self.tok_bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.tok_bytes.append(self.tok_bytes[a] + self.tok_bytes[b])

    def _encode_chunk(self, ids):
        while len(ids) >= 2:
            # find the mergeable adjacent pair with the best (lowest) rank
            best, best_rank = None, None
            for p in zip(ids, ids[1:]):
                r = self.rank.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            new_id = 256 + best_rank
            merged, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        return ids

    def encode(self, text):
        out = []
        for chunk in PRETOK.findall(text):
            out.extend(self._encode_chunk(list(chunk.encode("utf-8"))))
        return out

    def decode(self, ids):
        b = b"".join(self.tok_bytes[i] for i in ids)
        return b.decode("utf-8", errors="replace")


def load(path=None):
    path = path or _MERGES_FILE
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return BPETokenizer(data["merges"])
    return ByteTokenizer()
