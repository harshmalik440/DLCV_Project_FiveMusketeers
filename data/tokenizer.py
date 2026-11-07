"""
Lightweight, reproducible tokenizer for Sign Language Translation datasets.
Supports special tokens (<pad>, <s>, </s>, <unk>), vocabulary construction,
and bidirectional encoding/decoding.
"""

class SimpleSignTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        self.token2id = {t: idx for idx, t in enumerate(self.special_tokens)}
        self.id2token = {idx: t for idx, t in enumerate(self.special_tokens)}

    def build_vocab(self, texts):
        idx = len(self.special_tokens)
        for t in texts:
            words = t.strip().lower().split()
            for w in words:
                if w not in self.token2id:
                    self.token2id[w] = idx
                    self.id2token[idx] = w
                    idx += 1

    def encode(self, text, max_len=64):
        words = text.strip().lower().split()
        ids = [self.token2id.get(self.bos_token)]
        for w in words:
            ids.append(self.token2id.get(w, self.token2id[self.unk_token]))
        ids.append(self.token2id.get(self.eos_token))

        if len(ids) < max_len:
            ids = ids + [self.token2id[self.pad_token]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            t = self.id2token.get(int(i), self.unk_token)
            if t == self.eos_token:
                break
            if t not in [self.pad_token, self.bos_token]:
                tokens.append(t)
        return " ".join(tokens)

    @property
    def vocab_size(self):
        return len(self.token2id)

