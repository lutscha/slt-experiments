from collections import Counter

class Vocab:
    def __init__(self, counter, min_freq=1, specials=["<unk>"]):
        self.specials = specials
        self.itos = []
        self.stoi = {}

        # Add specials first
        for sp in specials:
            self.stoi[sp] = len(self.itos)
            self.itos.append(sp)

        # Filter tokens by min_freq and sort like torchtext
        sorted_tokens = sorted(
            [(tok, freq) for tok, freq in counter.items() if freq >= min_freq],
            key=lambda x: (-x[1], x[0])
        )

        # Add tokens to vocab
        for tok, _ in sorted_tokens:
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

        # default index mechanism
        self.default_index = self.stoi.get("<unk>", None)

    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)

    def __len__(self):
        return len(self.itos)

    def set_default_index(self, idx):
        self.default_index = idx

    def lookup_indices(self, tokens):
        return [self[token] for token in tokens]

    def lookup_tokens(self, indices):
        return [self.itos[i] for i in indices]


def build_vocab_from_iterator(iterator, min_freq=1, specials=["<unk>"]):
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    return Vocab(counter, min_freq=min_freq, specials=specials)