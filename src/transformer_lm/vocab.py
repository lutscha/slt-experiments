from collections import Counter

class Vocab:
    def __init__(self, counter, specials=["<unk>"]):
        self.specials = specials
        # Start with specials
        self.itos = list(specials)
        self.stoi = {tok: i for i, tok in enumerate(specials)}

        # Sort tokens by freq (descending), then alphabetically
        sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        for tok, _ in sorted_tokens:
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

        self.default_index = self.stoi["<unk>"]

    def __len__(self):
        return len(self.itos)

    def lookup_indices(self, tokens):
        return [self.stoi.get(t, self.default_index) for t in tokens]

    def lookup_tokens(self, indices):
        return [self.itos[i] for i in indices]

def build_vocab_from_iterator(iterator, specials=["<unk>"]):
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    return Vocab(counter, specials)