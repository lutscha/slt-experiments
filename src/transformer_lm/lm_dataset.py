# lm_dataset.py
import torch
from datasets import load_dataset
from collections import Counter

# ------------------------------------------
# Load WikiText-2 using HuggingFace
# ------------------------------------------
def load_raw_wikitext2():
    ds = load_dataset("wikitext", "wikitext-2-v1")
    train = "\n".join(ds["train"]["text"])
    valid = "\n".join(ds["validation"]["text"])
    test = "\n".join(ds["test"]["text"])
    return train, valid, test


# ------------------------------------------
# Build vocabulary
# ------------------------------------------
class Vocab:
    def __init__(self, tokens, min_freq=1):
        counter = Counter(tokens)
        self.itos = ["<unk>", "<pad>"]
        self.itos += [tok for tok, c in counter.items() if c >= min_freq]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, 0) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


# ------------------------------------------
# Tokenization (simple whitespace)
# ------------------------------------------
def tokenize(text):
    return text.strip().split()


# ------------------------------------------
# Create tensor dataset
# ------------------------------------------
def make_tensor_dataset(text, vocab):
    tokens = tokenize(text)
    ids = vocab.encode(tokens)
    return torch.tensor(ids, dtype=torch.long)


# ------------------------------------------
# BPTT batch maker
# ------------------------------------------
def batchify(data, batch_size):
    # Divide data into batch_size columns
    seq_len = data.size(0) // batch_size
    data = data[: seq_len * batch_size]
    data = data.view(batch_size, seq_len).t().contiguous()
    return data


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target


# ------------------------------------------
# Main loader
# ------------------------------------------
def load_wikitext2_lm(bptt=35, batch_size=20):
    # 1. Load raw text
    train_txt, valid_txt, test_txt = load_raw_wikitext2()

    # 2. Build vocab from training data only
    train_tokens = tokenize(train_txt)
    vocab = Vocab(train_tokens)

    # 3. Convert splits to tensors
    train_ids = make_tensor_dataset(train_txt, vocab)
    valid_ids = make_tensor_dataset(valid_txt, vocab)
    test_ids  = make_tensor_dataset(test_txt, vocab)

    # 4. Batchify for LM training
    train_data = batchify(train_ids, batch_size)
    valid_data = batchify(valid_ids, batch_size)
    test_data  = batchify(test_ids, batch_size)

    return train_data, valid_data, test_data, vocab
