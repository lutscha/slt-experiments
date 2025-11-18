import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

# -------------------------------------------------------------
# Simple whitespace tokenizer
# -------------------------------------------------------------
def tokenize(text):
    return text.strip().split()

# -------------------------------------------------------------
# Convert flat token-id tensor into BPTT minibatches
# -------------------------------------------------------------
def batchify(data_ids, batch_size):
    # data_ids: 1D tensor [N]
    N = data_ids.size(0)
    n_batch = N // batch_size
    data = data_ids[: n_batch * batch_size]
    data = data.view(batch_size, -1).t()  # (batch, seq)
    return data

# -------------------------------------------------------------
# Produce (X, Y) batches using BPTT slicing
# -------------------------------------------------------------
def get_bptt_iter(data, bptt):
    
    seq_len, batch_size = data.size()
    for i in range(0, seq_len - 1, bptt):
        X = data[i : i + bptt, :]
        Y = data[i + 1 : i + 1 + bptt, :]
        yield X, Y

# -------------------------------------------------------------
# Load WikiText-2 and build vocab manually
# -------------------------------------------------------------
def load_wikitext2_lm(bptt=35, batch_size=20):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")

    # build vocab
    from .vocab import build_vocab_from_iterator

    def token_iter():
        for split in ["train", "validation", "test"]:
            for item in ds[split]["text"]:
                yield tokenize(item)

    vocab = build_vocab_from_iterator(token_iter(), min_freq=2, specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    # encode function
    def encode_split(split):
        ids = []
        for item in ds[split]["text"]:
            toks = tokenize(item)
            ids.extend(vocab.lookup_indices(toks))
        return torch.tensor(ids, dtype=torch.long)

    train_ids = encode_split("train")
    valid_ids = encode_split("validation")
    test_ids  = encode_split("test")

    print(f"Vocab size = {len(vocab)}")
    print(f"Train tokens = {len(train_ids)}")
    print(f"Test  tokens = {len(test_ids)}")

    # reshape into (batch, seq)
    train_data = batchify(train_ids, batch_size)
    valid_data = batchify(valid_ids, batch_size)
    test_data  = batchify(test_ids, batch_size)

    # return iterators for (X,Y)
    train_iter = list(get_bptt_iter(train_data, bptt))
    valid_iter = list(get_bptt_iter(valid_data, bptt))
    test_iter  = list(get_bptt_iter(test_data, bptt))
    

    print(f"Vocab size = {len(vocab)}")
    print(f"Train iter = {len(train_iter)}")
    print(f"Test  iter = {len(test_iter)}")

    print('-----------------------')
    return train_iter, valid_iter, test_iter, vocab
