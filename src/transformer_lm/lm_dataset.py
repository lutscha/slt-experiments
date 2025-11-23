# lm_dataset_new.py
import os
import torch
from datasets import load_dataset
from .vocab import build_vocab_from_iterator

DATASETS_FOLDER = os.environ["DATASETS"]


def tokenize(text):
    return text.strip().split()


def batchify(data_ids, batch_size):
    num_batches = data_ids.size(0) // batch_size
    data = data_ids[:num_batches * batch_size]
    data = data.view(batch_size, -1).t().contiguous()
    return data


def get_batch(source, i, bptt):
    seq_len = min(bptt, source.size(0) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+seq_len+1].reshape(-1)
    return data, target


def load_wikitext2(bptt=35, batch_size=20, min_freq=2):
    os.makedirs(DATASETS_FOLDER, exist_ok=True)
    cache_file = os.path.join(DATASETS_FOLDER, "wikitext2_cached.pt")

    if os.path.exists(cache_file):
        print("✓ Loading WikiText-2 from cache:", cache_file)
        saved = torch.load(cache_file)
        return (
            saved["train_data"],
            saved["valid_data"],
            saved["test_data"],
            saved["vocab"],
        )
    print("⚠ Cache not found. Downloading WikiText-2...")

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")

    # 1) Build vocab using YOUR class
    def token_iter():
        for split in ["train"]:
            for line in ds[split]["text"]:
                yield tokenize(line)

    vocab = build_vocab_from_iterator(token_iter(), min_freq=min_freq, specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    # 2) Encode using YOUR vocab safely
    def encode_split(split):
        out = []
        for line in ds[split]["text"]:
            toks = tokenize(line)
            out.extend(vocab.lookup_indices(toks))
        return torch.tensor(out, dtype=torch.long)

    train_ids = encode_split("train")
    valid_ids = encode_split("validation")
    test_ids  = encode_split("test")

    train_data = batchify(train_ids, batch_size)
    valid_data = batchify(valid_ids, batch_size)
    test_data  = batchify(test_ids, batch_size)

    # 3) SAFETY CHECK
    ntokens = len(vocab)
    assert train_data.max() < ntokens, "Train contains invalid token"
    assert valid_data.max() < ntokens, "Valid contains invalid token"
    assert test_data.max() < ntokens,  "Test contains invalid token"

    print("✓ Vocabulary OK")
    print("✓ Token IDs all within range")

    print("✓ Saving preprocessed data to:", cache_file)
    torch.save(
        {
            "train_data": train_data,
            "valid_data": valid_data,
            "test_data": test_data,
            "vocab": vocab,
        },
        cache_file,
    )

    return train_data, valid_data, test_data, vocab
