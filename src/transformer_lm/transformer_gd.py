from .lm_model import TransformerLM, generate_square_subsequent_mask
from .lm_dataset import load_wikitext2, get_batch

import os
import torch
import torch.nn as nn
import math
from torch.nn.utils import clip_grad_norm_
from utilities import get_hessian_eigenvalues
from typing import List, Tuple, Iterable
from utilities import save_files, save_files_final

def get_directory(dataset: str, lr: float, wd: float= 0.0):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]
    directory = f"{results_dir}/{dataset}/lr_{lr}/wd_{wd}"
    os.makedirs(directory, exist_ok=True)
    return directory

def train_one_epoch(model, train_data, optimizer, criterion, bptt, device):
    model.train()
    ntokens = model.decoder.out_features
    seq_len, batch_size = train_data.size()

    optimizer.zero_grad()
    total_loss = 0
    total_tokens = 0

    i = 0
    while i < seq_len - 1:
        data, targets = get_batch(train_data, i, bptt)
        data = data.to(device)
        targets = targets.to(device)

        # Forward
        output = model(data)
        loss = criterion(output.reshape(-1, ntokens), targets)

        # Backward (accumulated gradient)
        loss.backward()    # retain_graph=False by default → SAFE

        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()
        i += bptt

    # Now apply the full accumulated gradient
    clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    return total_loss / total_tokens



def evaluate(model, data_source, criterion, bptt, device):
    model.eval()
    total_loss = 0
    ntokens = model.decoder.out_features
    seq_len, batch_size = data_source.size()

    with torch.no_grad():
        i = 0
        while i < seq_len - 1:
            data, targets = get_batch(data_source, i, bptt)
            data = data.to(device)
            targets = targets.to(device)

            output = model(data)
            loss = criterion(output.reshape(-1, ntokens), targets)
            total_loss += loss.item()
            i += bptt

    return total_loss / ((seq_len - 1) // bptt)

def make_lm_dataset(data_source, bptt):
    seq_len, batch_size = data_source.size()
    dataset = []
    i = 0
    while i < seq_len - 1:
        X, Y = get_batch(data_source, i, bptt)
        dataset.append((X, Y))
        i += bptt
    return dataset


def run_training(neigs,
                 eig_freq,
                lr=0.1, 
                 epochs=5, 
                 bptt=35, 
                 batch_size=20, 
                 ninp=100, 
                 nhead=1, 
                 nhid=100, 
                 nlayers=1, 
                 wd = 0.0,
                 device="cuda",
                 seed=0,
                 save_model = False,
                 save_freq = -1):

    train_data, valid_data, test_data, vocab = load_wikitext2(bptt, batch_size)
    ntokens = len(vocab)

    
    save_dir = get_directory(dataset='wikitext2', lr=lr)
    train_loss, val_loss, train_acc, test_acc = \
        torch.zeros(epochs), torch.zeros(epochs), torch.zeros(epochs), torch.zeros(epochs)
    eigs = torch.zeros(epochs // eig_freq if eig_freq >= 0 else 0, neigs)

    torch.manual_seed(seed)
    model = TransformerLM(
        ntoken=ntokens,
        ninp=ninp,
        nhead=nhead,
        nhid=nhid,
        nlayers=nlayers,
        dropout=0.0
    ).to(device)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epochs):
        train_loss[epoch] = train_one_epoch(model, train_data, optimizer, criterion, bptt, device)
        val_loss[epoch] = evaluate(model, valid_data, criterion, bptt, device)
        print(f"Epoch {epoch} | Train NLL {train_loss[epoch]:.2f} | Val NLL {val_loss[epoch]:.2f}")


        if eig_freq > 0 and epoch % eig_freq == 0:
            print("  Computing Sharpness...")
            
            hessian_dataset = make_lm_dataset(train_data, bptt) #[(X,Y)] format
            print(f'hessian size: {hessian_dataset.size}')
            eigvals = get_hessian_eigenvalues(
                model, criterion, hessian_dataset[:1], neigs=neigs #Gives (35*batchsize) train examples for hessian compute
            )
            print(f'eig_size: {eigs.size}')
            print( f'index {epoch// eig_freq}')
            eigs[epoch // eig_freq] = eigvals
            print("  Top hessian eigenvalues:", eigvals.tolist())

        if save_freq != -1 and epoch % save_freq == 0:
            print('.....saving weights')
            save_files(save_dir, [("eigs", eigs[:epoch // eig_freq]),
                                   ("train_loss", train_loss[:epoch]), ("test_loss", val_loss[:epoch]),
                                  ])
            if save_model:
                print('.....saving model checkpoint')
                torch.save(model.state_dict(), f'{save_dir}/snapshot')

    test_loss = evaluate(model, test_data, criterion, bptt, device)
    print("Final test NLL:", test_loss)
    save_files_final(save_dir,
                    [("eigs", eigs[:epoch // eig_freq]),
                    ("train_loss", train_loss[:epoch]), ("test_loss", val_loss[:epoch])])
    
    if save_model:
        torch.save(model.state_dict(), f"{save_dir}/snapshot_final")