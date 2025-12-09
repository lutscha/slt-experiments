# lm_model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hessian_safe_modules import MyTransformerEncoderLayer


# -------------------------------------------------------
# Positional Encoding
# -------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len, 1, d_model)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# -------------------------------------------------------
# Transformer Language Model
# -------------------------------------------------------
class TransformerLM(nn.Module):
    def __init__(
        self,
        ntoken,
        ninp=200,
        nhead=2,
        nhid=200,
        nlayers=2,
        dropout=0.0
    ):
        super().__init__()

        self.ninp = ninp
        self.encoder = nn.Embedding(ntoken, ninp) #(batch, seq_lenght, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        # #This is wehere multihead attention is used, scaled_dot_product_attention does not support 2. derivatives
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=ninp,
        #     nhead=nhead,
        #     dim_feedforward=nhid,
        #     dropout=dropout,
        #     activation="relu",
        #     # batch_first=True,  #(Batch, seq_length)
        # )
        #Custom encoder layer that supports 2. derivative calculations
        encoder_layer = MyTransformerEncoderLayer(
                d_model=ninp,
                nhead=nhead,
                dim_feedforward=nhid,
                dropout=dropout,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=nlayers,
        )

        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, src, src_mask=None):
    # src expected shape: (seq, batch) check this with the True flag
        
        seq_len = src.size(0)
        if src_mask is None or src_mask.size(0) != seq_len: #Dynamical shaping
            src_mask = generate_square_subsequent_mask(seq_len).to(src.device)


        src = self.encoder(src) * math.sqrt(self.ninp) #Scale embeddings
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, mask=src_mask)
        output = self.decoder(output)

        return F.log_softmax(output, dim=-1)

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)
    return mask
