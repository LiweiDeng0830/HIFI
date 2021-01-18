''' Define the Transformer model '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Models.Layers import EncoderLayer, DecoderLayer
from Models.FeatureInteraction import graph_constructor, mixprop

###################################################################
# The implementation of attention-based temporal module is based on https://github.com/jadore801120/attention-is-all-you-need-pytorch.
###################################################################

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.d_hid = d_hid
        self.n_position = n_position

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

    def __repr__(self):
        return "PositionalEncoding ({}, {})".format(self.d_hid, self.n_position)


class VEncoder(nn.Module):
    """Variational Encoder"""
    def __init__(
            self, ad_size, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_position=200, sequence_length=100,
            gcn_layers=2, gcn_alpha=0.2, k=2):

        super().__init__()

        self.ad_size = ad_size
        self.d_word_vec = d_word_vec
        self.dropout_rate = dropout

        # Multivariate Feature Interaction Module
        self.src_emb = nn.Linear(ad_size, d_word_vec)
        self.gc = graph_constructor(d_word_vec, d_k, k=k)
        self.mixgcn_left = mixprop(d_word_vec, d_word_vec, gcn_layers, gcn_alpha)
        self.mixgcn_right = mixprop(d_word_vec, d_word_vec, gcn_layers, gcn_alpha)
        self.gcn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.mu_layer = nn.Linear(d_model, d_model)
        self.mu_bn = nn.BatchNorm1d(sequence_length)
        self.mu_bn.weight.requires_grad = False
        # Batch Normalized is appended to Mean layer which is inspired from "A Batch Normalized Inference Network Keeps the KL Vanishing Away"
        self.logvar_layer = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward

        src_features = self.src_emb(src_seq)
        src_features = F.dropout(src_features, self.dropout_rate, training=self.training)
        adj = self.gc(torch.arange(self.d_word_vec).to(src_features.device))
        ho = self.mixgcn_left(src_features, adj) + self.mixgcn_right(src_features, adj.transpose(1, 0))
        src_features = ho + src_features
        src_features = self.gcn_layer_norm(src_features)

        enc_output = self.dropout(self.position_enc(src_features))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        mean = self.mu_bn(self.mu_layer(enc_output))
        logvar = self.logvar_layer(enc_output)

        if return_attns:
            return mean, logvar, enc_slf_attn_list
        return mean, logvar,

    def encode(self, src_seq, src_mask, return_attns=False):

        # batch_size, sequence_length, d_model
        mu, logvar = self.forward(src_seq, src_mask, return_attns)

        # batch_size, sequence_length, d_model
        # For now, number of samples are set to one. As shown in original variational autoencoder, one sample is enough.
        z = self.reparameterize(mu, logvar)

        # batch_size, sequence_length
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar -1).sum(-1)

        return z, KL

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = torch.zeros_like(std).normal_()
        return mu + torch.mul(eps, std)


class VDecoder(nn.Module):

    def __init__(
            self, ad_size, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_position=200, dropout=0.1,
            gcn_layers=2, gcn_alpha=0.2, k=2):

        super().__init__()

        self.ad_size = ad_size
        self.dropout_rate = dropout
        self.d_word_vec = d_word_vec
        self.src_emb = nn.Linear(ad_size, d_word_vec)

        # Multivariate Feature Interaction Module
        self.gc = graph_constructor(d_word_vec, d_k, k=k)
        self.mixgcn_left = mixprop(d_word_vec, d_word_vec, gcn_layers, gcn_alpha)
        self.mixgcn_right = mixprop(d_word_vec, d_word_vec, gcn_layers, gcn_alpha)
        self.gcn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.trg_emb = nn.Linear(ad_size, d_word_vec)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        trg_features = self.trg_emb(trg_seq)
        trg_features = F.dropout(trg_features, self.dropout_rate, training=self.training)
        adj = self.gc(torch.arange(self.d_word_vec).to(trg_features.device))
        ho = self.mixgcn_left(trg_features, adj) + self.mixgcn_right(trg_features, adj.transpose(1, 0))
        trg_features = ho + trg_features
        trg_features = self.gcn_layer_norm(trg_features)

        # -- Forward
        dec_output = self.dropout(self.position_enc(trg_features))

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        dec_output = self.layer_norm(dec_output)

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class HIFI(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, ad_size,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200, sequence_length=100,
            gcn_layers=2, gcn_alpha=0.2, k=2):

        super().__init__()

        self.encoder = VEncoder(
            ad_size=ad_size,
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout,
            sequence_length=sequence_length,
            gcn_layers=gcn_layers, gcn_alpha=gcn_alpha, k=k)

        self.decoder = VDecoder(
            ad_size=ad_size,
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        # Map the encoded hidden representation to input space
        self.decoder_to_input = nn.Linear(d_model, ad_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, src_seq, trg_seq):

        src_mask = torch.ones(src_seq.shape[0], src_seq.shape[1]).to(src_seq.device).unsqueeze(-2)
        trg_mask = torch.ones(src_seq.shape[0], src_seq.shape[1]).to(src_seq.device)
        trg_mask = get_subsequent_mask(trg_mask)

        enc_output, KL = self.encoder.encode(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        reconstruct_input = self.decoder_to_input(dec_output)

        return reconstruct_input, KL

