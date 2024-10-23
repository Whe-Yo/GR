# region Import.

import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import List


import random
from math import ceil
from functools import partial
from itertools import zip_longest

import torch
from torch import nn
import torch.nn.functional as F
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize

from einops import rearrange, repeat, reduce, pack, unpack

# endregion


__all__ = ["Conv1d", "Transformer", "VQVAE_gcn", "VQVAE_conv", "RVQ_gcn", "RVQ_conv", "autoencoder_conv", "VAE_conv"]

def contrastive_loss(emb_idxs, temperature=0.07):
    # Calculate pairwise cosine similarity between embedding vectors
    similarity_matrix = F.cosine_similarity(emb_idxs.unsqueeze(0), emb_idxs.unsqueeze(1), dim=-1)

    # Diagonal elements (similarity to self) should be excluded from positive pairs
    eye = torch.eye(emb_idxs.size(0), dtype=torch.bool).to(emb_idxs.device)
    pos_pairs = similarity_matrix[~eye].view(emb_idxs.size(0), -1)  # Positive pairs

    # Negative pairs: all pairs excluding positive pairs
    neg_pairs = similarity_matrix[eye == 0].view(emb_idxs.size(0), -1)  # Negative pairs

    # Contrastive loss: maximize similarity for positive pairs and minimize similarity for negative pairs
    logits = torch.cat([pos_pairs, neg_pairs], dim=1) / temperature
    labels = torch.zeros(emb_idxs.size(0), dtype=torch.long).to(emb_idxs.device)  # 0 for positive pairs, 1 for negative pairs
    loss = F.cross_entropy(logits, labels)

    return loss


class Conv1d(nn.Module):
    def __init__(self,
                 encoder_config: List[List[int]],
                 decoder_config: List[List[int]]) -> None:
        super().__init__()

        num_layers = len(encoder_config)

        modules = []
        for i, c in enumerate(encoder_config):
            modules.append(nn.Conv1d(in_channels=c[0], out_channels=c[1], kernel_size=c[2], stride=c[3], padding=c[4]))

            if i < (num_layers - 1):
                modules.append(nn.BatchNorm1d(c[1]))
                modules.append(nn.LeakyReLU(1.0, inplace=True))

        self.encoder = nn.Sequential(*modules)

        num_layers = len(decoder_config)

        modules = []
        for i, c in enumerate(decoder_config):
            modules.append(nn.ConvTranspose1d(in_channels=c[0], out_channels=c[1], kernel_size=c[2], stride=c[3], padding=c[4]))

            if i < (num_layers - 1):
                modules.append(nn.BatchNorm1d(c[1]))
                modules.append(nn.LeakyReLU(1.0, inplace=True))

        self.decoder = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor):
        """
        x : (batch_size, dim_feat, time).
        """

        latent_code = self.encoder(x)

        return latent_code, self.decoder(latent_code), 0


# --------------------------------------------------------
# Reference: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
# --------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100) -> None:
        super().__init__()

        assert d_model % 2 == 0

        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.pe.data[0, :, 0::2].copy_(torch.sin(position * div_term))
        self.pe.data[0, :, 1::2].copy_(torch.cos(position * div_term))

    def forward(self, x) -> torch.Tensor:
        """
        x: [N, L, D]
        """
        x = x + self.pe[:, :x.shape[1], :]

        return self.dropout(x)


# --------------------------------------------------------
# Reference: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, mo_dim, lxm_dim,
                 embed_dim=512, depth=6, num_heads=8,
                 decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=8,
                 mlp_ratio=4, activation='gelu', norm_layer=nn.LayerNorm, dropout=0.1) -> None:
        super().__init__()

        # --------------------------------------------------------------------------
        # encoder specifics
        self.encoder_embed = nn.Linear(mo_dim, embed_dim, bias=True)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = PositionalEncoding(embed_dim, dropout)  # hack: max len of position is 100

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim,
                                                                        nhead=num_heads,
                                                                        dim_feedforward=int(mlp_ratio * embed_dim),
                                                                        dropout=dropout,
                                                                        activation=activation,
                                                                        batch_first=True),
                                             num_layers=depth)

        self.norm = norm_layer(embed_dim)
        self.lxm_embed = nn.Linear(embed_dim, lxm_dim, bias=True)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # decoder specifics
        self.decoder_embed = nn.Linear(lxm_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = PositionalEncoding(decoder_embed_dim, dropout)  # hack: max len of position is 100

        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=decoder_embed_dim,
                                                                        nhead=decoder_num_heads,
                                                                        dim_feedforward=int(mlp_ratio * decoder_embed_dim),
                                                                        dropout=dropout,
                                                                        activation=activation,
                                                                        batch_first=True),
                                             num_layers=decoder_depth)

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, mo_dim, bias=True)
        # --------------------------------------------------------------------------

        self.initialize_weights()

        # --------------------------------------------------------------------------
        self.codebook = VectorQuantizer_gcn(num_embeddings=50,
                                        embedding_dim=192,
                                        beta=0.25)


    def initialize_weights(self):
        # initialization
        # initialize nn.Parameter
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embed.pe, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed.pe, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)

    def forward_encoder(self, x):
        """
        x: [N, D, L]
        """
        x = torch.einsum('NDL->NLD', x)

        # embed motion sequence
        x = self.encoder_embed(x)  # (N, L, embed_dim)

        # append cls token
        cls_token = self.cls_token.repeat(x.shape[0], x.shape[1], 1)  # (
        x = torch.cat([cls_token, x], dim=1)

        # add pos embed
        x = self.pos_embed(x)

        # apply Transformer blocks
        x = self.encoder(x)
        x = self.norm(x)
        x = self.lxm_embed(x)

        lxm = torch.einsum('NLD->NDL', x[:, :1, :])

        return lxm

    def forward_decoder(self, lxm, mo_len):
        '''
        lxm: [N, D, L]
        '''
        lxm = torch.einsum('NDL->NLD', lxm)

        # embed lexeme
        x = self.decoder_embed(lxm)

        # append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], mo_len, 1)
        x = torch.cat([x, mask_tokens], dim=1)

        # add pos embed
        x = self.decoder_pos_embed(x)

        # add Transformer blocks
        x = self.decoder(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove lxm token
        x = x[:, 1:, :]

        x = torch.einsum('NLD->NDL', x)

        return x

    def forward(self, x):
        """
        x: [N, D, L]
        """
        _, _, L = x.shape  # L = 10

        z_e = self.forward_encoder(x)

        z_e = z_e.squeeze(2)

        # z_q, indices, embedding_loss, qq_loss = self.vector_quantization(z_e)
        z_q, indices, embedding_loss, qq_loss = self.codebook(z_e, None)

        z_q = z_q.unsqueeze(2)

        x_hat = self.forward_decoder(z_q, L)

        indices = []
        # vq_loss = torch.tensor(0.0)
        # qq_loss = torch.tensor(0.0)

        return z_q, x_hat, indices, embedding_loss, qq_loss

    def forward_forlxm(self, lxm):
        """
        x: [N, D, L]
        """
        L = 10
        x_hat = self.forward_decoder(lxm, L)

        return lxm, x_hat, 0


# --------------------------------------------------------
# Reference: https://github.com/MishaLaskin/vqvae/blob/master/models/vqvae.py
# --------------------------------------------------------

from modules import StgcnBlock, SpatialDownsample, SpatialUpsample, GraphJoint, ResBlock


class Encoder_gcn(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden

        self.stgcn1 = StgcnBlock(dim_in=dim_in, dim_out=dim_hidden, kernel_size=(7, 3), padding=(3, 2), stride=(1, 1))
        self.t_down1 = nn.AvgPool2d(kernel_size=(2, 1))
        self.s_down1 = SpatialDownsample(0)
        self.bn1 = nn.BatchNorm2d(dim_hidden, affine=True)

        self.stgcn2 = StgcnBlock(dim_in=dim_hidden, dim_out=dim_hidden, kernel_size=(5, 2), padding=(2, 1), stride=(1, 1))
        self.t_down2 = nn.AvgPool2d(kernel_size=(2, 1))
        self.s_down2 = SpatialDownsample(1)
        self.bn2 = nn.BatchNorm2d(dim_hidden, affine=True)

        self.stgcn3 = StgcnBlock(dim_in=dim_hidden, dim_out=dim_hidden, kernel_size=(5, 2), padding=(2, 1), stride=(1, 1))

        # self.res = ResBlock(dim_in=dim_hidden, dim_out=dim_hidden, kernel_size=(3, 2), padding=(1, 1), stride=(1, 1))

        self.down = nn.AvgPool2d(kernel_size=(2, 4))
        # self.down = nn.AvgPool2d(kernel_size=(15, 4))
        self.fc = nn.Linear(dim_hidden, dim_out)

        self.actv = nn.ReLU()

        # 인접 joint의 그래프를 만들어주는 코드
        self.graph_top = GraphJoint(hierarchy_index=0)
        # adj_top : (3, 16, 16)
        adj_top = torch.tensor(self.graph_top.adj_mat, dtype=torch.float32, requires_grad=False)
        self.register_buffer('adj_top', adj_top)
        # joint의 그래프를 학습 시 업데이트 값으로 취급
        self.edge_importance_top = nn.Parameter(torch.ones(self.adj_top.size()))

        self.graph_mid = GraphJoint(hierarchy_index=1)
        adj_mid = torch.tensor(self.graph_mid.adj_mat, dtype=torch.float32, requires_grad=False)
        self.register_buffer('adj_mid', adj_mid)
        self.edge_importance_mid = nn.Parameter(torch.ones(self.adj_mid.size()))

        self.graph_bot = GraphJoint(hierarchy_index=2)
        adj_bot = torch.tensor(self.graph_bot.adj_mat, dtype=torch.float32, requires_grad=False)
        self.register_buffer('adj_bot', adj_bot)
        self.edge_importance_bot = nn.Parameter(torch.ones(self.adj_bot.size()))

    def forward(self, x):
        # print(x.shape)              # B R T(10) J(16)
        x_encode = self.stgcn1(x, self.adj_top * self.edge_importance_top) # B 256 T J
        x_encode = self.actv(x_encode)
        x_encode = self.t_down1(x_encode) # B 256 0.5*t J
        x_encode = self.s_down1(x_encode) # B 256 0.5*t 0.5*J
        x_encode = self.bn1(x_encode)

        x_encode = self.stgcn2(x_encode, self.adj_mid * self.edge_importance_mid) # B 256 0.5*t 0.5*J
        x_encode = self.actv(x_encode)
        x_encode = self.t_down2(x_encode) # B 256 0.25*t 0.5*J
        x_encode = self.s_down2(x_encode) # B 256 0.25*t 0.25*J
        x_encode = self.bn2(x_encode)

        x_encode = self.stgcn3(x_encode, self.adj_bot * self.edge_importance_bot) # B 256 0.25*t 0.25*J
        x_encode = self.actv(x_encode) # B 256 0.25*t 0.25*J

        x_encode = self.down(x_encode)   # B 256 1 1
        x_encode = x_encode.view(x_encode.shape[0], -1) # B 256
        x_encode = self.fc(x_encode) # B 192

        return x_encode


class Decoder_gcn(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden

        self.fc = nn.Linear(dim_in, dim_hidden)
        # self.up = nn.Upsample(scale_factor=(15, 4), mode='nearest')
        self.up = nn.Upsample(scale_factor=(2, 4), mode='nearest')
        self.up2 = nn.Upsample(size=(10, 16), mode='bilinear', align_corners=False)

        # self.res = ResBlock(dim_in=dim_hidden, dim_out=dim_hidden, kernel_size=(3, 2), padding=(1, 1), stride=(1, 1))

        self.stgcn1 = StgcnBlock(dim_in=dim_hidden, dim_out=dim_hidden, kernel_size=(5, 2), padding=(2, 1), stride=(1, 1))
        self.t_up1 = nn.Upsample(scale_factor=(2, 1), mode='nearest')
        self.s_up1 = SpatialUpsample(hierarchy_index=0)
        self.bn1 = nn.BatchNorm2d(dim_hidden, affine=True)

        self.stgcn2 = StgcnBlock(dim_in=dim_hidden, dim_out=dim_hidden, kernel_size=(5, 2), padding=(2, 1), stride=(1, 1))
        self.t_up2 = nn.Upsample(scale_factor=(2, 1), mode='nearest')
        self.s_up2 = SpatialUpsample(hierarchy_index=1)
        self.bn2 = nn.BatchNorm2d(dim_hidden, affine=True)

        self.stgcn3 = StgcnBlock(dim_in=dim_hidden, dim_out=dim_out, kernel_size=(7, 3), padding=(3, 2), stride=(1, 1))

        self.actv = nn.ReLU()

        self.graph_top = GraphJoint(hierarchy_index=0)
        adj_top = torch.tensor(self.graph_top.adj_mat, dtype=torch.float32, requires_grad=False)
        self.register_buffer('adj_top', adj_top)
        self.edge_importance_top = nn.Parameter(torch.ones(self.adj_top.size()))

        self.graph_mid = GraphJoint(hierarchy_index=1)
        adj_mid = torch.tensor(self.graph_mid.adj_mat, dtype=torch.float32, requires_grad=False)
        self.register_buffer('adj_mid', adj_mid)
        self.edge_importance_mid = nn.Parameter(torch.ones(self.adj_mid.size()))

        self.graph_bot = GraphJoint(hierarchy_index=2)
        adj_bot = torch.tensor(self.graph_bot.adj_mat, dtype=torch.float32, requires_grad=False)
        self.register_buffer('adj_bot', adj_bot)
        self.edge_importance_bot = nn.Parameter(torch.ones(self.adj_bot.size()))

        self.tcn = nn.Conv2d(dim_out, dim_out, kernel_size=(7, 1), padding=(3, 0), stride=(1, 1))

    def forward(self, x):  # B, C -> B, JR, T
        x_decode = self.actv(self.fc(x))  # tensor(100, 192)
        x_decode = x_decode.view(x_decode.shape[0], -1, 1, 1)  # tensor(100, 256)
        x_decode = self.up(x_decode)  # tensor(100, 256, 2, 4)

        # x_decode = self.res(x_decode, self.adj_bot * self.edge_importance_bot)

        x_decode = self.stgcn1(x_decode, self.adj_bot * self.edge_importance_bot)  # tensor(100, 256, 2, 4)
        x_decode = self.actv(x_decode)  # tensor(100, 256, 2, 4)
        x_decode = self.t_up1(x_decode)  # tensor(100, 256, 4, 4)
        x_decode = self.s_up1(x_decode)  # tensor(100, 256, 4, 8)
        x_decode = self.bn1(x_decode)  # tensor(100, 256, 4, 8)

        x_decode = self.stgcn2(x_decode, self.adj_mid * self.edge_importance_mid)  # tensor(100, 256, 4, 8)
        x_decode = self.actv(x_decode)  # tensor(100, 256, 4, 8)
        x_decode = self.t_up2(x_decode)  # tensor(100, 256, 8, 8)
        x_decode = self.s_up2(x_decode)  # tensor(100, 256, 8, 16)
        x_decode = self.bn2(x_decode)  # tensor(100, 256, 8, 16)

        x_decode = self.stgcn3(x_decode, self.adj_top * self.edge_importance_top)  # tensor(100, 3, 8, 16)
        x_decode = self.actv(x_decode)  # tensor(100, 3, 8, 16)
        x_decode = self.up2(x_decode)

        x_decode = self.tcn(x_decode)  # tensor(100, 3, 8, 16), but must be same with input of encoder

        return x_decode


class VectorQuantizer_gcn(nn.Module):
    def __init__(self, num_embeddings=50, embedding_dim=192, beta=0.25):
        super(VectorQuantizer_gcn, self).__init__()
        self.device = torch.device("cuda:{}".format(torch.cuda.current_device()))

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embed = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # self.embed.weight.data.normal_()
        # self.embed.weight.data.uniform_(-1, 1)
        self.embed.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.ema_w = nn.Parameter(torch.Tensor(num_embeddings, self.embedding_dim))
        self.ema_w.data.normal_()

        self.decay = 0.99
        self.epsilon = 1e-5

        self.beta = beta

        self.contloss = contrastive_loss

    def forward(self, inputs, index):
        flat_input = inputs.view(-1, self.embedding_dim)  # tensor(100, 192)

        # Compute distances to embedding vectors
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embed.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embed.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # tensor(100, 1)
        if index != None:
            encoding_indices = torch.LongTensor([index] * self.batch_size).to(self.device).unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)  # tensor(100, 200)
        encodings.scatter_(1, encoding_indices, 1)

        # 4. Quantize and unflatten
        quantized = torch.matmul(encodings, self.embed.weight)  # tensor(100, 192)

        indices = torch.unique(encoding_indices)
        # emb_idxs = self.embed.weight[indices]

        # qqloss
        # contrastloss = contrastive_loss(emb_idxs)

        # loss = nn.functional.mse_loss(quantized.detach(), inputs) * self.beta   # Vector quantization objective
        commitloss = torch.mean((quantized.detach()-inputs)**2) + self.beta * torch.mean((quantized - inputs.detach())**2)
        # commitloss = F.mse_loss(inputs, quantized.detach()) + self.beta * F.mse_loss(inputs.detach(), quantized)
        # commitloss = ((inputs.detach() - quantized) ** 2).mean() + ((inputs - quantized.detach()) ** 2).mean()
        # loss += F.mse_loss(quantized, inputs.detach())          # Commitment objective

        quantized = inputs + (quantized - inputs).detach()  # tensor(100, 192)

        contrastloss = torch.tensor(0.0)

        return quantized, encoding_indices, commitloss, contrastloss



def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        *,
        dim,
        num_quantizers,
        beta,
        vector_quantize,
        codebook_dim = None,
        shared_codebook = False,
        heads = 1,
        quantize_dropout = False,
        quantize_dropout_cutoff_index = 0,
        quantize_dropout_multiple_of = 1,
        accept_image_fmap = False,
        **kwargs):

        super().__init__()
        assert heads == 1, 'residual vq is not compatible with multi-headed codes'
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.num_quantizers = num_quantizers

        self.accept_image_fmap = accept_image_fmap

        # gcn, conv의 vq 적용
        # self.layers = nn.ModuleList([VectorQuantize(dim = codebook_dim, codebook_dim = codebook_dim, accept_image_fmap = accept_image_fmap, **kwargs) for _ in range(num_quantizers)])
        self.layers = nn.ModuleList([vector_quantize for _ in range(num_quantizers)])

        # assert all([not vq.has_projections for vq in self.layers])

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

        if not shared_codebook:
            return

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            vq._codebook = codebook

    @property
    def codebooks(self):
        codebooks = [layer._codebook.embed for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        codebooks = rearrange(codebooks, 'q 1 c d -> q c d')
        return codebooks

    def get_codes_from_indices(self, indices):

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)

        indices, ps = pack([indices], 'b * q')

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0., 'quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations'
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # get ready for gathering

        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b = batch)
        gather_indices = repeat(indices, 'b n q -> q b n d', d = codebooks.shape[-1])

        # take care of quantizer dropout

        mask = gather_indices == -1.
        gather_indices = gather_indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        all_codes = codebooks.gather(2, gather_indices) # gather all codes

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(mask, 0.)

        # if (accept_image_fmap = True) then return shape (quantize, batch, height, width, dimension)

        all_codes, = unpack(all_codes, ps, 'q b * d')

        return all_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def forward(
        self,
        x,
        mask = None,
        indices = None,
        return_all_codes = False,
        sample_codebook_temp = None,
        rand_quantize_dropout_fixed_seed = None):

        num_quant, quant_dropout_multiple_of, return_loss, device = self.num_quantizers, self.quantize_dropout_multiple_of, exists(indices), x.device

        x = self.project_in(x)

        assert not (self.accept_image_fmap and exists(indices))

        quantized_out = 0.
        residual = x

        all_vqlosses = []
        all_qqlosses = []
        all_indices = []

        if return_loss:
            assert not torch.any(indices == -1), 'some of the residual vq indices were dropped out. please use indices derived when the module is in eval mode to derive cross entropy loss'
            ce_losses = []

        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices and loss

        if should_quantize_dropout:
            rand = random.Random(rand_quantize_dropout_fixed_seed) if exists(rand_quantize_dropout_fixed_seed) else random

            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1

            null_indices_shape = (x.shape[0], *x.shape[-2:]) if self.accept_image_fmap else tuple(x.shape[:2])
            null_indices = torch.full(null_indices_shape, -1., device = device, dtype = torch.long)
            null_vqloss = torch.full((1,), 0., device = device, dtype = x.dtype)
            null_qqloss = torch.full((1,), 0., device = device, dtype = x.dtype)

        # go through the layers

        for quantizer_index, layer in enumerate(self.layers):

            if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_vqlosses.append(null_vqloss)
                all_qqlosses.append(null_qqloss)
                continue

            layer_indices = None
            if return_loss:
                layer_indices = indices[..., quantizer_index]

            # quantized, *rest = layer(
            #     residual,
            #     mask = mask,
            #     indices = layer_indices,
            #     sample_codebook_temp = sample_codebook_temp,
            # )

            quantized, *rest = layer(residual)

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            if return_loss:
                ce_loss = rest[0]
                ce_losses.append(ce_loss)
                continue

            embed_indices, vqloss, qqloss = rest

            all_indices.append(embed_indices)
            all_vqlosses.append(vqloss)
            all_qqlosses.append(qqloss)

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # whether to early return the cross entropy loss

        if return_loss:
            return quantized_out, sum(ce_losses)

        # stack all losses and indices

        all_vqlosses, all_qqlosses, all_indices = map(partial(torch.stack, dim = -1), (all_vqlosses, all_qqlosses, all_indices))

        ret = (quantized_out, all_indices, all_vqlosses, all_qqlosses)

        if return_all_codes:
            # whether to return all codes from all codebooks across layers
            all_codes = self.get_codes_from_indices(all_indices)

            # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)
            ret = (*ret, all_codes)

        return ret



class VQVAE_gcn(nn.Module):
    def __init__(self, mo_dim, feature_size, lexicon_size, rotation_size, joint_size, beta) -> None:
        super().__init__()

        self.encoder = Encoder_gcn(dim_in=rotation_size,
                               dim_out=feature_size,
                               dim_hidden=1024)
        self.decoder = Decoder_gcn(dim_in=feature_size,
                               dim_out=rotation_size,
                               dim_hidden=1024)
        self.codebook = VectorQuantizer_gcn(num_embeddings=lexicon_size,
                                        embedding_dim=feature_size,
                                        beta=beta)
        #
        # self.codebook = VectorQuantizer_conv(lexicon_size, feature_size, beta)

        self.tanh = nn.Tanh()

        self.rotation_size = rotation_size
        self.joint_size = joint_size

    def forward(self, input_data, index=None):
        """
        VQVAE class 는 세 가지 단계로 크게 구성됨
        Encoder, Codebook, Decoder
        위의 순서로 feature 가 전달됨
        Args:
            input_data: input data
            index: 만약 test 를 돌릴 때, index 를 지정하면 codebook 에서 해당 index 에 해당하는 gesture 의 vector 값만 나오도록 강제할 수 있음
        Returns:
            x_rec: recon 한 motion
            vq_loss: vq 에 대한 loss
            z_q: codebook 에서 뽑힌 code 값
            indices: codebook 에서 뽑힌 code 의 index 값
        """
        x = input_data  # B x JR X T, T = L JR = D

        x = x.view(x.shape[0], self.rotation_size, x.shape[2], self.joint_size) # B x R x T x J

        z = self.encoder(x)  # B, C
        # z = self.encoder(x)  # B, C
        z_q, indices, embedding_loss, qq_loss = self.codebook(z, index)  # B, C
        # z_q, indices, embedding_loss, qq_loss = self.codebook(z)
        x_rec = self.decoder(z_q)  # B x R x T x J

        x_rec = self.tanh(x_rec)
        x_rec = x_rec.view(x_rec.shape[0], (self.joint_size * self.rotation_size), x_rec.shape[2]).contiguous()

        z_q = z_q.unsqueeze(2)

        indices = []
        embedding_loss = torch.tensor(0.0)
        qq_loss = torch.tensor(0.0)

        return z_q, x_rec, indices, embedding_loss, qq_loss

    def forward_forlxm(self, z_q):
        x_rec = self.decoder(z_q)  # B x R x T x J

        x_rec = self.tanh(x_rec)
        x_rec = x_rec.view(x_rec.shape[0], (self.joint_size * self.rotation_size), x_rec.shape[2]).contiguous()

        z_q = z_q.unsqueeze(2)

        return z_q, x_rec


class RVQ_gcn(nn.Module):
    def __init__(self, mo_dim, feature_size, lexicon_size, rotation_size, joint_size, beta, num_quantizers) -> None:
        super().__init__()

        self.encoder = Encoder_gcn(dim_in=rotation_size,
                               dim_out=feature_size,
                               dim_hidden=256, )
        self.decoder = Decoder_gcn(dim_in=feature_size,
                               dim_out=rotation_size,
                               dim_hidden=267, )
        self.rvq = ResidualVQ(dim = feature_size, num_quantizers = num_quantizers, codebook_size = lexicon_size, beta = beta,
                              vector_quantize = VectorQuantizer_gcn(num_embeddings=lexicon_size, embedding_dim=feature_size, beta=beta))

        self.tanh = nn.Tanh()

        self.rotation_size = rotation_size
        self.joint_size = joint_size

    def forward(self, input_data, index=None):
        """
        VQVAE class 는 세 가지 단계로 크게 구성됨
        Encoder, Codebook, Decoder
        위의 순서로 feature 가 전달됨
        Args:
            input_data: input data
            index: 만약 test 를 돌릴 때, index 를 지정하면 codebook 에서 해당 index 에 해당하는 gesture 의 vector 값만 나오도록 강제할 수 있음
        Returns:
            x_rec: recon 한 motion
            vq_loss: vq 에 대한 loss
            z_q: codebook 에서 뽑힌 code 값
            indices: codebook 에서 뽑힌 code 의 index 값
        """
        x = input_data  # B x JR X T, T = L JR = D

        x = x.permute(0, 2, 1)  # B x T x JR
        x = x.view(x.shape[0], x.shape[1], self.joint_size, self.rotation_size)  # B x T x J x R
        x = x.permute(0, 3, 1, 2).contiguous()  # B x R x T x J

        z = self.encoder(x)  # B, C
        z_q, indices, vq_loss_each, qq_loss_each = self.rvq(z) # B, C
        vq_loss = torch.sum(vq_loss_each)
        x_rec = self.decoder(z_q)  # B x R x T x J
        x_rec = x_rec.permute(0, 2, 3, 1).contiguous()  # B x T x J x R

        x_rec = self.tanh(x_rec)
        x_rec = x_rec.view(x_rec.shape[0], x_rec.shape[1], (self.joint_size * self.rotation_size))
        x_rec = x_rec.permute(0, 2, 1)

        z_q = z_q.unsqueeze(2)

        qq_loss = 0
        return z_q, x_rec, indices, vq_loss, qq_loss

    def forward_forlxm(self, z_q):
        z_q = z_q.permute((0, 2, 1)).contiguous()
        z_q = z_q.squeeze(0)

        x_rec = self.decoder(z_q)  # B x R x T x J
        x_rec = x_rec.permute(0, 2, 3, 1).contiguous()  # B x T x J x R

        x_rec = self.tanh(x_rec)
        x_rec = x_rec.view(x_rec.shape[0], x_rec.shape[1], (self.joint_size * self.rotation_size))
        x_rec = x_rec.permute(0, 2, 1)

        vq_loss = 0

        x_rec = x_rec.reshape(1, x_rec.shape[0] * 10, x_rec.shape[1])

        return z_q, x_rec, vq_loss




# --------------------------------------------------------
# Reference: https://github.com/MishaLaskin/vqvae/blob/master/models/vqvae.py
# --------------------------------------------------------

class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        residual = x
        x = self.relu1(x)  # 1000 128 2
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = x + residual
        return x



class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = nn.functional.relu(x)
        return x

class Encoder_conv(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, out_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder_conv, self).__init__()
        kernel = 4
        stride = 2

        # 첫 번째 Conv1d: 채널 수를 in_channels에서 h_dim으로 변경
        self.conv1 = nn.Conv1d(in_dim, h_dim//4, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(h_dim//4, h_dim//2, kernel_size=kernel, stride=stride)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(h_dim//2, h_dim, kernel_size=kernel, stride=stride)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(h_dim, out_dim, kernel_size=1, stride=1)

        # # ResidualStack
        self.residual_stack = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)


    def forward(self, x):

        # temporal
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)

        return x

class VectorQuantizer_conv(nn.Module):
    def __init__(self, n_e, e_dim, beta, lambda_reg=0.1, uniform_weight=0.1):
        super(VectorQuantizer_conv, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.uniform_weight = uniform_weight

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, input):
        z = input

        # z의 차원 변환 및 플래튼
        z_flattened = z.view(-1, self.e_dim)

        # z와 각 임베딩 벡터 간의 거리 계산
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # 가장 가까운 임베딩 찾기
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # 양자화된 잠재 벡터 계산
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # 양자화 오차 계산
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # 그래디언트 보존
        z_q = z + (z_q - z).detach()

        # 임베딩 벡터 간의 거리 계산
        embedding_distances = torch.cdist(self.embedding.weight, self.embedding.weight, p=2)

        # 임베딩 벡터 간의 거리의 최소값과 최대값 계산
        min_distance, _ = torch.min(embedding_distances, dim=1, keepdim=True)
        max_distance, _ = torch.max(embedding_distances, dim=1, keepdim=True)

        # 균일성 손실 계산: 최소 거리와 최대 거리의 차이
        uniform_loss = torch.mean(max_distance - min_distance)

        # qq_loss를 균일성 손실로 계산
        qq_loss = self.uniform_weight * uniform_loss

        # 정규화 항 추가 (L2 정규화)
        reg_loss = self.lambda_reg * torch.sum(self.embedding.weight ** 2)

        # 최종 손실 계산
        qq_loss = qq_loss + reg_loss

        # 분산 계산
        variance = torch.var(embedding_distances)
        print("분산:", variance.item())

        return z_q, min_encoding_indices, loss, qq_loss

class VectorQuantizer_conv_avg(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer_conv_avg, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, input):
        z = input

        # z의 차원 변환 및 플래튼
        z_flattened = z.view(-1, self.e_dim)

        # z와 각 임베딩 벡터 간의 거리 계산
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # z와 각 임베딩 벡터 간의 거리 별 정렬 (가까운 순)
        top_k_indices = torch.topk(d, 5, largest=False, dim=1).indices

        # 가장 가까운 5개의 임베딩 벡터 선택
        top_k_embeddings = self.embedding.weight[top_k_indices]

        # 5개의 임베딩 벡터의 평균값 계산
        z_q = torch.mean(top_k_embeddings, dim=1)

        # 원래의 차원으로 다시 변환
        z_q = z_q.view(z.shape)

        # 양자화 오차 계산
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # 그래디언트 보존
        z_q = z + (z_q - z).detach()

        qq_loss = torch.tensor(0.0)

        return z_q, top_k_indices, loss, qq_loss


class Decoder_conv(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, out_dim, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder_conv, self).__init__()
        kernel = 4
        stride = 2

        # ResidualStack
        self.residual_stack = ResidualStack(in_dim, in_dim, res_h_dim, n_res_layers)

        # ConvTranspose1d 레이어들
        self.conv1 = nn.ConvTranspose1d(in_dim, h_dim, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.ConvTranspose1d(h_dim, h_dim//2, kernel_size=kernel, stride=stride)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.ConvTranspose1d(h_dim//2, h_dim//4, kernel_size=kernel, stride=stride)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.ConvTranspose1d(h_dim//4, out_dim, kernel_size=1, stride=1)


    def forward(self, x):

        # x = self.residual_stack(x)

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)

        return x

class VQVAE_conv(nn.Module):
    def __init__(self, mo_dim, n_hiddens, n_residual_hiddens, n_residual_layers,
                 n_embeddings, embedding_dim, beta):
        super(VQVAE_conv, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder_conv(mo_dim, embedding_dim, n_hiddens, n_residual_layers, n_residual_hiddens)

        # pass continuous latent vector through discretization bottleneck
        # self.codebook = VectorQuantizer_gcn(num_embeddings=n_embeddings, embedding_dim=embedding_dim, beta=beta)
        self.vector_quantization = VectorQuantizer_conv(n_embeddings, embedding_dim, beta)

        # decode the discrete latent representation
        self.decoder = Decoder_conv(mo_dim, embedding_dim, n_hiddens, n_residual_layers, n_residual_hiddens)

    def forward(self, input, verbose=False):

        """
        x: [N, D, L]
        """

        x = input # B J*R T

        # VQVAE

        z_e = self.encoder(x) # B, RJ, T
        z_e = z_e.squeeze(2)

        z_q, indices, embedding_loss, qq_loss = self.vector_quantization(z_e)

        z_q = z_q.unsqueeze(2)
        x_hat = self.decoder(z_q)

        qq_loss = torch.tensor(0.0)

        # 모든 원소와 첫 번째 원소 사이의 거리 계산
        # distances_input = []
        # distances_hat = []
        # for i in range(10):
        #     dis_input = torch.norm(input[:, :, 0] - input[:, :, i])
        #     dis_hat = torch.norm(x_hat[:, :, 0] - x_hat[:, :, i])
        #     distances_input.append(dis_input)
        #     distances_hat.append(dis_hat)
        # distances_input = torch.tensor(distances_input)
        # distances_hat = torch.tensor(distances_hat)
        #
        # # input과 hat의 활동량 비슷하게 유지, 다만 peak action의 위치는 상관없음
        # distance_farthest_input, _ = torch.max(distances_input, 0)
        # distance_farthest_hat, _ = torch.max(distances_hat, 0)
        #
        # ac_loss = (distance_farthest_input - distance_farthest_hat).abs()

        ac_loss = torch.tensor(0.0)

        # #

        # #

        return z_q, x_hat, indices, embedding_loss, qq_loss, ac_loss


    def forward_forlxm(self, x, verbose=False):

        x = x.permute((2, 1, 0)).contiguous()

        z_q = x

        x_hat = self.decoder(z_q)

        x_hat = x_hat.permute((2, 0, 1)).contiguous()
        x_hat = x_hat.reshape(1, x_hat.shape[1]*10, x_hat.shape[2])

        embedding_loss = 0

        return z_q, x_hat, embedding_loss


class autoencoder_conv(nn.Module):
    def __init__(self, mo_dim, n_hiddens, n_residual_hiddens, n_residual_layers,
                 n_embeddings, embedding_dim, rotation_size, joint_size, beta, save_img_embedding_map=False):
        super(autoencoder_conv, self).__init__()
        # self.encoder = Encoder_conv(mo_dim, embedding_dim, n_hiddens, n_residual_layers, n_residual_hiddens)
        self.encoder = nn.Conv1d(mo_dim, n_hiddens, kernel_size=3, stride=1)
        # self.decoder = Decoder_conv(mo_dim, embedding_dim, n_hiddens, n_residual_layers, n_residual_hiddens)
        self.decoder = nn.ConvTranspose1d(n_hiddens, mo_dim, kernel_size=3, stride=1)

    def forward(self, input, verbose=False):

        """
        x: [N, D, L]
        """

        x = input  # B J*R T

        # CNN

        z_q = self.encoder(x)

        x_hat = self.decoder(z_q)

        indices = []
        embedding_loss = torch.tensor(0.0)
        qq_loss = torch.tensor(0.0)

        # #

        return z_q, x_hat, indices, embedding_loss, qq_loss


class VAE_conv(nn.Module):
    def __init__(self, mo_dim, n_hiddens, n_residual_hiddens, n_residual_layers,
                 n_embeddings, embedding_dim, rotation_size, joint_size, beta, save_img_embedding_map=False):
        super(VAE_conv, self).__init__()
        self.encoder = Encoder_conv(mo_dim, embedding_dim, n_hiddens, n_residual_layers, n_residual_hiddens)
        self.vector_quantization = VectorQuantizer_conv(n_embeddings, embedding_dim, beta)
        self.mu_layer = nn.Linear(embedding_dim, embedding_dim)
        self.logvar_layer = nn.Linear(embedding_dim, embedding_dim)
        self.decoder = Decoder_conv(mo_dim, embedding_dim, n_hiddens, n_residual_layers, n_residual_hiddens)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input, verbose=False):

        """
        x: [N, D, L]
        """

        x = input  # B J*R T

        # VAE

        z_e = self.encoder(x)
        z_e = z_e.squeeze(2)

        mu = self.mu_layer(z_e)
        logvar = self.logvar_layer(z_e)
        z_q = self.reparameterize(mu, logvar)

        z_q = z_q.unsqueeze(2)
        x_hat = self.decoder(z_q)

        indices = []
        embedding_loss = torch.tensor(0.0)
        qq_loss = torch.tensor(0.0)

        # #

        return z_q, x_hat, indices, embedding_loss, qq_loss


class VQVAE_transformer(nn.Module):
    def __init__(self, mo_dim, n_hiddens, n_residual_hiddens, n_residual_layers,
                 n_embeddings, embedding_dim, rotation_size, joint_size, beta, save_img_embedding_map=False):
        super(VQVAE_transformer, self).__init__()

        # self.codebook = VectorQuantizer_gcn(num_embeddings=n_embeddings,
        #                                 embedding_dim=embedding_dim,
        #                                 beta=beta)
        self.vector_quantization = VectorQuantizer_conv(n_embeddings, embedding_dim, beta)

    def forward(self, input, verbose=False):

        """
        x: [N, D, L]
        """

        x = input # B J*R T

        return z_q, x_hat, indices, embedding_loss, qq_loss


class RVQ_conv(nn.Module):
    def __init__(self, mo_dim, n_hiddens, n_residual_hiddens, n_residual_layers,
                 n_embeddings, embedding_dim, beta, num_quantizers, save_img_embedding_map=False):
        super(RVQ_conv, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder_conv(mo_dim, embedding_dim, n_hiddens, n_residual_layers, n_residual_hiddens)

        # self.pre_quantization_conv = nn.Conv1d(n_hiddens, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.rvq = ResidualVQ(dim = embedding_dim, num_quantizers = num_quantizers, codebook_size = n_embeddings, beta = beta,
                              vector_quantize = VectorQuantizer_conv(n_embeddings, embedding_dim, beta))
                              # vector_quantize = VectorQuantizer_conv_avg(n_embeddings, embedding_dim, beta))

        # decode the discrete latent representation
        self.decoder = Decoder_conv(mo_dim, embedding_dim, n_hiddens, n_residual_layers, n_residual_hiddens)

    def forward(self, input, verbose=False):

        """
        x: [N, D, L]
        """

        x = input # B J*R T

        # VQVAE

        z_e = self.encoder(x)
        z_e = z_e.squeeze(2)

        # z_q, indices, embedding_loss, qq_loss = self.vector_quantization(z_e)
        z_q, indices, embedding_loss, qq_loss = self.rvq(z_e, None)

        z_q = z_q.unsqueeze(2)
        x_hat = self.decoder(z_q)

        embedding_loss = torch.mean(embedding_loss)
        qq_loss = torch.mean(qq_loss)

        # 모든 원소와 첫 번째 원소 사이의 거리 계산
        distances_input = []
        distances_hat = []
        for i in range(10):
            dis_input = torch.norm(input[:, :, 0] - input[:, :, i])
            dis_hat = torch.norm(x_hat[:, :, 0] - x_hat[:, :, i])
            distances_input.append(dis_input)
            distances_hat.append(dis_hat)
        distances_input = torch.tensor(distances_input)
        distances_hat = torch.tensor(distances_hat)

        # input과 hat의 활동량 비슷하게 유지, 다만 peak action의 위치는 상관없음
        distance_farthest_input, _ = torch.max(distances_input, 0)
        distance_farthest_hat, _ = torch.max(distances_hat, 0)

        ac_loss = (distance_farthest_input - distance_farthest_hat).abs()

        # #

        # qq_loss = torch.tensor(0.0)
        # ac_loss = torch.tensor(0.0)

        return z_q, x_hat, indices, embedding_loss, qq_loss, ac_loss

    def forward_forlxm(self, x, verbose=False):

        x = x.permute((2, 1, 0)).contiguous()

        z_q = x

        x_hat = self.decoder(z_q)

        x_hat = x_hat.permute((2, 0, 1)).contiguous()
        x_hat = x_hat.reshape(1, x_hat.shape[1]*10, x_hat.shape[2])

        embedding_loss = 0

        return z_q, x_hat, embedding_loss


# region Test.

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # region Conv1d

    # encoder_config = [
    #     [42, 64, 5, 1, 0],
    #     [64, 128, 4, 2, 0],
    #     [128, 156, 4, 1, 0],
    #     [156, 192, 4, 1, 0]
    # ]
    # decoder_config = [
    #     [192, 156, 4, 1, 0],
    #     [156, 128, 4, 1, 0],
    #     [128, 64, 4, 2, 0],
    #     [64, 42, 5, 1, 0]
    # ]
    #
    # conv_1d = Conv1d(encoder_config, decoder_config).to(device)
    #
    # x = torch.randn((5, 42, 20)).to(device)
    # motif, x_hat = conv_1d(x)
    #
    # print(motif.shape, x_hat.shape)

    # endregion

    # region Transformer

    # model = Transformer(48, 96).to(device)
    #
    # x = torch.randn((5, 48, 10)).to(device)  # [N, D, L]
    #
    # lexeme, x_hat = model(x)
    #
    # print(lexeme.shape)
    # print(x_hat.shape)

    # endregion

    # region network statistics

    # def get_parameter_number(model):
    #     total_num = sum(p.numel() for p in model.parameters())
    #     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     return {'Total': total_num, 'Trainable': trainable_num}
    #
    # print(get_parameter_number(model))

    # endregion

# endregion