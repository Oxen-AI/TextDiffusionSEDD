import oxen.datasets
import torch
from torch import nn
import torch.nn.functional as F
import math
from itertools import chain
import torch.optim as optim
from torch import Tensor
import abc
import os
import json
import numpy as np
from einops import rearrange
from typing import Optional
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

import oxen
from aim import Run
from tqdm import tqdm

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

# Partially based on: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the result of
                `model.parameters()`.
            decay: The exponential decay.
            use_num_updates: Whether to use number of updates when computing
                averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) /
                        (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))


    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']

# build model
class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Lookup from indices to token embeddings
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
            # This makes the transformation on v an identity.
            self.cos_cached[:,:,2,:,:].fill_(1.)
            self.sin_cached[:,:,2,:,:].fill_(0.)

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=-1
    )

def apply_rotary_pos_emb(qkv, cos, sin):
    import flash_attn.layers.rotary
    cos = cos[0,:,0,0,:cos.shape[-1]//2]
    sin = sin[0,:,0,0,:sin.shape[-1]//2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(
        qkv, cos, sin
    )

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift

@torch.jit.script
def modulate_fused(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return modulate(x, shift, scale)

def bias_dropout_add_scale(
    x: Tensor, bias: Optional[Tensor], scale: Tensor, residual: Optional[Tensor], prob: float, training: bool
) -> Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)

    return _bias_dropout_add

@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: Tensor, bias: Optional[Tensor], scale: Tensor, residual: Optional[Tensor], prob: float
) -> Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: Tensor, bias: Optional[Tensor], scale: Tensor, residual: Optional[Tensor], prob: float
) -> Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]

class DDiTBlock(nn.Module):

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout


        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        # dtype0 = x.dtype

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        if seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)
        x = flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0., causal=False)

        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
        return x

# class DDiTBlock(nn.Module):

#     def __init__(self, n_embd, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
#         super().__init__()

#         assert n_embd % n_heads == 0

#         self.n_heads = n_heads
#         self.n_embd = n_embd
#         self.dropout = dropout

#         # Attention
#         self.norm1 = LayerNorm(n_embd)
#         self.attn_qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
#         self.attn_out = nn.Linear(n_embd, n_embd, bias=False)
#         self.attn_dropout = nn.Dropout(dropout)
#         self.resid_dropout = nn.Dropout(dropout)

#         # MLP
#         self.norm2 = LayerNorm(n_embd)
#         self.mlp = nn.Sequential(
#             nn.Linear(n_embd, mlp_ratio * n_embd, bias=True),
#             nn.GELU(approximate="tanh"),
#             nn.Linear(mlp_ratio * n_embd, n_embd, bias=True)
#         )

#         self.adaLN_modulation = nn.Linear(cond_dim, 6 * n_embd, bias=True)
#         self.adaLN_modulation.weight.data.zero_()
#         self.adaLN_modulation.bias.data.zero_()


#     def _get_bias_dropout_scale(self):
#         return (
#             bias_dropout_add_scale_fused_train
#             if self.training
#             else bias_dropout_add_scale_fused_inference
#         )


#     def forward(self, x):
#         B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

#         # calculate query, key, values for all heads in batch and move head forward to be the batch dim
#         q, k, v  = self.attn_qkv(x).split(self.n_embd, dim=2)
#         k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
#         q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
#         v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

#         # manual implementation of attention
#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_dropout(att)
#         y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

#         # re-assemble all head outputs side by side
#         y = y.transpose(1, 2).contiguous().view(B, T, C)

#         # output projection
#         y = self.resid_dropout(self.attn_out(y))
#         return y


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# class SEDD(nn.Module):
#     def __init__(self, vocab_size, hidden_size=64, cond_dim=32, n_heads=4, n_blocks=8, dropout=0.1):
#         super().__init__()

#         self.vocab_embed = EmbeddingLayer(hidden_size, vocab_size)
#         self.pos_encoder = PositionalEncoding(hidden_size, dropout)
#         self.sigma_map = TimestepEmbedder(cond_dim)

#         # Forget the transformer for now
#         if False:
#             # Let's just get it working with a big linear layer
#             self.linear_1 = nn.Linear(hidden_size, hidden_size)
#             self.linear_2 = nn.Linear(hidden_size, hidden_size)
#             self.linear_3 = nn.Linear(hidden_size, hidden_size)
#             self.linear_4 = nn.Linear(hidden_size, hidden_size)

#             self.output_layer = DDitFinalLayer(hidden_size, vocab_size, cond_dim)
#         else:
#             self.rotary_emb = Rotary(hidden_size // n_heads)

#             self.blocks = nn.ModuleList([
#                 DDiTBlock(hidden_size, n_heads, cond_dim, dropout=dropout) for _ in range(n_blocks)
#             ])

#             self.output_layer = DDitFinalLayer(hidden_size, vocab_size, cond_dim)

#     def forward(self, indices, sigma):
#         if False:
#             # print("Got x")
#             # print(x.shape)

#             # print("Got c")
#             # print(c.shape)
#             x = self.linear_1(x)
#             x = self.linear_2(x)
#             x = self.linear_3(x)
#             x = self.linear_4(x)
#             x = self.output_layer(x, c)

#             # TODO is this optional?: scale_by_sigma=True
#             # esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
#             # x = x - esigm1_log - np.log(x.shape[-1] - 1) # this will be approximately averaged at 0

#             x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
#         else:
#             x = self.vocab_embed(indices)
#             x = self.pos_encoder(x)

#             # rotary_cos_sin = self.rotary_emb(x)
#             # x = x + c # Time Embeddings

#             for i in range(len(self.blocks)):
#                 x = self.blocks[i](x)

#             c = F.silu(self.sigma_map(sigma))
#             x = self.output_layer(x, c)

#             esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
#             x = x - esigm1_log - np.log(x.shape[-1] - 1)# this will be approximately averaged at 0
#             x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

#             return x

#         return x

class SEDD(nn.Module):
    def __init__(self, vocab_size, hidden_size=64, cond_dim=32, n_heads=4, n_blocks=8, dropout=0.1):
        super().__init__()

        self.vocab_embed = EmbeddingLayer(hidden_size, vocab_size)
        # self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.sigma_map = TimestepEmbedder(cond_dim)

        self.vocab_embed = EmbeddingLayer(hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(hidden_size // n_heads)

        self.blocks = nn.ModuleList([
            DDiTBlock(hidden_size, n_heads, cond_dim, dropout=dropout) for _ in range(n_blocks)
        ])

        self.output_layer = DDitFinalLayer(hidden_size, vocab_size, cond_dim)


    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    def forward(self, indices, sigma):

        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)

            x = self.output_layer(x, c)


        # if self.scale_by_sigma:
        #     assert self.absorb, "Haven't configured this to work."
        esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
        x = x - esigm1_log - np.log(x.shape[-1] - 1)# this will be approximately averaged at 0

        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

        return x

class GeometricNoise(nn.Module):
    def __init__(self, sigma_min=1e-3, sigma_max=1, learnable=False):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])
        if learnable:
            self.sigmas = nn.Parameter(self.sigmas)
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (self.sigmas[1].log() - self.sigmas[0].log())

    def total_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t

class LogLinearNoise(nn.Module):
    """
    Log Linear noise schedule built so that 1 - 1/e^(n(t)) interpolates between 0 and ~1
    when t goes from 0 to 1. Used for absorbing

    Total noise is -log(1 - (1 - eps) * t), so the sigma will be (1 - eps) * t
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)

class AbsorbingGraph:
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim + 1

    @property
    def absorb(self):
        return True

    def rate(self, i):
        # edge = - F.one_hot(i, num_classes=self.dim)
        # edge.scatter_add_(-1, i[..., None], torch.ones_like(edge[..., :1]))
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)

    def transp_rate(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(self, i, sigma):
        pass

    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert

    def staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, sigma, x, x0):
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        # positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

        # constant term
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy

def score_fn(model, x, sigma, train=False, sampling=False):
    sigma = sigma.reshape(-1)
    score = model(x, sigma)

    if sampling:
        # when sampling return true score (not log used for training)
        return score.exp()

    return score

def optimize_fn(optimizer,
                    params,
                    step,
                    lr,
                    warmup,
                    grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""

    if warmup > 0:
        for g in optimizer.param_groups:
            g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
        torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

    optimizer.step()

def loss_fn(model, noise, graph, batch, idx2token, train=True, t=None, perturbed_batch=None):
    """
    Batch shape: [B, L] int. D given from graph
    """
    sampling_eps=1e-3

    if t is None:
        t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps

    sigma, dsigma = noise(t)
    # print("t", t)
    # print("sigma", sigma.item())
    # print("dsigma", dsigma.item())

    # print("batch")
    # print(batch[0])

    if perturbed_batch is None:
        perturbed_batch = graph.sample_transition(batch, sigma[:, None])

    # print("perturbed_batch")
    # print(perturbed_batch[0])
    # print(detokenize(perturbed_batch[0].tolist(), idx2token))

    log_score = score_fn(model, perturbed_batch, sigma, train=train, sampling=False)
    loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

    loss = (dsigma[:, None] * loss).sum(dim=-1)

    return loss

def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, model, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass

class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, model, x, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(model, x, curr_sigma)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)

class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, model, x, t):
        sigma = self.noise(t)[0]

        score = score_fn(model, x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]

        #return probs.argmax(dim=-1)
        return sample_categorical(probs)

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cuda'), proj_fun=lambda x: x, idx2token=None):
    predictor = AnalyticPredictor(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        # print("Time steps")
        # print(timesteps)
        # print(timesteps.shape)

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(score_fn, model, x, t, dt)

            if idx2token is not None:

                if i % 10 == 0:
                    print(f"{i} -> {t.item()}", detokenize(x[0].tolist(), idx2token))


        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(score_fn, model, x, t)
            print(f"Denoised: ", detokenize(x[0].tolist(), idx2token))


        return x

    return pc_sampler

def sample(model, graph, noise, idx2token, context_length=1024, steps=128):
    sampling_fn = get_pc_sampler(
        graph, noise, (1, context_length), 'analytic', steps, idx2token=idx2token
    )

    samples = sampling_fn(model)
    # print("Samples")
    # print(samples)

    chars = detokenize(samples[0].tolist(), idx2token)

    return "".join(chars)

def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].module.load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].module.state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)

def gen_vocabs():
    # Generate simple vocab
    # '_' is our padding token
    raw_vocab = "_abcdefghijklmnopqrstuvwxyz .?\n"
    token2idx = {}
    idx2token = {}
    for i, v in enumerate(raw_vocab):
        idx2token[i] = v
        token2idx[v] = i

    print("Vocab")
    print(token2idx)
    return token2idx, idx2token

def tokenize(text, token2idx, context_length=1024):
    tokens = [token2idx[t] for t in text]
    padded = np.zeros(context_length, dtype=np.int64)
    padded[:len(tokens)] = tokens
    return padded

def detokenize(tokens, idx2token):
    return [idx2token[t] for t in tokens]

def prepare_data(context_length=1024):
    dataset_file = 'babi_train.jsonl'
    if not os.path.exists(dataset_file):
        oxen.datasets.download('datasets/babi_qa', dataset_file)

    examples = []
    sum_lens = 0
    with open(dataset_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = data['context'].lower()
            sum_lens += len(data['context'])
            # trucate to context length
            if len(text) > context_length:
                text = text[:context_length]
            examples.append(text)


    avg_len = sum_lens / len(examples)
    print(f"Average length: {avg_len}")

    return examples

def main():
    run = Run()

    run["hparams"] = {
        "learning_rate": 1e-4,
        # "learning_rate": 3e-3,
        "n_epochs": 1_000_000,
        "hidden_size": 768,
        "cond_dim": 128,
        "n_heads": 8,
        "n_blocks": 8,
        "dropout": 0.1,
        # "context_length": 128,
        "context_length": 64,
    }

    token2idx, idx2token = gen_vocabs()
    # training_data = prepare_data(context_length=run["hparams"]["context_length"])

    training_data = [
        'the cat sat on the mat',
        'ow now brown cow',
        'unique new york'
    ] * 200

    sentence = training_data[0]

    tokens = tokenize(sentence, token2idx)
    print(f"Tokenized: {tokens}")

    sentence = detokenize(tokens, idx2token)
    print(f"Sentence: {''.join(sentence)}")

    vocab_size = len(token2idx) - 1

    # build token graph
    graph = AbsorbingGraph(vocab_size)

    model = SEDD(
        len(token2idx),
        hidden_size=run["hparams"]["hidden_size"],
        cond_dim=run["hparams"]["cond_dim"],
        n_heads=run["hparams"]["n_heads"],
        n_blocks=run["hparams"]["n_blocks"],
        dropout=run["hparams"]["dropout"]
    ).to(torch.device('cuda'))
    noise = LogLinearNoise().to(torch.device('cuda'))

    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)

    optimizer = optim.Adam(
        chain(model.parameters(), noise.parameters()),
        lr=run["hparams"]["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0
    )

    state = dict(optimizer=optimizer, model=model, noise=noise, ema=ema, step=0)

    print(f"Optimizer: {optimizer}")
    for e in range(run["hparams"]["n_epochs"]):
        # print(f"Epoch: {e}")
        losses = []
        for i, s in enumerate(tqdm(training_data)):
            # print(f"Step: {s}")
            padded = tokenize(s, token2idx, context_length=run["hparams"]["context_length"])
            batch = torch.Tensor([padded]*32).long().to(torch.device('cuda'))
            # print(s)

            loss = loss_fn(model, noise, graph, batch, idx2token).mean()
            losses.append(loss.item())

            # optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
            # print("Loss", loss.item())
            run.track(loss.item(), name='loss', step=state['step'], context={ "subset":"train" })

            # Backprop
            state['step'] += 1
            loss.backward()
            optimizer.step()
            state['ema'].update(model.parameters())
            optimizer.zero_grad()

        # compute average loss
        avg_loss = sum(losses) / len(losses)
        run.track(avg_loss, name='avg_loss', step=e, context={ "subset":"train_avg" })

        print(f"Epoch {e} average loss: {avg_loss}")
        # if e % 10 == 0 and e > 0:
        # save_checkpoint(f"checkpoints/{e}.pt", state)
        example = sample(model, graph, noise, idx2token, context_length=run['hparams']['context_length'])
        print(example)
        # break



if __name__ == "__main__":
    main()