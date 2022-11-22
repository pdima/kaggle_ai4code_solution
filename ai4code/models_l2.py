import copy
import math
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PositionalEncodingCombined(nn.Module):
    """
    Pos encoding, to combine the absolute position and the "scaled" position to position, relative to rel_size.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, rel_size=1024, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.rel_size = rel_size

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        self.abs_pos_enc = nn.Linear(d_model, d_model)
        self.rel_pos_enc = nn.Linear(d_model, d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, embedding_dim]
        """
        size = x.size(0)
        rel_idx = torch.arange(size)

        if size > 1:
            rel_idx = torch.round(rel_idx.float() * (self.rel_size-1) / (size-1)).long()

        x = x + self.abs_pos_enc(self.pe[:size]) + self.rel_pos_enc(self.pe[rel_idx])
        return self.dropout(x)


class CrossAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, rescale):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.rescale = rescale
        self.hidden_dim = hidden_dim

        if rescale:
            nn.init.xavier_uniform_(self.query.weight)
            nn.init.xavier_uniform_(self.key.weight)

    def forward(self, query, key):
        res = torch.matmul(self.query(query), self.key(key).T)
        if self.rescale:
            # / (query.shape[0] ** 0.5)
            res = res / (self.hidden_dim ** 0.5)

        return res


class CrossAttentionSequencePool(nn.Module):
    def __init__(self, input_dim, hidden_dim, pool_mode='max'):
        super().__init__()
        hidden_dim = int(hidden_dim)

        self.query1 = nn.Linear(input_dim, hidden_dim)
        self.query2 = nn.Linear(hidden_dim, hidden_dim)

        self.key1 = nn.Linear(input_dim * 3, hidden_dim)
        self.key2 = nn.Linear(hidden_dim, hidden_dim)

        self.hidden_dim = hidden_dim
        self.pool_mode = pool_mode

        nn.init.xavier_uniform_(self.query2.weight)
        nn.init.xavier_uniform_(self.key2.weight)

    def forward(self, query, key):
        if self.pool_mode == 'mean':
            x_before, x_after = mean_before_after(key)
        elif self.pool_mode == 'max':
            x_before, x_after = max_before_after(key)
        else:
            raise RuntimeError(f'Invalid pool mode: {self.mode}')
        x = torch.cat([key, x_before, x_after], dim=1)
        x_key = self.key2(F.relu(self.key1(x)))

        x_query = self.query2(F.relu(self.query1(query)))

        res = torch.matmul(x_query, x_key.T)
        res = res / (self.hidden_dim ** 0.5)

        return res



class L2Transformer(nn.Module):
    def __init__(
            self,
            nb_combine_cells_around,
            nhead_code=32,
            nhead_md=32,
            num_decoder_layers=4,
            dec_dim=1024,
            dim_feedforward=2048,
            encoder_code_dim=768*2,
            encoder_md_dim=768*2,
            ca_mul=2,
            combined_pos_enc=True,
            add_extra_outputs=True,
            rescale_att=True
    ):
        super().__init__()

        dec_dim = dec_dim
        dim_feedforward = dim_feedforward

        nhead_code = nhead_code
        nhead_md = nhead_md
        self.num_decoder_layers = num_decoder_layers

        dropout = 0.1
        activation = F.relu
        layer_norm_eps: float = 1e-5
        batch_first: bool = False
        norm_first: bool = False

        # decoder_norm = nn.LayerNorm(dec_dim, eps=layer_norm_eps)

        code_decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_dim, nhead=nhead_code, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)
        self.code_decoders = nn.ModuleList([copy.deepcopy(code_decoder_layer) for i in range(self.num_decoder_layers)])

        md_decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_dim, nhead=nhead_md, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)

        self.md_decoders = nn.ModuleList([copy.deepcopy(md_decoder_layer) for i in range(self.num_decoder_layers)])

        encoder_code_pool_dim = encoder_code_dim
        encoder_md_pool_dim = encoder_md_dim

        # self.code_2_dec = nn.Linear(bert_pool_dim, dec_dim)
        self.code_2_dec = nn.Conv1d(encoder_code_pool_dim, dec_dim, kernel_size=nb_combine_cells_around*2, padding=nb_combine_cells_around)

        self.md_2_dec = nn.Linear(encoder_md_pool_dim, dec_dim)

        if combined_pos_enc:
            self.code_pos_enc = PositionalEncodingCombined(dec_dim)
        else:
            self.code_pos_enc = PositionalEncoding(dec_dim)

        self.output_md_after_code = CrossAttention(dec_dim, dec_dim * ca_mul, rescale=rescale_att)
        self.output_md_between_code = CrossAttention(dec_dim, dec_dim * ca_mul, rescale=rescale_att)
        if add_extra_outputs:
            self.output_md_right_before_code = CrossAttention(dec_dim, dec_dim * ca_mul, rescale=rescale_att)

        self.output_md_after_md = CrossAttention(dec_dim, dec_dim * ca_mul, rescale=rescale_att)
        if add_extra_outputs:
            self.output_md_right_before_md = CrossAttention(dec_dim, dec_dim * ca_mul, rescale=rescale_att)
            self.output_md_right_after_md = CrossAttention(dec_dim, dec_dim * ca_mul, rescale=rescale_att)

        self.output_list = False

    def forward(self, x_code_pooled, x_md_pooled):
        x_code = self.code_2_dec(x_code_pooled[None, :, :].permute(0, 2, 1))
        x_code = x_code[0].permute(1, 0)
        # print(x_code_pooled.shape, x_code.shape)
        # x_code.shape[0] == x_code_pooled.shape[0] + 1, so x_code points between code cells
        x_code = self.code_pos_enc(x_code)

        x_md = self.md_2_dec(x_md_pooled)

        for step in range(self.num_decoder_layers):
            x_code = self.code_decoders[step](x_code, x_md)
            x_md = self.md_decoders[step](x_md, x_code)

        md_after_code = self.output_md_after_code(x_md, x_code)
        # md_right_before_code = self.output_md_right_before_code(x_md, x_code)
        # md_right_after_code = md_right_before_code
        # md_right_after_code = self.output_md_right_after_code(x_md, x_code)
        md_between_code = self.output_md_between_code(x_md, x_code)
        # md_right_before_code = md_between_code

        # md_between_code = F.log_softmax(md_between_code, dim=1)

        md_after_md = self.output_md_after_md(x_md, x_md)
        # md_right_before_md = self.output_md_right_before_md(x_md, x_md)
        # md_right_after_md = self.output_md_right_after_md(x_md, x_md)

        if self.output_list:
            return md_after_code, md_between_code, md_after_md

        return dict(
            md_after_code=md_after_code,
            # md_right_before_code=md_right_before_code,
            # md_right_after_code=md_right_after_code,
            md_between_code=md_between_code,
            md_after_md=md_after_md,
            # md_right_before_md=md_right_before_md,
            # md_right_after_md=md_right_after_md
        )


class L2TransformerSequencePool(nn.Module):
    def __init__(
            self,
            nb_combine_cells_around,
            nhead_code=32,
            nhead_md=32,
            num_decoder_layers=4,
            dec_dim=1024,
            dim_feedforward=2048,
            encoder_code_dim=768*2,
            encoder_md_dim=768*2,
            ca_mul=2,
            combined_pos_enc=True
    ):
        super().__init__()

        dec_dim = dec_dim
        dim_feedforward = dim_feedforward

        nhead_code = nhead_code
        nhead_md = nhead_md
        self.num_decoder_layers = num_decoder_layers

        dropout = 0.1
        activation = F.relu
        layer_norm_eps: float = 1e-5
        batch_first: bool = False
        norm_first: bool = False

        # decoder_norm = nn.LayerNorm(dec_dim, eps=layer_norm_eps)

        code_decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_dim, nhead=nhead_code, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)
        self.code_decoders = nn.ModuleList([copy.deepcopy(code_decoder_layer) for i in range(self.num_decoder_layers)])

        md_decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_dim, nhead=nhead_md, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)

        self.md_decoders = nn.ModuleList([copy.deepcopy(md_decoder_layer) for i in range(self.num_decoder_layers)])

        encoder_code_pool_dim = encoder_code_dim
        encoder_md_pool_dim = encoder_md_dim

        # self.code_2_dec = nn.Linear(bert_pool_dim, dec_dim)
        self.code_2_dec = nn.Conv1d(encoder_code_pool_dim, dec_dim, kernel_size=nb_combine_cells_around*2, padding=nb_combine_cells_around)

        self.md_2_dec = nn.Linear(encoder_md_pool_dim, dec_dim)

        if combined_pos_enc:
            self.code_pos_enc = PositionalEncodingCombined(dec_dim)
        else:
            self.code_pos_enc = PositionalEncoding(dec_dim)

        self.output_md_after_code = CrossAttentionSequencePool(dec_dim, dec_dim * ca_mul, pool_mode='max')
        self.output_md_between_code = CrossAttention(dec_dim, dec_dim * ca_mul, rescale=True)
        self.output_md_after_md = CrossAttention(dec_dim, dec_dim * ca_mul, rescale=True)

    def forward(self, x_code_pooled, x_md_pooled):
        x_code = self.code_2_dec(x_code_pooled[None, :, :].permute(0, 2, 1))
        x_code = x_code[0].permute(1, 0)
        # print(x_code_pooled.shape, x_code.shape)
        # x_code.shape[0] == x_code_pooled.shape[0] + 1, so x_code points between code cells
        x_code = self.code_pos_enc(x_code)

        x_md = self.md_2_dec(x_md_pooled)

        for step in range(self.num_decoder_layers):
            x_code = self.code_decoders[step](x_code, x_md)
            x_md = self.md_decoders[step](x_md, x_code)

        md_after_code = self.output_md_after_code(x_md, x_code)
        md_between_code = self.output_md_between_code(x_md, x_code)
        md_after_md = self.output_md_after_md(x_md, x_md)

        return dict(
            md_after_code=md_after_code,
            md_between_code=md_between_code,
            md_after_md=md_after_md,
        )


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=(0.0, 0.0)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvNeXtBlock1D(nn.Module):
    """ ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.

    Args:
        dim (int): Number of input channels.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, conv_kernel_size=7,  ls_init_value=1e-6, conv_mlp=False, mlp_ratio=4):
        super().__init__()

        self.use_conv_mlp = conv_mlp
        self.conv_dw = nn.Conv1d(dim, dim, kernel_size=conv_kernel_size, padding=(conv_kernel_size-1)//2, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(mlp_ratio * dim), act_layer=nn.GELU)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x[None, :, :].permute(0, 2, 1))
        x = x[0].permute(1, 0)

        x = self.norm(x)
        x = self.mlp(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1))
        x = x + shortcut
        return x


class L2TransformerWithConv7(nn.Module):
    def __init__(
            self,
            nb_combine_cells_around,
            conv_layer_nums,
            nhead_code=32,
            nhead_md=32,
            num_decoder_layers=4,
            dec_dim=1024,
            dim_feedforward=2048,
            encoder_code_dim=768*2,
            encoder_md_dim=768*2,
            mlp_ratio=4,
            conv_kernel_size=7,
            combined_pos_enc=False,
            add_extra_outputs=True,
            rescale_att=False
    ):
        super().__init__()

        dec_dim = dec_dim
        dim_feedforward = dim_feedforward

        nhead_code = nhead_code
        nhead_md = nhead_md
        self.num_decoder_layers = num_decoder_layers

        dropout = 0.1
        activation = F.relu
        layer_norm_eps: float = 1e-5
        batch_first: bool = False
        norm_first: bool = False

        assert len(conv_layer_nums) == num_decoder_layers

        # decoder_norm = nn.LayerNorm(dec_dim, eps=layer_norm_eps)

        code_decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_dim, nhead=nhead_code, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)
        self.code_decoders = nn.ModuleList([copy.deepcopy(code_decoder_layer) for i in range(self.num_decoder_layers)])

        md_decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_dim, nhead=nhead_md, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)

        self.md_decoders = nn.ModuleList([copy.deepcopy(md_decoder_layer) for i in range(self.num_decoder_layers)])

        self.conv_decoders = nn.ModuleList(
            [nn.Sequential(*[ConvNeXtBlock1D(dim=dec_dim, mlp_ratio=mlp_ratio, conv_kernel_size=conv_kernel_size)
                             for _ in range(conv_layer_nums[i])])
             for i in range(self.num_decoder_layers)])

        encoder_code_pool_dim = encoder_code_dim
        encoder_md_pool_dim = encoder_md_dim

        # self.code_2_dec = nn.Linear(bert_pool_dim, dec_dim)
        self.code_2_dec = nn.Conv1d(encoder_code_pool_dim, dec_dim, kernel_size=nb_combine_cells_around*2, padding=nb_combine_cells_around)

        self.md_2_dec = nn.Linear(encoder_md_pool_dim, dec_dim)

        if combined_pos_enc:
            self.code_pos_enc = PositionalEncodingCombined(dec_dim)
        else:
            self.code_pos_enc = PositionalEncoding(dec_dim)

        self.output_md_after_code = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)
        self.output_md_between_code = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)
        if add_extra_outputs:
            self.output_md_right_before_code = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)

        self.output_md_after_md = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)
        if add_extra_outputs:
            self.output_md_right_before_md = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)
            self.output_md_right_after_md = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)

    def forward(self, x_code_pooled, x_md_pooled):
        x_code = self.code_2_dec(x_code_pooled[None, :, :].permute(0, 2, 1))
        x_code = x_code[0].permute(1, 0)
        # print(x_code_pooled.shape, x_code.shape)
        # x_code.shape[0] == x_code_pooled.shape[0] + 1, so x_code points between code cells
        x_code = self.code_pos_enc(x_code)

        x_md = self.md_2_dec(x_md_pooled)

        for step in range(self.num_decoder_layers):
            x_code = self.conv_decoders[step](x_code)
            x_code = self.code_decoders[step](x_code, x_md)
            x_md = self.md_decoders[step](x_md, x_code)

        md_after_code = self.output_md_after_code(x_md, x_code)
        # md_right_before_code = self.output_md_right_before_code(x_md, x_code)
        # md_right_after_code = md_right_before_code
        # md_right_after_code = self.output_md_right_after_code(x_md, x_code)
        md_between_code = self.output_md_between_code(x_md, x_code)
        # md_right_before_code = md_between_code

        # md_between_code = F.log_softmax(md_between_code, dim=1)

        md_after_md = self.output_md_after_md(x_md, x_md)
        # md_right_before_md = self.output_md_right_before_md(x_md, x_md)
        # md_right_after_md = self.output_md_right_after_md(x_md, x_md)

        return dict(
            md_after_code=md_after_code,
            # md_right_before_code=md_right_before_code,
            # md_right_after_code=md_right_after_code,
            md_between_code=md_between_code,
            md_after_md=md_after_md,
            # md_right_before_md=md_right_before_md,
            # md_right_after_md=md_right_after_md
        )


def mean_before_after(x: torch.tensor):
    n, depth = x.shape
    x2_before = [torch.zeros_like(x[0])]

    for i in range(1, n):
        x2_before.append(torch.mean(x[:i], dim=0))
    x2_before = torch.stack(x2_before, 0)

    x2_after = []
    for i in range(n-1):
        x2_after.append(torch.mean(x[i+1:], dim=0))
    x2_after.append(torch.zeros_like(x[0]))
    x2_after = torch.stack(x2_after, 0)

    return x2_before, x2_after


def max_before_after(x: torch.tensor):
    n, depth = x.shape
    x2_before = [torch.zeros_like(x[0])]

    for i in range(1, n):
        x2_before.append(torch.max(x[:i], dim=0)[0])
    x2_before = torch.stack(x2_before, 0)

    x2_after = []
    for i in range(n-1):
        x2_after.append(torch.max(x[i+1:], dim=0)[0])
    x2_after.append(torch.zeros_like(x[0]))
    x2_after = torch.stack(x2_after, 0)

    return x2_before, x2_after


def test_mean_before_after():
    from torch.testing import assert_close

    x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=torch.float32)

    x2_before, x2_after = mean_before_after(x)

    assert_close(x2_before[0], torch.tensor([0.0, 0.0]))
    assert_close(x2_before[1], x[:1].mean(dim=0))
    assert_close(x2_before[2], x[:2].mean(dim=0))
    assert_close(x2_before[3], x[:3].mean(dim=0))
    assert_close(x2_before[4], x[:4].mean(dim=0))

    assert_close(x2_after[0], x[1:].mean(dim=0))
    assert_close(x2_after[1], x[2:].mean(dim=0))
    assert_close(x2_after[2], x[3:].mean(dim=0))
    assert_close(x2_after[4], torch.tensor([0.0, 0.0]))


test_mean_before_after()

def test_max_before_after():
    from torch.testing import assert_close

    x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=torch.float32)

    x2_before, x2_after = max_before_after(x)

    assert_close(x2_before[0], torch.tensor([0.0, 0.0]))
    assert_close(x2_before[1], torch.max(x[:1], dim=0)[0])
    assert_close(x2_before[2], torch.max(x[:2], dim=0)[0])
    assert_close(x2_before[3], torch.max(x[:3], dim=0)[0])
    assert_close(x2_before[4], torch.max(x[:4], dim=0)[0])

    assert_close(x2_after[0], torch.max(x[1:], dim=0)[0])
    assert_close(x2_after[1], torch.max(x[2:], dim=0)[0])
    assert_close(x2_after[2], torch.max(x[3:], dim=0)[0])
    assert_close(x2_after[3], torch.max(x[4:], dim=0)[0])
    assert_close(x2_after[4], torch.tensor([0.0, 0.0]))

test_max_before_after()


class BeforeAfterPoolBlock(nn.Module):
    def __init__(self, dim, dim_feedforward, mode: str, ls_init_value=1e-6, dropout: float = 0.1):
        super().__init__()

        self.norm = nn.LayerNorm(dim_feedforward, eps=1e-6)
        self.mode = mode

        self.linear1 = nn.Linear(dim*3, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim)

        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim))

    def forward(self, x):
        shortcut = x

        if self.mode == 'mean':
            x_before, x_after = mean_before_after(x)
        elif self.mode == 'max':
            x_before, x_after = max_before_after(x)
        else:
            raise RuntimeError(f'Invalid pool mode: {self.mode}')

        x = torch.cat([x, x_before, x_after], dim=1)
        x = F.relu(self.linear1(x))
        x = self.norm(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = x.mul(self.gamma.reshape(1, -1))
        # x = self.norm(x + shortcut)
        x = x + shortcut
        return x


class BeforeAfterPoolBlockV2(nn.Module):
    def __init__(self, dim, dim_feedforward, mode: str, ls_init_value=1e-6, dropout: float = 0.1):
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mode = mode

        self.linear1 = nn.Linear(dim*3, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim)

        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim))

    def forward(self, x):
        shortcut = x

        if self.mode == 'mean':
            x_before, x_after = mean_before_after(x)
        elif self.mode == 'max':
            x_before, x_after = max_before_after(x)
        else:
            raise RuntimeError(f'Invalid pool mode: {self.mode}')

        x = torch.cat([x, x_before, x_after], dim=1)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm(x + shortcut)
        return x


class L2TransformerWithBeforeAfterPool(nn.Module):
    def __init__(
            self,
            nb_combine_cells_around,
            nhead_code=32,
            nhead_md=32,
            num_decoder_layers=4,
            dec_dim=1024,
            dim_feedforward=2048,
            dim_feedforward_pool=2048,
            pool_mode='mean',
            encoder_code_dim=768*2,
            encoder_md_dim=768*2,
            combined_pos_enc=True,
            rescale_att=False
    ):
        super().__init__()

        dec_dim = dec_dim
        dim_feedforward = dim_feedforward

        nhead_code = nhead_code
        nhead_md = nhead_md
        self.num_decoder_layers = num_decoder_layers

        dropout = 0.1
        activation = F.relu
        layer_norm_eps: float = 1e-5
        batch_first: bool = False
        norm_first: bool = False

        code_decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_dim, nhead=nhead_code, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)
        self.code_decoders = nn.ModuleList([copy.deepcopy(code_decoder_layer) for i in range(self.num_decoder_layers)])

        md_decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_dim, nhead=nhead_md, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)

        self.md_decoders = nn.ModuleList([copy.deepcopy(md_decoder_layer) for i in range(self.num_decoder_layers)])

        if pool_mode == 'mean_max':
            self.pool_decoders = nn.ModuleList(
                [nn.Sequential(BeforeAfterPoolBlockV2(dim=dec_dim, dim_feedforward=dim_feedforward_pool, mode='mean'),
                               BeforeAfterPoolBlockV2(dim=dec_dim, dim_feedforward=dim_feedforward_pool, mode='max'))
                 for i in range(self.num_decoder_layers)])
        else:
            self.pool_decoders = nn.ModuleList(
                [BeforeAfterPoolBlockV2(dim=dec_dim, dim_feedforward=dim_feedforward_pool, mode=pool_mode)
                 for i in range(self.num_decoder_layers)])

        encoder_code_pool_dim = encoder_code_dim
        encoder_md_pool_dim = encoder_md_dim

        # self.code_2_dec = nn.Linear(bert_pool_dim, dec_dim)
        self.code_2_dec = nn.Conv1d(encoder_code_pool_dim, dec_dim, kernel_size=nb_combine_cells_around*2, padding=nb_combine_cells_around)

        self.md_2_dec = nn.Linear(encoder_md_pool_dim, dec_dim)

        if combined_pos_enc:
            self.code_pos_enc = PositionalEncodingCombined(dec_dim)
        else:
            self.code_pos_enc = PositionalEncoding(dec_dim)

        self.output_md_after_code = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)
        self.output_md_between_code = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)
        self.output_md_after_md = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)


    def forward(self, x_code_pooled, x_md_pooled):
        x_code = self.code_2_dec(x_code_pooled[None, :, :].permute(0, 2, 1))
        x_code = x_code[0].permute(1, 0)
        # print(x_code_pooled.shape, x_code.shape)
        # x_code.shape[0] == x_code_pooled.shape[0] + 1, so x_code points between code cells
        x_code = self.code_pos_enc(x_code)

        x_md = self.md_2_dec(x_md_pooled)

        for step in range(self.num_decoder_layers):
            x_code = self.pool_decoders[step](x_code)
            x_code = self.code_decoders[step](x_code, x_md)
            x_md = self.md_decoders[step](x_md, x_code)

        md_after_code = self.output_md_after_code(x_md, x_code)
        # md_right_before_code = self.output_md_right_before_code(x_md, x_code)
        # md_right_after_code = md_right_before_code
        # md_right_after_code = self.output_md_right_after_code(x_md, x_code)
        md_between_code = self.output_md_between_code(x_md, x_code)
        # md_right_before_code = md_between_code

        # md_between_code = F.log_softmax(md_between_code, dim=1)

        md_after_md = self.output_md_after_md(x_md, x_md)
        # md_right_before_md = self.output_md_right_before_md(x_md, x_md)
        # md_right_after_md = self.output_md_right_after_md(x_md, x_md)

        return dict(
            md_after_code=md_after_code,
            # md_right_before_code=md_right_before_code,
            # md_right_after_code=md_right_after_code,
            md_between_code=md_between_code,
            md_after_md=md_after_md,
            # md_right_before_md=md_right_before_md,
            # md_right_after_md=md_right_after_md
        )



class L2GruTransformer(nn.Module):
    def __init__(
            self,
            nb_combine_cells_around,
            gru_code_layers=2,
            gru_code_dropout=0.0,
            nhead_code=32,
            nhead_md=32,
            num_decoder_layers=4,
            dec_dim=1024,
            dim_feedforward=2048,
            encoder_code_dim=768*2,
            encoder_md_dim=768*2,
            combined_pos_enc=False,
            rescale_att=False
    ):
        super().__init__()

        dec_dim = dec_dim
        dim_feedforward = dim_feedforward

        nhead_code = nhead_code
        nhead_md = nhead_md
        self.num_decoder_layers = num_decoder_layers

        dropout = 0.1
        activation = F.relu
        layer_norm_eps: float = 1e-5
        batch_first: bool = False
        norm_first: bool = False

        # decoder_norm = nn.LayerNorm(dec_dim, eps=layer_norm_eps)
        self.gru_code_decoder = nn.GRU(input_size=dec_dim, hidden_size=dec_dim // 2, num_layers=gru_code_layers, dropout=gru_code_dropout,
                                       bidirectional=True)

        code_decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_dim, nhead=nhead_code, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)
        self.code_decoders = nn.ModuleList([copy.deepcopy(code_decoder_layer) for i in range(self.num_decoder_layers)])

        md_decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_dim, nhead=nhead_md, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)

        self.md_decoders = nn.ModuleList([copy.deepcopy(md_decoder_layer) for i in range(self.num_decoder_layers)])

        encoder_code_pool_dim = encoder_code_dim
        encoder_md_pool_dim = encoder_md_dim

        # self.code_2_dec = nn.Linear(bert_pool_dim, dec_dim)
        self.code_2_dec = nn.Conv1d(encoder_code_pool_dim, dec_dim, kernel_size=nb_combine_cells_around*2, padding=nb_combine_cells_around)

        self.md_2_dec = nn.Linear(encoder_md_pool_dim, dec_dim)

        if combined_pos_enc:
            self.code_pos_enc = PositionalEncodingCombined(dec_dim)
        else:
            self.code_pos_enc = PositionalEncoding(dec_dim)

        self.output_md_after_code = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)
        self.output_md_between_code = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)
        self.output_md_right_before_code = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)

        self.output_md_after_md = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)
        self.output_md_right_before_md = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)
        self.output_md_right_after_md = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)

    def forward(self, x_code_pooled, x_md_pooled):
        x_code = self.code_2_dec(x_code_pooled[None, :, :].permute(0, 2, 1))
        x_code = x_code[0].permute(1, 0)
        x_code = self.gru_code_decoder(x_code)[0]
        # print(x_code_pooled.shape, x_code.shape)
        # x_code.shape[0] == x_code_pooled.shape[0] + 1, so x_code points between code cells
        x_code = self.code_pos_enc(x_code)

        x_md = self.md_2_dec(x_md_pooled)

        for step in range(self.num_decoder_layers):
            x_code = self.code_decoders[step](x_code, x_md)
            x_md = self.md_decoders[step](x_md, x_code)

        md_after_code = self.output_md_after_code(x_md, x_code)
        md_right_before_code = self.output_md_right_before_code(x_md, x_code)
        md_right_after_code = md_right_before_code
        # md_right_after_code = self.output_md_right_after_code(x_md, x_code)
        md_between_code = self.output_md_between_code(x_md, x_code)
        # md_right_before_code = md_between_code

        # md_between_code = F.log_softmax(md_between_code, dim=1)

        md_after_md = self.output_md_after_md(x_md, x_md)
        md_right_before_md = self.output_md_right_before_md(x_md, x_md)
        md_right_after_md = self.output_md_right_after_md(x_md, x_md)

        return dict(
            md_after_code=md_after_code,
            md_right_before_code=md_right_before_code,
            md_right_after_code=md_right_after_code,
            md_between_code=md_between_code,
            md_after_md=md_after_md,
            md_right_before_md=md_right_before_md,
            md_right_after_md=md_right_after_md
        )




class L2GruCodeMdTransformer(nn.Module):
    def __init__(
            self,
            nb_combine_cells_around,
            nhead_md=32,
            num_decoder_layers=2,
            dec_dim=1024,
            code_layers=2,
            code_dropout=0.1,
            dim_feedforward=1024,
            encoder_code_dim=768*2,
            encoder_md_dim=768*2,
            rescale_att=True
    ):
        super().__init__()

        dec_dim = dec_dim
        dim_feedforward = dim_feedforward

        nhead_md = nhead_md
        self.num_decoder_layers = num_decoder_layers

        dropout = 0.1
        activation = F.relu
        layer_norm_eps: float = 1e-5
        batch_first: bool = False
        norm_first: bool = False

        # decoder_norm = nn.LayerNorm(dec_dim, eps=layer_norm_eps)
        self.code_decoder = nn.GRU(input_size=dec_dim, hidden_size=dec_dim//2, num_layers=code_layers, dropout=code_dropout, bidirectional=True)

        md_decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_dim, nhead=nhead_md, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)

        self.md_decoders = nn.ModuleList([copy.deepcopy(md_decoder_layer) for i in range(self.num_decoder_layers)])

        encoder_code_pool_dim = encoder_code_dim
        encoder_md_pool_dim = encoder_md_dim

        # self.code_2_dec = nn.Linear(bert_pool_dim, dec_dim)
        self.code_2_dec = nn.Conv1d(encoder_code_pool_dim, dec_dim, kernel_size=nb_combine_cells_around*2, padding=nb_combine_cells_around)

        self.md_2_dec = nn.Linear(encoder_md_pool_dim, dec_dim)
        # self.code_pos_enc = PositionalEncoding(dec_dim)

        self.output_md_after_code = CrossAttention(dec_dim, dec_dim, rescale=rescale_att)
        self.output_md_between_code = CrossAttention(dec_dim, dec_dim, rescale=rescale_att)
        # self.output_md_right_before_code = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)

        self.output_md_after_md = CrossAttention(dec_dim, dec_dim, rescale=rescale_att)
        # self.output_md_right_before_md = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)
        # self.output_md_right_after_md = CrossAttention(dec_dim, dec_dim * 2, rescale=rescale_att)

    def forward(self, x_code_pooled, x_md_pooled):
        x_code = self.code_2_dec(x_code_pooled[None, :, :].permute(0, 2, 1))
        x_code = x_code[0].permute(1, 0)
        # print(x_code_pooled.shape, x_code.shape)
        # x_code.shape[0] == x_code_pooled.shape[0] + 1, so x_code points between code cells
        # x_code = self.code_pos_enc(x_code)

        x_md = self.md_2_dec(x_md_pooled)

        x_code = self.code_decoder(x_code)[0]

        for step in range(self.num_decoder_layers):
            # x_code = self.code_decoders[step](x_code, x_md)
            x_md = self.md_decoders[step](x_md, x_code)

        md_after_code = self.output_md_after_code(x_md, x_code)
        # md_right_before_code = self.output_md_right_before_code(x_md, x_code)
        # md_right_after_code = md_right_before_code
        # md_right_after_code = self.output_md_right_after_code(x_md, x_code)
        md_between_code = self.output_md_between_code(x_md, x_code)
        # md_right_before_code = md_between_code

        # md_between_code = F.log_softmax(md_between_code, dim=1)

        md_after_md = self.output_md_after_md(x_md, x_md)
        # md_right_before_md = self.output_md_right_before_md(x_md, x_md)
        # md_right_after_md = self.output_md_right_after_md(x_md, x_md)

        return dict(
            md_after_code=md_after_code,
            # md_right_before_code=md_right_before_code,
            # md_right_after_code=md_right_after_code,
            md_between_code=md_between_code,
            md_after_md=md_after_md,
            # md_right_before_md=md_right_before_md,
            # md_right_after_md=md_right_after_md
        )


def check_cross_attention():
    att = CrossAttention(input_dim=1024, hidden_dim=1024, rescale=True)

    code = torch.normal(0, 1.0, (1024, 1024))
    md = torch.normal(0, 1.0, (15, 1024))
    a = att.forward(md, code)
    print(float(torch.mean(a)), float(torch.std(a)))

    code = torch.normal(0, 1.0, (15, 1024))
    md = torch.normal(0, 1.0, (1024, 1024))
    a = att.forward(md, code)
    print(float(torch.mean(a)), float(torch.std(a)))


def print_summary_l2():
    # import pytorch_model_summary

    model = L2Transformer(
        nb_combine_cells_around=1,
        nhead_code=12,
        nhead_md=12,
        num_decoder_layers=2,
        dec_dim=768,
        dim_feedforward=1024,
        encoder_code_dim=256,
        encoder_md_dim=256,
        combined_pos_enc=True,
        rescale_att=True
    )

    code = torch.ones((9, 256))
    md = torch.ones((5, 256))

    device = torch.device("cpu")
    model.to(device)

    res = model(code, md)
    print({k: r.shape for k, r in res.items()})

    # print(pytorch_model_summary.summary(model, code, md, max_depth=2))


if __name__ == '__main__':
    print_summary_l2()
