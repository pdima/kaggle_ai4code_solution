import torch
import torch.nn as nn
import torch.nn.functional as F

import code_dataset
import config
import numpy as np
import math
import copy

from collections import OrderedDict
from typing import List, Callable, Optional, Union, Tuple

import transformers
import models_l2

from models_l2 import PositionalEncoding, PositionalEncodingCombined, CrossAttention


def masked_avg(t, mask):
    return (t * mask[:, :, None]).sum(dim=1)/mask.sum(dim=1)[:, None]


def masked_avg_max(t, mask):
    avg = masked_avg(t, mask)
    max_t = (t + (mask[:, :, None] - 1) * 100.0).max(dim=1)[0]
    return torch.cat([avg, max_t], dim=1)


class DualBertWithL2(nn.Module):
    def __init__(
            self,
            code_model_name,
            md_model_name,
            l2_name,
            l2_params,
            pool_mode='avg',
            pretrained=False,
            enable_gradient_checkpointing=False
    ):
        super(DualBertWithL2, self).__init__()
        self.bert_code = transformers.AutoModel.from_pretrained(code_model_name)
        self.bert_md = transformers.AutoModel.from_pretrained(md_model_name)
        self.pool_mode = pool_mode

        if l2_name == 'L2Transformer':
            self.l2 = models_l2.L2Transformer(**l2_params)
        elif l2_name == 'L2TransformerWithConv7':
            self.l2 = models_l2.L2TransformerWithConv7(**l2_params)
        elif l2_name == 'L2TransformerWithBeforeAfterPool':
            self.l2 = models_l2.L2TransformerWithBeforeAfterPool(**l2_params)
        elif l2_name == 'None':
            self.l2 = None
        else:
            raise RuntimeError('Invalid L2 model name ', l2_name)

        if enable_gradient_checkpointing:
            self.bert_code.gradient_checkpointing_enable()
            self.bert_md.gradient_checkpointing_enable()
        else:
            self.bert_code.gradient_checkpointing_disable()
            self.bert_md.gradient_checkpointing_disable()

    def get_code_model(self):
        return self.bert_code

    def get_md_model(self):
        return self.bert_md

    def set_backbone_trainable(self, requires_grad):
        for param in self.bert_code.parameters():
            param.requires_grad = requires_grad
        for param in self.bert_md.parameters():
            param.requires_grad = requires_grad

    def pool(self, x, mask):
        if self.pool_mode == 'avg':
            x_pooled = masked_avg(x.last_hidden_state, mask)
        elif self.pool_mode == 'avg_max':
            x_pooled = masked_avg_max(x.last_hidden_state, mask)
        elif self.pool_mode == 'marker':
            x_pooled = x.pooler_output
        else:
            raise Exception(f'Invalid pool mode: "{self.pool_mode}"')

        return x_pooled

    def combine_predictions(self, model: nn.Module, tokens: [code_dataset.TokenIdsBatch]):
        predictions = []

        for token in tokens:
            x = model(input_ids=token.input_ids, attention_mask=token.attention_mask)
            x = self.pool(x, token.attention_mask)
            predictions.append(x)

        predictions = torch.cat(predictions, dim=0)
        return predictions[code_dataset.restore_order_idx(tokens), :]

    def forward_combined(self, x_code_pooled, x_md_pooled):
        return self.l2(x_code_pooled, x_md_pooled)

    def forward(self, code_ids: [code_dataset.TokenIdsBatch], md_ids: [code_dataset.TokenIdsBatch]):
        # code_ids, ...: N, S
        x_code_pooled = self.combine_predictions(self.bert_code, code_ids)
        x_md_pooled = self.combine_predictions(self.bert_md, md_ids)

        return self.l2(x_code_pooled, x_md_pooled)


class SingleBertWithL2(nn.Module):
    def __init__(
            self,
            code_model_name,
            l2_name,
            l2_params,
            pool_mode='avg',
            pretrained=False,
            enable_gradient_checkpointing=False
    ):
        super(SingleBertWithL2, self).__init__()
        self.bert_code = transformers.AutoModel.from_pretrained(code_model_name)
        # self.bert_md = transformers.AutoModel.from_pretrained(md_model_name)
        self.pool_mode = pool_mode

        if l2_name == 'L2Transformer':
            self.l2 = models_l2.L2Transformer(**l2_params)
        elif l2_name == 'L2TransformerWithConv7':
            self.l2 = models_l2.L2TransformerWithConv7(**l2_params)
        elif l2_name == 'L2TransformerWithBeforeAfterPool':
            self.l2 = models_l2.L2TransformerWithBeforeAfterPool(**l2_params)
        elif l2_name == 'None':
            self.l2 = None
        else:
            raise RuntimeError('Invalid L2 model name ', l2_name)

        if enable_gradient_checkpointing:
            self.bert_code.gradient_checkpointing_enable()
        else:
            self.bert_code.gradient_checkpointing_disable()

    def get_code_model(self):
        return self.bert_code

    def get_md_model(self):
        return self.bert_code

    def set_backbone_trainable(self, requires_grad):
        for param in self.bert_code.parameters():
            param.requires_grad = requires_grad

    def pool(self, x, mask):
        if self.pool_mode == 'avg':
            x_pooled = masked_avg(x.last_hidden_state, mask)
        elif self.pool_mode == 'avg_max':
            x_pooled = masked_avg_max(x.last_hidden_state, mask)
        elif self.pool_mode == 'marker':
            x_pooled = x.pooler_output
        else:
            raise Exception(f'Invalid pool mode: "{self.pool_mode}"')

        return x_pooled

    def combine_predictions(self, model: nn.Module, tokens: [code_dataset.TokenIdsBatch]):
        predictions = []

        for token in tokens:
            x = model(input_ids=token.input_ids, attention_mask=token.attention_mask)
            x = self.pool(x, token.attention_mask)
            predictions.append(x)

        predictions = torch.cat(predictions, dim=0)
        return predictions[code_dataset.restore_order_idx(tokens), :]

    def forward_combined(self, x_code_pooled, x_md_pooled):
        return self.l2(x_code_pooled, x_md_pooled)

    def forward(self, code_ids: [code_dataset.TokenIdsBatch], md_ids: [code_dataset.TokenIdsBatch]):
        # code_ids, ...: N, S
        x_code_pooled = self.combine_predictions(self.bert_code, code_ids)
        x_md_pooled = self.combine_predictions(self.bert_code, md_ids)

        return self.l2(x_code_pooled, x_md_pooled)


class DualBertWithRNN(nn.Module):
    def __init__(
            self,
            code_model_name,
            md_model_name,
            nb_combine_cells_around,
            code_rnn='GRU',
            code_rnn_layers=2,
            code_rnn_dropout=0.1,
            nhead_md=12,
            num_decoder_layers=2,
            dec_dim=768,
            dim_feedforward=768,
            pool_mode='avg_max',
            encoder_code_dim=768,
            encoder_md_dim=768,
            enable_gradient_checkpointing=False
    ):
        super().__init__()
        self.code_model_name = code_model_name

        if '_t5_' in code_model_name:
            self.bert_code = transformers.T5EncoderModel.from_pretrained(code_model_name)
        else:
            self.bert_code = transformers.AutoModel.from_pretrained(code_model_name, use_cache=False)
        self.bert_md = transformers.AutoModel.from_pretrained(md_model_name)

        if enable_gradient_checkpointing:
            self.bert_code.gradient_checkpointing_enable()
            self.bert_md.gradient_checkpointing_enable()
        else:
            self.bert_code.gradient_checkpointing_disable()
            self.bert_md.gradient_checkpointing_disable()

        self.pool_mode = pool_mode

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
        if code_rnn == 'GRU':
            self.code_decoder = nn.GRU(input_size=dec_dim, hidden_size=dec_dim // 2,
                                       num_layers=code_rnn_layers,
                                       dropout=code_rnn_dropout,
                                       bidirectional=True)
        elif code_rnn == 'LSTM':
            self.code_decoder = nn.LSTM(input_size=dec_dim, hidden_size=dec_dim // 2,
                                        num_layers=code_rnn_layers,
                                        dropout=code_rnn_dropout,
                                        bidirectional=True)

        md_decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_dim, nhead=nhead_md, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)

        self.md_decoders = nn.ModuleList([copy.deepcopy(md_decoder_layer) for i in range(self.num_decoder_layers)])

        encoder_code_pool_dim = encoder_code_dim
        encoder_md_pool_dim = encoder_md_dim
        if pool_mode == 'avg_max':
            encoder_code_pool_dim = encoder_code_dim * 2
            encoder_md_pool_dim = encoder_md_dim * 2

        # self.code_2_dec = nn.Linear(bert_pool_dim, dec_dim)
        self.code_2_dec = nn.Conv1d(encoder_code_pool_dim, dec_dim, kernel_size=nb_combine_cells_around*2, padding=nb_combine_cells_around)

        self.md_2_dec = nn.Linear(encoder_md_pool_dim, dec_dim)

        self.output_md_after_code = CrossAttention(dec_dim, dec_dim * 2)
        self.output_md_between_code = CrossAttention(dec_dim, dec_dim * 2)
        self.output_md_right_before_code = CrossAttention(dec_dim, dec_dim * 2)

        # self.output_md_right_after_code = CrossAttention(dec_dim, dec_dim * 2)

        self.output_md_after_md = CrossAttention(dec_dim, dec_dim * 2)
        self.output_md_right_before_md = CrossAttention(dec_dim, dec_dim * 2)
        self.output_md_right_after_md = CrossAttention(dec_dim, dec_dim * 2)

    def set_backbone_trainable(self, requires_grad):
        for param in self.bert_code.parameters():
            param.requires_grad = requires_grad
        for param in self.bert_md.parameters():
            param.requires_grad = requires_grad

    def get_code_model(self):
        return self.bert_code

    def get_md_model(self):
        return self.bert_md

    def pool(self, x, mask):
        if self.pool_mode == 'avg':
            x_pooled = masked_avg(x.last_hidden_state, mask)
        elif self.pool_mode == 'avg_max':
            x_pooled = masked_avg_max(x.last_hidden_state, mask)
        elif self.pool_mode == 'marker':
            x_pooled = x.pooler_output
        else:
            raise Exception(f'Invalid pool mode: "{self.pool_mode}"')

        return x_pooled

    def combine_predictions(self, model: nn.Module, tokens: [code_dataset.TokenIdsBatch]):
        predictions = []

        for token in tokens:
            x = model(input_ids=token.input_ids, attention_mask=token.attention_mask)
            x = self.pool(x, token.attention_mask)
            predictions.append(x)

        predictions = torch.cat(predictions, dim=0)
        return predictions[code_dataset.restore_order_idx(tokens), :]

    def forward_combined(self, x_code_pooled, x_md_pooled):
        x_code = self.code_2_dec(x_code_pooled[None, :, :].permute(0, 2, 1))
        x_code = x_code[0].permute(1, 0)
        # print(x_code_pooled.shape, x_code.shape)
        # x_code.shape[0] == x_code_pooled.shape[0] + 1, so x_code points between code cells
        x_code = self.code_decoder(x_code)[0]
        x_md = self.md_2_dec(x_md_pooled)

        for step in range(self.num_decoder_layers):
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

    def forward(self, code_ids: [code_dataset.TokenIdsBatch], md_ids: [code_dataset.TokenIdsBatch]):
        # code_ids, ...: N, S
        x_code_pooled = self.combine_predictions(self.bert_code, code_ids)
        x_md_pooled = self.combine_predictions(self.bert_md, md_ids)

        return self.forward_combined(x_code_pooled, x_md_pooled)


class DualBertConv(nn.Module):
    def __init__(
            self,
            code_model_name,
            md_model_name,
            nhead_code=32,
            nhead_md=32,
            num_decoder_layers=4,
            dec_dim=1024,
            dim_feedforward=2048,
            pretrained=False,
            pool_mode='marker'
    ):
        super(DualBertConv, self).__init__()
        self.bert_code = transformers.AutoModel.from_pretrained(code_model_name)
        self.bert_md = transformers.AutoModel.from_pretrained(md_model_name)

        self.pool_mode = pool_mode
        bert_dim = 768

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

        bert_pool_dim = bert_dim
        if pool_mode == 'avg_max':
            bert_pool_dim = bert_dim * 2

        self.code_2_dec = nn.Linear(bert_pool_dim, dec_dim)
        self.md_2_dec = nn.Linear(bert_pool_dim, dec_dim)
        # self.code_pos_enc = PositionalEncoding(dec_dim)

        self.conv1 = nn.Conv1d(dec_dim, dec_dim, kernel_size=5, padding=2, bias=False)
        self.conv1_norm = nn.LayerNorm(dec_dim)
        self.conv2 = nn.Conv1d(dec_dim, dec_dim, kernel_size=7, padding=3, bias=False)
        self.conv2_norm = nn.LayerNorm(dec_dim)
        self.conv3 = nn.Conv1d(dec_dim, dec_dim, kernel_size=1)
        nn.init.constant_(self.conv3.weight, 0.)
        nn.init.constant_(self.conv3.bias, 0.)

        self.output_md_after_code = CrossAttention(dec_dim, dec_dim * 2)
        self.output_md_right_before_code = CrossAttention(dec_dim, dec_dim * 2)
        self.output_md_right_after_code = CrossAttention(dec_dim, dec_dim * 2)

        self.output_md_after_md = CrossAttention(dec_dim, dec_dim * 2)
        self.output_md_right_before_md = CrossAttention(dec_dim, dec_dim * 2)
        self.output_md_right_after_md = CrossAttention(dec_dim, dec_dim * 2)

    def set_backbone_trainable(self, requires_grad):
        if requires_grad:
            for param in self.bert_code.parameters():
                param.requires_grad = True
            for param in self.bert_md.parameters():
                param.requires_grad = True
        else:
            for param in self.bert_code.parameters():
                param.requires_grad = False
                param.grad = None
            for param in self.bert_md.parameters():
                param.requires_grad = False
                param.grad = None

    def pool(self, x, mask):
        if self.pool_mode == 'avg':
            x_pooled = masked_avg(x.last_hidden_state, mask)
        elif self.pool_mode == 'avg_max':
            x_pooled = masked_avg_max(x.last_hidden_state, mask)
        elif self.pool_mode == 'marker':
            x_pooled = x.pooler_output
        else:
            raise Exception(f'Invalid pool mode: "{self.pool_mode}"')

        return x_pooled

    def combine_predictions(self, model: nn.Module, tokens: [code_dataset.TokenIdsBatch]):
        predictions = []

        for token in tokens:
            x = model(input_ids=token.input_ids, attention_mask=token.attention_mask)
            x = self.pool(x, token.attention_mask)
            predictions.append(x)

        predictions = torch.cat(predictions, dim=0)
        return predictions[code_dataset.restore_order_idx(tokens), :]

    def forward_combined(self, x_code_pooled, x_md_pooled):
        x_code = self.code_2_dec(x_code_pooled)
        # x_code = self.code_pos_enc(x_code)

        x_code = self.conv1(x_code[None, :, :].permute(0, 2, 1))
        x_code = F.relu(x_code)

        # x_code2 = self.conv3(F.relu(self.conv2(x_code)))
        # x_code = x_code + x_code2
        # x_code = x_code[0].permute(1, 0)

        x_code = F.relu(self.conv2(x_code))
        x_code = x_code[0].permute(1, 0)

        x_md = self.md_2_dec(x_md_pooled)

        for step in range(self.num_decoder_layers):
            x_code = self.code_decoders[step](x_code, x_md)
            x_md = self.md_decoders[step](x_md, x_code)

        md_after_code = self.output_md_after_code(x_md, x_code)
        md_right_before_code = self.output_md_right_before_code(x_md, x_code)
        md_right_after_code = self.output_md_right_after_code(x_md, x_code)

        md_after_md = self.output_md_after_md(x_md, x_md)
        md_right_before_md = self.output_md_right_before_md(x_md, x_md)
        md_right_after_md = self.output_md_right_after_md(x_md, x_md)

        return dict(
            md_after_code=md_after_code,
            md_right_before_code=md_right_before_code,
            md_right_after_code=md_right_after_code,
            md_after_md=md_after_md,
            md_right_before_md=md_right_before_md,
            md_right_after_md=md_right_after_md
        )

    def forward(self, code_ids: [code_dataset.TokenIdsBatch], md_ids: [code_dataset.TokenIdsBatch]):
        # code_ids, ...: N, S
        x_code_pooled = self.combine_predictions(self.bert_code, code_ids)
        x_md_pooled = self.combine_predictions(self.bert_md, md_ids)

        return self.forward_combined(x_code_pooled, x_md_pooled)


def print_summary():
    import pytorch_model_summary

    model = DualBertConv(
        code_model_name='microsoft/codebert-base',
        md_model_name='microsoft/codebert-base',
        pool_mode='avg_max'
    )

    code = code_dataset.TokenIdsBatch(
        size=9,
        max_length=12,
        input_ids=torch.zeros((9, 12)).long(),
        attention_mask=torch.cat([torch.ones((9, 5)), torch.zeros((9, 7))], dim=1).long(),
        src_list_pos=list(range(9))
    )

    md = code_dataset.TokenIdsBatch(
        size=5,
        max_length=10,
        input_ids=torch.zeros((5, 10)).long(),
        attention_mask=torch.cat([torch.ones((5, 4)), torch.zeros((5, 6))], dim=1).long(),
        src_list_pos=list(range(5))
    )

    device = torch.device("cpu")
    model.to(device)

    res = model([code], [md])
    print({k: r.shape for k, r in res.items()})

    # print(pytorch_model_summary.summary(model, code_ids, code_mask, md_ids, md_mask, max_depth=2))


if __name__ == '__main__':
    print_summary()
