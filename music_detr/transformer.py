# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, args=None):
        super().__init__()
        self.args = args
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        if self.num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if self.num_decoder_layers > 0:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before, args)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec, args=args)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, target=None):
        """
        Inputs:
            src: [bs, seq, dim]  (segment_feats_fusion)
            mask: [bs, seq]  (segment_masks)
            query_embed (learnable): [#queries, dim]  (decoder_query_embed)
            pos_embed: [bs, seq, dim] the same as src
            target: [#queries, bs, dim]  (moment_query <= video_feats repeat)
        Returns:
        """
        # flatten NxCxHxW to HWxNxC
        bs, _, _ = src.shape
        
        # encoder
        src = src.permute(1, 0, 2)  # [seq, bs, dim]
        pos_embed = pos_embed.permute(1, 0, 2)   # [seq, bs, dim]
        memory = src
        if self.num_encoder_layers > 0:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # [seq, bs, dim]
        
        # decoder
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # Learnable Queries [#queries, bs, dim]
        if target is None:
            target = torch.zeros_like(query_embed)  # Decoder Embeddings
        hidden_states = src
        if self.num_decoder_layers > 0:
            hidden_states = self.decoder(target, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)  # [#layers, #queries, bs, dim]
        
        memory = memory.transpose(0, 1)  # [bs, seq, dim]
        hidden_states = hidden_states.transpose(1, 2)  # [#layers, bs, #queries, dim]
        return hidden_states, memory


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm  # if normalize_before is False, norm is None
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        intermediate = []
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
            if self.return_intermediate:
                intermediate.append(output)
        if self.norm is not None:
            output = self.norm(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        else:
            return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, args=None):
        super().__init__()
        self.args = args
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, target, memory,
                target_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                target_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = target
        intermediate = []
        for layer in self.layers:
            # 每次只更新output，不更新memory和query_pos！！！
            output = layer(output, memory, target_mask=target_mask,
                           memory_mask=memory_mask,
                           target_key_padding_mask=target_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)  # query_pos 每层都要传入
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        else:
            return output.unsqueeze(0)




class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        # norm
        src2 = self.norm1(src)
        # pos embedding
        q = k = self.with_pos_embed(src2, pos)
        # Multi-head Attention
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = self.dropout1(src2)
        # add & norm
        src = src2 + src
        src2 = self.norm2(src)
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = self.dropout2(src2)
        # add
        src = src2 + src
        return src

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # pos embedding in each layer
        q = k = self.with_pos_embed(src, pos)
        # Multi-head Attention
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = self.dropout1(src2)
        # add & norm
        src = src2 + src
        src = self.norm1(src)
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.dropout2(src2)
        # add & norm
        src = src2 + src
        src = self.norm2(src)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        else:
            return self.forward_post(src, src_mask, src_key_padding_mask, pos)




class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, args=None):
        super().__init__()
        self.args = args
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward_pre(self, target, memory,
                    target_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    target_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        target2 = self.norm1(target)
        q = k = self.with_pos_embed(target2, query_pos)
        target2 = self.self_attn(q, k, value=target2, attn_mask=target_mask,
                              key_padding_mask=target_key_padding_mask)[0]
        target = target + self.dropout1(target2)
        target2 = self.norm2(target)
        target2 = self.multihead_attn(query=self.with_pos_embed(target2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        target = target + self.dropout2(target2)
        target2 = self.norm3(target)
        target2 = self.linear2(self.dropout(self.activation(self.linear1(target2))))
        target = target + self.dropout3(target2)
        return target

    def forward_post(self, target, memory,
                     target_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     target_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # if self.args is not None:
        if self.args is not None and self.args.decoder_SA == 0:
            pass
        else:
            # pos embedding
            q = k = self.with_pos_embed(target, query_pos)
            # self-attention for target
            target2 = self.self_attn(q, k, value=target, attn_mask=target_mask, key_padding_mask=target_key_padding_mask)[0]
            # add & norm
            target = target + self.dropout1(target2)
            target = self.norm1(target)

        # Multi-head Attention
        target2 = self.multihead_attn(query=self.with_pos_embed(target, query_pos),  # pos embedding in each layer
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        target2 = self.dropout2(target2)
        # add & norm
        target = target + target2
        target = self.norm2(target)
        # FFN
        target2 = self.linear2(self.dropout(self.activation(self.linear1(target))))
        target2 = self.dropout3(target2)
        # add & norm
        target = target + target2
        target = self.norm3(target)
        return target

    def forward(self, target, memory,
                target_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                target_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(target, memory, target_mask, memory_mask,
                                    target_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        else:
            return self.forward_post(target, memory, target_mask, memory_mask,
                                    target_key_padding_mask, memory_key_padding_mask, pos, query_pos)



def build_transformer(args):
    return Transformer(
        d_model=args.detr_hidden_dim,  # 256
        dropout=args.detr_dropout,  # 0.1
        nhead=args.detr_nheads,  # 8
        dim_feedforward=args.detr_dim_feedforward,  # 1024
        num_encoder_layers=args.detr_enc_layers,  # 2
        num_decoder_layers=args.detr_dec_layers,  # 2
        normalize_before=args.detr_pre_norm,  # false
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU.
    stream: LN -> Dropout -> Linear -> ReLU
    """

    def __init__(self, in_dim, out_dim, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_dim)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        '''
        Input:
            x: (N, L, in_dim)
        Output:
            x: (N, L, out_dim)
        '''
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, d]


'''
class TransformerDecoderLayerThin(nn.Module):
    """removed intermediate layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_model)
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, target, memory,
                     target_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     target_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(target, query_pos)
        target2 = self.self_attn(q, k, value=target, attn_mask=target_mask,
                              key_padding_mask=target_key_padding_mask)[0]
        target = target + self.dropout1(target2)
        target = self.norm1(target)
        target2 = self.multihead_attn(query=self.with_pos_embed(target, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        target2 = self.linear1(target2)
        target = target + self.dropout2(target2)
        target = self.norm2(target)
        # target = target + self.dropout2(target2)
        # target = self.norm2(target)
        # target2 = self.linear2(self.dropout(self.activation(self.linear1(target))))
        # target = target + self.dropout3(target2)
        # target = self.norm3(target)
        return target

    def forward_pre(self, target, memory,
                    target_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    target_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        target2 = self.norm1(target)
        q = k = self.with_pos_embed(target2, query_pos)
        target2 = self.self_attn(q, k, value=target2, attn_mask=target_mask,
                              key_padding_mask=target_key_padding_mask)[0]
        target = target + self.dropout1(target2)
        target2 = self.norm2(target)
        target2 = self.multihead_attn(query=self.with_pos_embed(target2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        target = target + self.dropout2(target2)
        target2 = self.norm3(target)
        target2 = self.linear2(self.dropout(self.activation(self.linear1(target2))))
        target = target + self.dropout3(target2)
        return target

    def forward(self, target, memory,
                target_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                target_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(target, memory, target_mask, memory_mask,
                                    target_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(target, memory, target_mask, memory_mask,
                                 target_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerEncoderLayerThin(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = self.linear(src2)
        src = src + self.dropout(src2)
        src = self.norm(src)
        # src = src + self.dropout1(src2)
        # src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """not used"""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
'''