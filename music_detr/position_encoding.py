# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn


class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        """
        Input:
            input_feat: (N, L, D)
        """
        bs, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bs, 1)  # (N, L)

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images. (To 1D sequences)
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature  # the period (振幅) of the sine and cosine functions
        self.normalize = normalize  # whether to normalize the pos embedding
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        """
        Inputs:
            x: torch.tensor, [bs, L, dim]
            mask: torch.tensor, [bs, L], with 1 as valid
        Returns:
        
        """
        assert mask is not None
        x_embed = mask.cumsum(1, dtype=torch.float32)  # [bs, L]  (eg: [1, 1, 1, 0, 0] -> [1, 2, 3, 3, 3])
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale  # x_embed[:, -1:] == sum valid values in mask

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  # [num_pos_feats]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t  # [bs, L, num_pos_feats]
        # 生成不同频率的正弦和余弦波形。偶数索引对应正弦波，奇数索引对应余弦波。
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)  # [bs, L, num_pos_feats]
        return pos_x


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x, mask):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim  # 256
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    # elif args.position_embedding in ('v3', 'learned'):
    #     position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    video_pos_embed = TrainablePositionalEncoding(max_position_embeddings=args.max_snippet_num, hidden_size=args.hidden_dim, dropout=args.input_dropout)
    return position_embedding, video_pos_embed
