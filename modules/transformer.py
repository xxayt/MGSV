import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadedAttention(nn.Module):
    '''
    X-Pool original function
    From: https://github.com/layer6ai-labs/xpool/blob/bc88718199007d395136adc99a3e69adbee874a0/modules/transformer.py#L7
    '''
    def __init__(self, args):
        super(MultiHeadedAttention, self).__init__()
        self.args = args
        self.embed_dim = args.dim_input
        self.num_heads = 1
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        # print('text_embeds.shape:', text_embeds.shape)
        num_texts, _ = text_embeds.shape
        # num_texts x embed_dim
        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        # num_texts x head_dim x num_texts
        q = q.permute(1,2,0)

        num_vids, num_frames, _ = video_embeds.shape
        # num_vids x num_frames x embed_dim
        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0,2,1,3)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0,2,3,1)

        # The dot product attention gives relevancy weights from a text to each frame.
        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q  # [num_vids x num_heads x num_frames x head_dim] @ [num_heads x head_dim x num_texts] = [num_vids x num_heads x num_frames x num_texts]
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights  # [num_vids x num_heads x head_dim x num_frames] @ [num_vids x num_heads x num_frames x num_texts] = [num_vids x num_heads x head_dim x num_texts]
        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)  # num_vids x num_texts x embed_dim

        # num_vids x num_texts x embed_dim
        o = self.out_proj(attention)
        return o


class MultiHeadedAttention_mask(nn.Module):
    def __init__(self, args):
        super(MultiHeadedAttention_mask, self).__init__()
        self.args = args
        self.embed_dim = args.dim_input
        self.num_heads = 1
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, video_embeds, music_embeds, music_mask=None):
        """
        Input
            video_embeds: (num_vids, embed_dim)
            music_embeds: (num_music, num_segments, embed_dim)
            music_mask: (num_music, num_segments)
        Output
            o: num_music x num_vids x embed_dim
        """
        num_vids, _ = video_embeds.shape
        # num_vids x embed_dim
        q = self.q_proj(video_embeds)
        q = rearrange(q, 'n (h d) -> h n d', h=self.num_heads, d=self.head_dim)  # （num_heads, num_vids, head_dim）

        num_music, num_segments, _ = music_embeds.shape
        k = self.k_proj(music_embeds)  # (num_music, num_segments, embed_dim)
        k = rearrange(k, 'm s (h d) -> m h s d', h=self.num_heads, d=self.head_dim)  # (num_music, num_heads, num_segments, head_dim)
        
        v = self.v_proj(music_embeds)  # (num_music, num_segments, embed_dim)
        v = rearrange(v, 'm s (h d) -> m h s d', h=self.num_heads, d=self.head_dim)  # (num_music, num_heads, num_segments, head_dim)


        # The dot product attention gives relevancy weights from a video to each segment. 
        attention_logits = torch.matmul(q.unsqueeze(0), k.transpose(-1, -2))  # (num_music, num_heads, num_vids, num_segments)
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        
        # mask the attention_logits
        if music_mask is not None:
            music_mask = music_mask[:, None, None, :]  # (num_music, 1, 1, num_segments)
            attention_logits = attention_logits.masked_fill(music_mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_logits, dim=-1)  # (num_music, num_heads, num_vids, num_segments)

        attention = torch.matmul(attention_weights, v)  # (num_music, num_heads, num_vids, head_dim)
        attention = rearrange(attention, 'm h n d -> m n (h d)')  # (num_music, num_vids, num_heads*head_dim)

        o = self.out_proj(attention)  # (num_music, num_vids, embed_dim)
        return o




class Transformer_XA(nn.Module):
    def __init__(self, args):
        super(Transformer_XA, self).__init__()
        self.args = args
        self.embed_dim = args.dim_input
        dropout = 0.3 # args.transformer_dropout
        
        if self.args.fusion_mask == 1:
            self.cross_attn = MultiHeadedAttention_mask(self.args)
        else:
            self.cross_attn = MultiHeadedAttention(self.args)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, video_embeds, music_embeds, music_mask=None):
        """
        Input
            video_embeds: num_vids x embed_dim
            music_embeds: num_music x num_frames x embed_dim
        Output
            out: num_music x num_vids x embed_dim
        """
        video_embeds = self.layer_norm1(video_embeds)  # [num_vids, embed_dim]
        music_embeds = self.layer_norm1(music_embeds)  # [num_music, num_frames, embed_dim]

        # num_music x num_vids x embed_dim
        if self.args.fusion_mask == 1:
            attn_out = self.cross_attn(video_embeds, music_embeds, music_mask)
        else:
            attn_out = self.cross_attn(video_embeds, music_embeds)
        # attn_out: an aggregated video embedding conditioned on the text t
        # Note!!! Here should not be a residual connection
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        return out


