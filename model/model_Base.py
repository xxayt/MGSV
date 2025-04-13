import torch
import numpy as np
import os
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange
import clip
import time
from model.ast_models import ASTModel
from modules.loss import *
from music_detr.transformer import build_transformer
from music_detr.position_encoding import build_position_encoding



class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform quick gelu."""
        return x * torch.sigmoid(1.702 * x)

class FeedForward(nn.Module):
    def __init__(self, dim_in, hidden_dim, dropout = 0., init_method = "kaiming"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_in),
            nn.Dropout(dropout)
        )
        # Initialize linear layer weights
        self.init_linear_weights(self.net, init_method)

    def init_linear_weights(self, model, init_method):
        for m in model:
            if isinstance(m, nn.Linear):
                if init_method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len = 7, dim_model = 512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_len, dim_model, requires_grad = False)  # [seq_len, dim_model]
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)  # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(math.log(10000.0) / dim_model))  # [dim_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # [seq_len, dim_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)  # [seq_len, dim_model/2]
        pe = pe.unsqueeze(0)  # [1, seq_len, dim_model]
        self.register_buffer('pe', pe)

    def forward(self, length):
        return self.pe[:, :length]  # [1, length, dim_model]



class Transformer_enhancement(nn.Module):
    def __init__(self, dim_in, depth, heads, dim_head, mlp_dim, dim_out, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            attn = nn.MultiheadAttention(dim_in, heads, dropout=dropout)
            ff = nn.Sequential(
                nn.Linear(dim_in, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, dim_in),
                nn.Dropout(dropout)
            )
            norm1 = nn.LayerNorm(dim_in)
            norm2 = nn.LayerNorm(dim_in)
            self.layers.append(nn.ModuleList([norm1, attn, norm2, ff]))
        self.final_linear = nn.Linear(dim_in, dim_out)

    def forward(self, x, mask=None):
        x = x.permute(1, 0, 2)  # [seq_len, bs, dim]
        mask = mask if mask is not None else None
        for norm1, attn, norm2, ff in self.layers:
            x = norm1(x)
            x = attn(x, x, x, key_padding_mask=~(mask.bool()), need_weights=False)[0] + x
            x = norm2(x)
            x = ff(x) + x
        x = x.permute(1, 0, 2)  # [bs, seq_len, dim]
        return self.final_linear(x)







class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, init_method="kaiming", dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # Initialize linear layer weights
        self.init_linear_weights(nn.Sequential(self.to_q, self.to_kv), init_method, bias=False)
        self.init_linear_weights(self.to_out, init_method)

    def init_linear_weights(self, model, init_method, bias=True):
        for m in model:
            if isinstance(m, nn.Linear):
                if init_method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight)
                if bias:
                    m.bias.data.fill_(0.01)

    def forward(self, query, context, q_mask=None, kv_mask=None):
        '''
        Input:
            query: [bs, seq_len_q, dim]
            context: [bs, seq_len_kv, dim]
            q_mask: [bs, seq_len_q]
            kv_mask: [bs, seq_len_kv] (optional)
        '''
        q = self.to_q(query)  # [bs, seq_len_q, inner_dim]
        k, v = self.to_kv(context).chunk(2, dim=-1)  # [bs, seq_len_kv, inner_dim]
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)  # [bs, heads, seq_len_q, dim_head]
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)  # [bs, heads, seq_len_kv, dim_head]
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)  # [bs, heads, seq_len_kv, dim_head]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [bs, heads, seq_len_q, seq_len_kv]
        
        # https://github.com/Kyubyong/transformer/blob/master/modules.py: line103
        # mask for Query (perform masking after the softmax. Because the softmax is applied to the last dimension, attn will be nan if dots is masked_fill by -inf)
        if q_mask is not None and kv_mask is None:
            attn = self.attend(dots)
            q_mask = q_mask[:, None, :, None]  # [bs, 1, seq_len_q, 1]
            attn = attn.masked_fill(q_mask == 0, 0)  # [bs, heads, seq_len_q, seq_len_kv] (broadcast)
        
        # mask for Key and Value (perform masking before the softmax)
        if q_mask is None and kv_mask is not None:
            kv_mask = kv_mask[:, None, None, :]  # [bs, 1, 1, seq_len_kv]
            dots = dots.masked_fill(kv_mask == 0, float('-inf'))  # [bs, heads, seq_len_q, seq_len_kv] (broadcast)
            attn = self.attend(dots)
        
        if q_mask is not None and kv_mask is not None:
            q_mask = q_mask[:, None, :, None]
            kv_mask = kv_mask[:, None, None, :]
            dots = dots.masked_fill(kv_mask == 0, float('-inf'))  # kv_mask before softmax
            attn = self.attend(dots)
            attn = attn.masked_fill(q_mask == 0, 0)  # q_mask after softmax

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn

class CrossTransformer(nn.Module):
    def __init__(self, dim_in, depth, heads, dim_head, mlp_dim, dim_out, dropout=0., init_method="kaiming"):
        '''
        Input:
            dim_in: 256
            depth: 1
            heads: 3
            dim_head: 100
            mlp_dim: 1024
            dim_out: 768
            dropout: 0.8
            init_method: "kaiming"
        '''
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CrossAttention(dim_in, heads=heads, dim_head=dim_head, init_method=init_method, dropout=dropout),
                FeedForward(dim_in, mlp_dim, dropout=dropout, init_method=init_method)
            ]))
        self.attention_query_layer_norms = nn.ModuleList([nn.LayerNorm(dim_in) for _ in range(depth)])
        self.attention_context_layer_norms = nn.ModuleList([nn.LayerNorm(dim_in) for _ in range(depth)])
        self.ff_layer_norms = nn.ModuleList([nn.LayerNorm(dim_in) for _ in range(depth)])
        self.final_linear = nn.Linear(dim_in, dim_out)

    def forward(self, query, context, q_mask=None, kv_mask=None):
        '''
        Input:
            query: [bs, seq_len_q, dim]
            context: [bs, seq_len_kv, dim]
            q_mask: [bs, seq_len_q]
            kv_mask: [bs, seq_len_kv] (optional)
        '''
        x = query
        last_attn = None
        for i, (cross_attn, ff) in enumerate(self.layers):
            norm_x = self.attention_query_layer_norms[i](x)
            norm_context = self.attention_context_layer_norms[i](context)
            x_res, attn = cross_attn(norm_x, norm_context, q_mask, kv_mask)
            attn_x = x_res + x  # add
            norm_x = self.ff_layer_norms[i](attn_x)  # norm
            x = ff(norm_x) + attn_x
            last_attn = attn
        x = self.final_linear(x)
        return x, last_attn  # [bs, seq_len_q, dim_out], [bs, heads, seq_len_q, seq_len_kv]


class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, momentum = 0.99, hidden_size = 0, channel = 0, init_method = "kaiming"):
        super(EmbeddingNet, self).__init__()
        self.init_method = init_method
        modules = []
        if hidden_size > 0:
            modules.append(nn.Linear(in_features = input_size, out_features = hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features = channel))
            modules.append(nn.ReLU())
            # modules.append(nn.Sigmoid())
            modules.append(nn.Linear(in_features = hidden_size, out_features = output_size))
            modules.append(nn.BatchNorm1d(num_features = channel, momentum = momentum))
        else:
            modules.append(nn.Linear(in_features = input_size, out_features = output_size))
            modules.append(nn.BatchNorm1d(num_features = channel))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(output_size, output_size))
        self.net = nn.Sequential(*modules)
        # Initialize linear layer weights
        self.init_linear_weights(self.net, self.init_method)
    def init_linear_weights(self, model, init_method):
        for m in model:
            if isinstance(m, nn.Linear):
                if init_method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)
    def forward(self, x):
        output = self.net(x)
        return output
    def get_embedding(self, x):
        return self.forward(x)




class Base_model(nn.Module):
    def __init__(self, args, device=None, logger=None):
        super(Base_model, self).__init__()
        self.args = args
        self.device = device
        self.logger = logger
        self.music_frozen_feature_path = args.music_frozen_feature_path
        self.frame_frozen_feature_path = args.frame_frozen_feature_path
        
        self.dim_input = args.dim_input
        if hasattr(args, 'hidden_dim') and "MVPt" not in args.name:
            print("#####  hidden_dim is set in args, use it  #####")
            assert args.hidden_dim == self.dim_input, "hidden_dim must equal to dim_input"
            self.detr_transformer = build_transformer(args)
            self.music_position_embedding, _ = build_position_encoding(args)

        if args.local_rank == 0:
            logger.info(f'Initialize {args.audio_encoder_type} audio encoder')
            logger.info(f'Initialize {args.video_encoder_type} video encoder')
        # AST Encoder
        if args.audio_encoder_type == "AST":
            self.ast_audio_dim = 768  # fixed
            input_tdim = 1024
            self.ast_model = ASTModel(input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False).to(self.device)
            self.ast_model = torch.nn.parallel.DistributedDataParallel(self.ast_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
            checkpoint_path = './model/pretrained_models/audioset_0.4593.pth'
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.ast_model.load_state_dict(checkpoint)
            self.ast_proj = nn.Linear(self.ast_audio_dim, self.dim_input)

        # CLIP visual Encoder
        if args.video_encoder_type == "ViT":
            # https://blog.csdn.net/qq_42283621/article/details/125052688
            self.vit_video_dim = 512  # fixed
            self.vit_model, self.vit_visual_preprocess = clip.load("ViT-B/32", device=self.device, download_root='./model/pretrained_models')
            self.vit_proj = nn.Linear(self.vit_video_dim, self.dim_input)

        # Transformer & Projection
        self.video_attention_seqlen = self.args.video_attention_seqlen
        self.audio_attention_seqlen = 300
        self.encoder_attention_hidden_dim = 1024
        self.cross_hidden_dim = 1024
        self.r_enc = 0.8
        self.init_method = "xavier"  # "kaiming" / "xavier"
        self.act_after_proj = QuickGELU()
        # whether share transformer or projection
        self.transformer_is_share = False
        if args.transformer_is_share and args.video_transformer_depth == args.audio_transformer_depth and args.video_transformer_depth > 0:
            self.transformer_is_share = True
        # how to aggregate segment features to video/audio feature
        self.agg_module = args.agg_module  # "transf" / "mlp"
        if "transf" in self.agg_module:
            assert args.video_transformer_depth > 0 and args.audio_transformer_depth > 0, f"video_transformer_depth ({args.video_transformer_depth}) and audio_transformer_depth ({args.audio_transformer_depth}) must > 0 when agg_module is transformer"
        elif self.agg_module == "mlp":
            assert args.video_transformer_depth == 0 and args.audio_transformer_depth == 0, f"video_transformer_depth ({args.video_transformer_depth}) and audio_transformer_depth ({args.audio_transformer_depth}) must = 0 when agg_module is mlp or afa"

        if args.local_rank == 0:
            logger.info('Initializing transformer modules')
        if "transf" in self.agg_module and self.args.video_transformer_depth > 0 and self.args.audio_transformer_depth > 0:
            # Video position embedding
            if self.args.with_cls_token:
                self.video_cls_token = nn.Parameter(torch.zeros(1, 1, self.dim_input))
                nn.init.trunc_normal_(self.video_cls_token, std=0.02)
            self.video_position_embedding = PositionalEncoding(seq_len = self.video_attention_seqlen, dim_model = self.dim_input)
            # Music position embedding
            if self.args.with_cls_token:
                self.audio_cls_token = nn.Parameter(torch.zeros(1, 1, self.dim_input))
                nn.init.trunc_normal_(self.audio_cls_token, std=0.02)
            self.audio_position_embedding = PositionalEncoding(seq_len = self.audio_attention_seqlen, dim_model = self.dim_input)
            
            if self.transformer_is_share:
                # Share Transformer
                self.share_transformer = Transformer_enhancement(
                    dim_in = self.dim_input,
                    depth = self.args.video_transformer_depth,
                    heads = self.args.SA_temporal_heads,
                    dim_head = 64,
                    mlp_dim = self.encoder_attention_hidden_dim,
                    dim_out = self.dim_input,
                    dropout = self.r_enc,
                )
            else:
                # Video encoder Transformer
                self.video_transformer = Transformer_enhancement(
                    dim_in = self.dim_input,
                    depth = self.args.video_transformer_depth,
                    heads = self.args.SA_temporal_heads,
                    dim_head = 64,
                    mlp_dim = self.encoder_attention_hidden_dim,
                    dim_out = self.dim_input,
                    dropout = self.r_enc,
                )
                # Music encoder Transformer
                self.audio_transformer = Transformer_enhancement(
                    dim_in = self.dim_input,
                    depth = self.args.audio_transformer_depth,
                    heads = self.args.SA_temporal_heads,
                    dim_head = 64,
                    mlp_dim = self.encoder_attention_hidden_dim,
                    dim_out = self.dim_input,
                    dropout = self.r_enc,
                )
            
        elif self.agg_module == "mlp":
            # Video encoder projection
            self.Video_encoder_projection = EmbeddingNet(
                input_size = self.dim_input,
                hidden_size = 1024,
                output_size = self.dim_input,
                channel = self.args.max_v_frames,  
                dropout = 0.5,
                use_bn = True,
                init_method = self.init_method
            )
            # Music encoder projection
            self.Music_encoder_projection = EmbeddingNet(
                input_size = self.dim_input,
                hidden_size = 1024,
                output_size = self.dim_input,
                channel = self.args.max_snippet_num,
                dropout = 0.5,
                use_bn = True,
                init_method = self.init_method
            )

    def get_projection_parameter(self):
        params = []
        # video projection
        if self.args.video_encoder_type == "ViT":
            params += list(self.vit_proj.parameters())
        # audio projection
        if self.args.audio_encoder_type == "AST":
            params += list(self.ast_proj.parameters())
        return params
    
    def get_SA_parameter(self):
        params = []
        if "transf" in self.agg_module and self.args.video_transformer_depth > 0 and self.args.audio_transformer_depth > 0:
            if self.transformer_is_share:
                params += list(self.share_transformer.parameters())
            else:
                params += list(self.video_transformer.parameters())
                params += list(self.audio_transformer.parameters())
            if self.args.with_cls_token:
                params += [self.video_cls_token]
                params += [self.audio_cls_token]
        elif "mlp" in self.agg_module:
            params += list(self.Video_encoder_projection.parameters())
            params += list(self.Music_encoder_projection.parameters())
        return params
    

    def forward_video_encoder_ViT(self, videos, video_masks=None, video_ids=None):
        '''
        Inputs:
            videos: [bs, max_v_frames, 3, 224, 224]
            video_masks: [bs, max_v_frames]
        Return:
            video_feats: [bs, dim_input]
        '''
        video_set = []
        video_mask_set = []
        all_feature_loaded = True
        for idx, (video, video_id) in enumerate(zip(videos, video_ids)):  # video: [max_v_frames, C, H, W]
            feature_path = os.path.join(self.frame_frozen_feature_path, 'vit_feature', f'{video_id}.pt') if self.frame_frozen_feature_path != "vit_feature1" else "None"
            mask_path = os.path.join(self.frame_frozen_feature_path, 'vit_mask', f'{video_id}.pt') if self.frame_frozen_feature_path != "vit_feature1" else "None"
            if (feature_path != "None" and os.path.exists(feature_path) and os.path.getsize(feature_path) > 0) and (mask_path != "None" and os.path.exists(mask_path) and os.path.getsize(mask_path) > 0):# and self.args.train_data.startswith("kuai"):
                segment_set = torch.load(feature_path, map_location='cpu')  # 'cpu' | f'cuda:{self.args.local_rank}'
                video_set.append(segment_set.to(videos.device))
                segment_mask = torch.load(mask_path, map_location='cpu')
                video_mask_set.append(segment_mask.to(videos.device))
                continue
            all_feature_loaded = False
            # get ViT feature
            segment_set = []
            for image in video:  # image: [C, H, W]
                self.vit_model = self.vit_model.to(videos.device)
                # dim must: [1, 3, 224, 224] -> [1, 512]
                image_feats = self.vit_model.encode_image(image.unsqueeze(0)).squeeze()  # [512]
                segment_set.append(image_feats)
            segment_set = torch.stack(segment_set, dim=0)  # [max_v_frames, 512]
            # save ViT feature for each video
            if feature_path != "None" and mask_path != "None" and video_id is not None:
                torch.save(segment_set, feature_path)
                torch.save(video_masks[idx], mask_path)
            video_set.append(segment_set)
        video_feats = torch.stack(video_set, dim=0)  # [bs, max_v_frames, 512]
        if self.frame_frozen_feature_path != "vit_feature1" and all_feature_loaded:
            video_masks = torch.stack(video_mask_set, dim=0)  # [bs, max_v_frames]
        video_feats = video_feats.masked_fill(video_masks.unsqueeze(-1) == 0, 0)  # [bs, max_v_frames, 512]
        # Linear Projection to change video feature dim
        video_feats = self.vit_proj(video_feats)  # [bs, max_v_frames, 256]
        if "transf" in self.agg_module:
            if self.args.with_cls_token:
                cls_token = self.video_cls_token.repeat(video_feats.shape[0], 1, 1)  # [bs, 1, 256]
                video_feats = torch.cat([cls_token, video_feats], dim=1)  # [bs, max_v_frames+1, 256]
                video_masks = torch.cat([torch.ones(video_masks.shape[0], 1).to(video_masks.device), video_masks], dim=1)  # [bs, max_v_frames+1]
            # Positional Embedding
            self.video_attention_seqlen = video_feats.shape[1]  # max_v_frames
            video_feats += self.video_position_embedding(self.video_attention_seqlen).repeat(video_feats.shape[0], 1, 1)  # [bs, max_v_frames, 256]
            # Video Transformer
            video_feats = self.video_transformer(video_feats, mask=video_masks) if not self.transformer_is_share else self.share_transformer(video_feats, mask=video_masks)  # [bs, max_v_frames, 256]
            video_feats = video_feats.masked_fill(video_masks.unsqueeze(-1) == 0, 0)  # [bs, max_v_frames, 256]
        if self.args.with_cls_token:  # cls token
            video_feats = video_feats[:, 0]  # [bs, 256]
        else:  # mean pooling
            video_feats = video_feats.sum(axis=1) / video_masks.sum(axis=1).unsqueeze(-1)  # [bs, 256]
        video_feats = F.normalize(video_feats, p=2, dim=-1)
        return video_feats  # [bs, 256]

    def forward_audio_encoder_AST(self, audios, audio_masks=None, music_ids=None):
        '''
        Inputs:
            audios: [bs, max_snippet_num, target_length, mel_bins]
            audio_masks: [bs, max_snippet_num]
        Return:
            audio_feats: [bs, dim_input]
        '''
        audio_set = []
        audio_mask_set = []
        all_feature_loaded = True
        for idx, (audio, audio_id) in enumerate(zip(audios, music_ids)):  # audio: [max_snippet_num, target_length, mel_bins]
            feature_path = os.path.join(self.music_frozen_feature_path, 'ast_feature', f'{audio_id}.pt') if (self.music_frozen_feature_path.startswith("ast_feature") is False) else "None"
            mask_path = os.path.join(self.music_frozen_feature_path, 'ast_mask', f'{audio_id}.pt') if (self.music_frozen_feature_path.startswith("ast_feature") is False) else "None"
            if (feature_path != "None" and os.path.exists(feature_path) and os.path.getsize(feature_path) > 0) and (mask_path != "None" and os.path.exists(mask_path) and os.path.getsize(mask_path) > 0):
                segment_set = torch.load(feature_path, map_location='cpu').to(self.device)
                audio_set.append(segment_set)
                segment_mask = torch.load(mask_path, map_location='cpu').to(self.device)
                audio_mask_set.append(segment_mask)
                continue
            all_feature_loaded = False
            # get AST feature
            _, cls_dist_token, _ = self.ast_model(audio)  # [max_snippet_num, 1, 768]  (view max_snippet_num as batch_size in AST)
            segment_set = cls_dist_token.squeeze()  # [max_snippet_num, 768]
            audio_set.append(segment_set)
            # save AST feature for each audio
            if feature_path != "None" and mask_path != "None" and audio_id is not None:
                torch.save(segment_set, feature_path)
                torch.save(audio_masks[idx], mask_path)
        audio_feats = torch.stack(audio_set, dim=0)  # [bs, max_snippet_num, 768]
        if (self.music_frozen_feature_path.startswith("ast_feature") is False) and all_feature_loaded:
            audio_masks = torch.stack(audio_mask_set, dim=0)
        audio_feats = audio_feats.masked_fill(audio_masks.unsqueeze(-1) == 0, 0)  # [bs, max_snippet_num, 768]
        
        # Linear Projection to change audio feature dim
        audio_feats = self.ast_proj(audio_feats)  # [bs, max_snippet_num, 256]
        if "transf" in self.agg_module:
            if self.args.with_cls_token:
                cls_token = self.audio_cls_token.repeat(audio_feats.shape[0], 1, 1)  # [bs, 1, 256]
                audio_feats = torch.cat([cls_token, audio_feats], dim=1)  # [bs, max_snippet_num+1, 256]
                audio_masks = torch.cat([torch.ones(audio_masks.shape[0], 1).to(audio_masks.device), audio_masks], dim=1)  # [bs, max_snippet_num+1]
            # Positional Embedding
            self.audio_attention_seqlen = audio_feats.shape[1]  # max_snippet_num
            audio_feats += self.audio_position_embedding(self.audio_attention_seqlen).repeat(audio_feats.shape[0], 1, 1)  # [bs, max_snippet_num, 256]
            # Audio Transformer
            audio_feats = self.audio_transformer(audio_feats, mask=audio_masks) if not self.transformer_is_share else self.share_transformer(audio_feats, mask=audio_masks)  # [bs, max_snippet_num, 256]
            audio_feats = audio_feats.masked_fill(audio_masks.unsqueeze(-1) == 0, 0)  # [bs, max_snippet_num, 256]
        if self.args.with_cls_token:
            audio_feats = audio_feats[:, 0]
        else:
            audio_feats = audio_feats.sum(axis=1) / audio_masks.sum(axis=1).unsqueeze(-1)
        audio_feats = F.normalize(audio_feats, p=2, dim=-1)
        return audio_feats  # [bs, 256]



    def temporal_transformer(self, local_feats, local_masks, position_embedding, transformer, cls_token=None):
        '''
        Input:
            local_feats: [bs, seq_len, 256]
            local_masks: [bs, seq_len]
            seq_len is max_v_frames / max_snippet_num
        '''
        if self.args.with_cls_token:
            cls_token = cls_token.repeat(local_feats.shape[0], 1, 1)  # [bs, 1, 256]
            local_feats = torch.cat([cls_token, local_feats], dim=1)  # [bs, max_v_frames+1, 256]
            local_masks = torch.cat([torch.ones(local_masks.shape[0], 1).to(local_masks.device), local_masks], dim=1)  # [bs, max_v_frames+1]
        # Positional Embedding
        self.video_attention_seqlen = local_feats.shape[1]  # max_v_frames
        local_feats += position_embedding(self.video_attention_seqlen).repeat(local_feats.shape[0], 1, 1)
        # Video Transformer
        if "cal" in self.agg_module:
            local_feats = local_feats.permute(1, 0, 2)
            local_feats = transformer(local_feats, None)  # [bs, max_v_frames(+1), 256]
            local_feats = local_feats.permute(1, 0, 2)
        else:
            local_feats = transformer(local_feats, mask=local_masks)  # [bs, max_v_frames(+1), 256]
        local_feats = local_feats.masked_fill(local_masks.unsqueeze(-1) == 0, 0)
        return local_feats, local_masks

    def forward_video_encoder_feature(self, frame_feats=None, frame_masks=None, video_ids=None):
        # 和 forward_video_encoder_ViT 的不同：默认已完整抽好 frame_frozen_feature_path 中的特征
        '''
        Inputs:
            frame_feats: [bs, max_v_frames, 512]
            frame_masks: [bs, max_v_frames]
        Return:
            frame_feats: [bs, max_v_frames, 256]
            video_feats: [bs, dim_input]
            frame_masks: [bs, max_v_frames+1]
        '''
        # Apply masking to the video features
        frame_feats = frame_feats.masked_fill(frame_masks.unsqueeze(-1) == 0, 0)

        # Linear Projection to change video feature dim
        frame_feats = self.vit_proj(frame_feats)  # [bs, max_v_frames, 256]
        if self.args.with_act_after_proj:
            frame_feats = self.act_after_proj(frame_feats)
        # Temporal Modeling
        if "transf" in self.agg_module and self.args.video_transformer_depth > 0:
            transformer = self.video_transformer if not self.transformer_is_share else self.share_transformer
            frame_feats, frame_masks = self.temporal_transformer(frame_feats, frame_masks, self.video_position_embedding, transformer, 
                                                    self.video_cls_token if self.args.with_cls_token else None)
        elif self.agg_module == "mlp":
            frame_feats = self.Video_encoder_projection(frame_feats)  # [bs, max_v_frames, 256]
            frame_feats = frame_feats.masked_fill(frame_masks.unsqueeze(-1) == 0, 0)  # [bs, max_v_frames, 256]

        # choose video feature
        if self.args.with_cls_token:  # cls token
            video_feats = frame_feats[:, 0]
            frame_feats, frame_masks = frame_feats[:, 1:], frame_masks[:, 1:]
        elif "transf" in self.agg_module and "cal" in self.agg_module and self.args.with_last_token:  # last valid token
            last_frame_index = frame_masks.sum(dim=1).long() - 1  # [bs]
            video_feats = frame_feats[torch.arange(frame_feats.shape[0]), last_frame_index]  # [bs, 256]
        else:  # mean pooling
            video_feats = frame_feats.sum(axis=1) / frame_masks.sum(axis=1).unsqueeze(-1)  # [bs, 256]
        video_feats = F.normalize(video_feats, p=2, dim=-1)
        return frame_feats, video_feats, frame_masks

    def forward_audio_encoder_feature(self, segment_feats=None, segment_masks=None, music_ids=None):
        # 和forward_audio_encoder_AST的不同：默认已完整抽好music_frozen_feature_path中的特征
        '''
        Inputs:
            segment_feats: [bs, max_snippet_num, 768]
            segment_masks: [bs, max_snippet_num]
        Return:
            segment_feats: [bs, max_snippet_num, 256]
            music_feats: [bs, 256]
            segment_masks: [bs, max_snippet_num]
        '''
        # Apply masking to the video features
        segment_feats = segment_feats.masked_fill(segment_masks.unsqueeze(-1) == 0, 0)  # [bs, max_snippet_num, 768]
        
        # Linear Projection to change audio feature dim
        segment_feats = self.ast_proj(segment_feats)  # [bs, max_snippet_num, 256]
        if self.args.with_act_after_proj:
            segment_feats = self.act_after_proj(segment_feats)
        # Temporal Modeling
        if "transf" in self.agg_module and self.args.audio_transformer_depth > 0:
            transformer = self.audio_transformer if not self.transformer_is_share else self.share_transformer
            segment_feats, segment_masks = self.temporal_transformer(segment_feats, segment_masks, self.audio_position_embedding, transformer, 
                                                      self.audio_cls_token if self.args.with_cls_token else None)
        elif self.agg_module == "mlp":
            segment_feats = self.Music_encoder_projection(segment_feats)  # [bs, max_v_frames, 256]
            segment_feats = segment_feats.masked_fill(segment_masks.unsqueeze(-1) == 0, 0)  # [bs, max_v_frames, 256]

        # choose music feature
        if self.args.with_cls_token:  # cls token
            music_feats = segment_feats[:, 0]  # [bs, 256]
            segment_feats, segment_masks = segment_feats[:, 1:], segment_masks[:, 1:]
        else:  # mean pooling
            music_feats = segment_feats.sum(axis=1) / segment_masks.sum(axis=1).unsqueeze(-1)  # [bs, 256]
        music_feats = F.normalize(music_feats, p=2, dim=-1)
        return segment_feats, music_feats, segment_masks  # [bs, max_snippet_num, 256], [bs, 256], [bs, max_snippet_num]


    def forward(self, videos, audios, video_masks, audio_masks, video_ids=None, music_ids=None, is_train=False):
        '''
        Inputs:
            videos: [bs, max_v_frames, 3, 224, 224]
            audios: [bs, max_snippet_num, target_length, mel_bins]
            video_masks: [bs, max_v_frames]
            audio_masks: [bs, max_snippet_num]
            video_ids: [bs]
            music_ids: [bs]
        '''
        return