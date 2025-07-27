"""
From X-Pool: https://github.com/layer6ai-labs/xpool/blob/main/modules/metrics.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
import scipy.stats

def sim_matrix_music_pooling(video_embeds, music_embeds_pooled):
    """
    Computes the similarity matrix using pooled music
    Input:
        video_embeds: [bs_v, dim]
        music_embeds_pooled: [bs_m, bs_v, dim]
    Output:
        sims: [bs_v, bs_m]
    """
    video_embeds = video_embeds / video_embeds.norm(dim=-1, keepdim=True)
    music_embeds_pooled = music_embeds_pooled / music_embeds_pooled.norm(dim=-1, keepdim=True)
    video_embeds = video_embeds.unsqueeze(1)  # [bs_v, 1, dim]
    music_embeds_pooled = rearrange(music_embeds_pooled, 'm v d -> v d m')  # [bs_v, dim, bs_m]
    sims = torch.bmm(video_embeds, music_embeds_pooled).squeeze(1)  # [bs_v, bs_m]
    return sims

def sim_matrix_video_pooling(video_embeds_pooled, music_embeds):
    """
    Computes the similarity matrix using pooled video
    Input:
        video_embeds_pooled: [bs_v, bs_m, dim]
        music_embeds: [bs_m, dim]
    Output:
        sims: [bs_v, bs_m]
    """
    video_embeds_pooled = video_embeds_pooled / video_embeds_pooled.norm(dim=-1, keepdim=True)
    music_embeds = music_embeds / music_embeds.norm(dim=-1, keepdim=True)
    music_embeds = music_embeds.unsqueeze(1)  # [bs_m, 1, dim]
    video_embeds_pooled = rearrange(video_embeds_pooled, 'v m d -> m d v')  # [bs_m, dim, bs_v]
    sims = torch.bmm(music_embeds, video_embeds_pooled).squeeze(1)  # [bs_m, bs_v]
    sims = sims.t()  # [bs_v, bs_m]
    return sims

def sim_matrix_both_pooling(video_embeds_pooled, music_embeds_pooled):
    """
    Computes the similarity matrix using pooled video and music
    Input:
        video_embeds_pooled: [bs_v, bs_m, dim]
        music_embeds_pooled: [bs_m, bs_v, dim]
    Output:
        sims: [bs_v, bs_m]
    """
    video_embeds_pooled = video_embeds_pooled / video_embeds_pooled.norm(dim=-1, keepdim=True)
    music_embeds_pooled = music_embeds_pooled / music_embeds_pooled.norm(dim=-1, keepdim=True)
    sims = torch.bmm(video_embeds_pooled, music_embeds_pooled.permute(1, 2, 0))  # [bs_v, bs_m, bs_m]
    sims = sims.mean(dim=1)  # [bs_v, bs_m]
    assert sims.shape == (video_embeds_pooled.shape[0], music_embeds_pooled.shape[0]), f"Shape mismatch: {sims.shape} != {(video_embeds_pooled.shape[0], music_embeds_pooled.shape[0])}"
    return sims