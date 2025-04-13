import torch
import torch.nn.functional as F
import numpy as np

def CLIPLoss(sims, logit_scale):
    """
    From X-Pool: https://github.com/layer6ai-labs/xpool/blob/main/modules/loss.py
    Inputs: cosine similarities
        sims: n x n (text is dim-0)
        logit_scale: 1 x 1
    """
    logit_scale = logit_scale.exp()
    logits = sims * logit_scale  # n x n
    
    # dim-0
    t2v_log_sm = F.log_softmax(logits, dim=1)  # n x n, apply softmax to each row
    t2v_neg_ce = torch.diag(t2v_log_sm)  # n x 1, take the diagonal elements
    t2v_loss = -t2v_neg_ce.mean()  # 1 x 1

    v2t_log_sm = F.log_softmax(logits, dim=0)
    v2t_neg_ce = torch.diag(v2t_log_sm)
    v2t_loss = -v2t_neg_ce.mean()

    return (t2v_loss + v2t_loss) / 2.0





def cal_distance(x, y, distance_type="COS"):
    """
    Shape:
        x (video feature): [bs_x, dim] 
        y (audio feature): [bs_y, dim]
    Return:
        dist_xy ("L2"): [bs_x, bs_y]  Taking values in the range [0, +∞)
        dist_xy ("COS"): [bs_x, bs_y]  Taking values in the range [-1, 1]
    """
    assert x.shape[1] == y.shape[1], "The second dimension of x and y must be the same."
    bs_x, bs_y = x.shape[0], y.shape[0]
    dim = x.shape[1]

    if distance_type == "L2":  # The smaller, the more similar
        diff = x.reshape(bs_x, 1, dim) - y.reshape(1, bs_y, dim)  # [bs_x, 1, dim] - [1, bs_y, dim] = [bs_x, bs_y, dim]
        if type(diff) == torch.Tensor:
            dist = torch.sum(diff ** 2, dim=-1)
            dist_xy = torch.sqrt(dist)
        elif type(diff) == np.ndarray:
            dist = np.sum(diff ** 2, axis=-1)
            dist_xy = np.sqrt(dist)
            dist_xy = dist_xy.astype(np.float64)
    elif distance_type == "COS":  # The larger, the more similar (cosine similarity is calculated after normalization)
        if type(x) == torch.Tensor:
            x = x / x.norm(p=2, dim=1, keepdim=True)
            y = y / y.norm(p=2, dim=1, keepdim=True)
            dist_xy = torch.matmul(x, y.t())
        elif type(x) == np.ndarray:
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            dist_xy = np.matmul(x, y.T)
            dist_xy = dist_xy.astype(np.float64)
    return dist_xy  # [bs_x, bs_y]



def InfoNCELoss(output, logit_scale, audio_id=None, distance_type = "COS", args=None, is_train=False):
    '''
    Shape:
        video_feats: [bs, dim]
        audio_feats: [bs, dim]
        logit_scale -> float: scalar
        audio_id -> list of str: len=bs
    Return:
        loss: float
    Reference Code:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L1143
        https://github.com/salesforce/LAVIS/blob/ac8fc98c93c02e2dfb727e24a361c4c309c8dbbc/lavis/models/blip2_models/blip2_qformer.py#L129
        https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py
        https://github.com/mlfoundations/open_clip/blob/73fa7f03a33da53653f61841eb6d69aef161e521/src/open_clip/loss.py#L19
        https://github.com/OFA-Sys/Chinese-CLIP/blob/2746589ee1c9fce925d3b4f70c767e99de47610f/cn_clip/training/train.py#L21
        https://github.com/OFA-Sys/Chinese-CLIP/issues/54
    '''
    # cosine similarity as logits
    logit_scale = logit_scale.exp()
    logits_per_video = output * logit_scale  # [bs, bs]
    logits_per_audio = logits_per_video.t()

    # loss for video to audio
    loss_v2a = 0.0
    if audio_id is not None and is_train and args.ignore_same_music == 0:
        # Ignore same positive pairs
        # audio_id = decode_tensor(audio_id)  # tensor -> list of str
        # Store unique music paths and their corresponding video indices
        music_dict = {}
        for i, path in enumerate(audio_id):
            if path not in music_dict:
                music_dict[path] = []
            music_dict[path].append(i)
        for video_index in range(logits_per_video.shape[0]):
            # Get the indices of videos with the same music path
            same_music_indices = music_dict[audio_id[video_index]]
            # Calculate the negative indices for the current video
            neg_indices = [j for j in range(logits_per_video.shape[1]) if j not in same_music_indices]
            # Get the similarity scores for the positive and negative indices
            sim_v2a_pos = torch.unsqueeze(logits_per_video[video_index, video_index], 0)  # scalar -> [1]
            sim_v2a_neg = logits_per_video[video_index, neg_indices]  # [num_neg]
            # concat
            sim_v2a_signal = torch.cat([sim_v2a_pos, sim_v2a_neg], dim=0)  # [1 + num_neg]
            sim_v2a_signal = sim_v2a_signal.view(1, -1)  # [1, 1 + num_neg]
            # label
            label_v2a_signal = torch.zeros(1, dtype=torch.long, device=sim_v2a_signal.device)
            # compute signal loss
            loss_v2a += F.cross_entropy(sim_v2a_signal, label_v2a_signal)
        loss_v2a /= logits_per_video.shape[0]
    else:
        label_v2a = torch.arange(logits_per_video.shape[0], device = logits_per_video.device)
        loss_v2a = F.cross_entropy(logits_per_video, label_v2a)
    
    # loss for audio to video
    label_a2v = torch.arange(logits_per_audio.shape[0], device = logits_per_audio.device)
    loss_a2v = F.cross_entropy(logits_per_audio, label_a2v)
    loss = (loss_v2a + loss_a2v) / 2
    return loss, logits_per_video, logits_per_audio



