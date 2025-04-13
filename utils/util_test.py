import torch
import torch.nn as nn
import numpy as np
from modules.loss import cal_distance
import json




def calc_similarity(video_feat_list, audio_feat_list, distance_type="COS"):
    """
    Input:
        video_feat_list: [bs, dim] x (val_len / bs)
        audio_feat_list: [bs, dim] x (val_len / bs)
    Return:
        sim_matrixs: [val_len, val_len]
    """
    sim_matrixs = []
    for video_feat in video_feat_list:  # [bs, 256]
        each_row = []
        for audio_feat in audio_feat_list:  # [bs, 256]
            sim = cal_distance(video_feat, audio_feat, distance_type=distance_type)  # [bs, bs]
            if isinstance(sim, torch.Tensor):
                sim = sim.cpu().detach().numpy()
            each_row.append(sim)
        each_row = np.concatenate(tuple(each_row), axis=-1)  # [bs, val_len]
        sim_matrixs.append(each_row)
    sim_matrixs = np.concatenate(tuple(sim_matrixs), axis=0)  # [val_len, val_len]
    return sim_matrixs


def Recall_metrics(sim_matrix, distance_type="COS", dedup=False, all_music_ids_list=None):
    '''
    Input:
        sim_matrix: [val_len, val_len] - The raw similarity matrix
        all_music_ids_list: [val_len] - The list of actual music IDs corresponding to each video
    Return:
        metrics: dict
        ind: np.array [val_len] - The position of each video's music ID in the sorted matrix
        topk_music_ids: dict - The topk music IDs corresponding to each video
    '''
    # When multiple videos correspond to the same music ID, this part will remove duplicate music IDs 
    # to more accurately assess the ranking of the GT (ground truth) music ID.
    if dedup and len(all_music_ids_list) > 0:
        # Get the indices of the sorted similarity matrix in descending order
        sort_indices = np.argsort(sim_matrix, axis=1)[:, ::-1]  # [val_len, val_len]
        
        ret_results_list = []
        ind = []
        for i, gt_music_id in enumerate(all_music_ids_list):
            seen_music_ids = set()  # Set used to track already encountered music IDs
            sorted_music_ids = [all_music_ids_list[idx] for idx in sort_indices[i]]
            # Find the position of the GT music ID in the sorted list
            for music_id in sorted_music_ids:
                if music_id not in seen_music_ids:
                    seen_music_ids.add(music_id)
                    if music_id == gt_music_id:
                        now_ind = len(seen_music_ids) - 1
                        ind.append(now_ind)  # Add the current music ID's ranking position after deduplication
                        break
            pred_dict_np = dict(
                music_id = gt_music_id,
                rank = now_ind + 1,
                topk_music_ids = sorted_music_ids[:1]
            )
            ret_results_list.append(pred_dict_np)
        ind = np.array(ind)  # [val_len]
        assert len(ind) == len(all_music_ids_list), "len(ind) != len(all_music_ids_list)"
    else:
        # Sort each row in descending order; larger values indicate higher similarity
        sort_sim = np.sort(sim_matrix, axis=1)[:, ::-1]  # Adding [::-1] to sort in descending order
        # Take the diagonal elements, which represent the similarity between each video and its matched music
        diag = np.diag(sim_matrix)[:, np.newaxis]
        # Compute the difference between the sorted matrix and diagonal elements to determine the position of the diagonal elements
        ind = sort_sim - diag
        ind = np.argmax(ind == 0, axis=1)
        # Extract the indices where the elements match the diagonal elements; 
        # these indices represent the position of the matched music in the sorted matrix
        ind = np.where(ind == 0)[1]  # The position of the GT music in the sorted matrix
    
    # Calculate the evaluation metrics
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R3'] = float(np.sum(ind < 3)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['R20'] = float(np.sum(ind < 20)) * 100 / len(ind)
    metrics['R25'] = float(np.sum(ind < 25)) * 100 / len(ind)
    metrics['R50'] = float(np.sum(ind < 50)) * 100 / len(ind)
    metrics['R100'] = float(np.sum(ind < 100)) * 100 / len(ind)
    metrics["MedianR"] = np.median(ind) + 1
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    # Compute MRR (Mean Reciprocal Rank)
    reciprocal_ranks = 1.0 / (ind + 1)
    metrics['MRR'] = np.mean(reciprocal_ranks)
    return metrics, ind, ret_results_list



def IoU_metrics(IoU_list):
    '''
    Input:
        IoU_list: [val_len]
    '''
    metrics = {}
    metrics["mIoU"] = sum(IoU_list) / len(IoU_list)
    metrics["IoU@0.3"] = sum(1 for iou in IoU_list if iou > 0.3) * 100 / len(IoU_list)
    metrics["IoU@0.5"] = sum(1 for iou in IoU_list if iou > 0.5) * 100 / len(IoU_list)
    metrics["IoU@0.7"] = sum(1 for iou in IoU_list if iou > 0.7) * 100 / len(IoU_list)
    return metrics

def individual_IoU(gt_st, gt_ed, gt_m_duration, pred_st, pred_ed, discounted=False):
    '''
    Input:
        gt_st, gt_ed, pred_st, pred_ed: float
    '''
    if gt_st >= gt_ed:
        return 0.0
    pred_st = max(pred_st, 0)
    pred_ed = min(pred_ed, gt_m_duration)
    inter_start = max(gt_st, pred_st)
    inter_end = min(gt_ed, pred_ed)
    # Intersection including Non-negative overlap score
    segments_intersection = max(inter_end - inter_start, 0)
    # Segment union.
    segments_union = (pred_ed - pred_st) + (gt_ed - gt_st) - segments_intersection
    if segments_union <= 0:
        return 0.0
    # Compute overlap as the ratio of the intersection over union of two segments
    iou = segments_intersection / segments_union
    if discounted:
        # discounted ratio
        alpha_st = 1 - abs(gt_st - pred_st) / gt_m_duration
        alpha_ed = 1 - abs(gt_ed - pred_ed) / gt_m_duration
        iou = iou * alpha_st * alpha_ed  # discounted IoU
    return iou


def Composite_metrics(ret_rank_list, IoU_list, mr_results_list=None, all_video_ids_list=None, all_music_ids_list=None):
    '''
    Inputs:
        ret_rank_list: [val_len], include topk rank
        IoU_list: [val_len]
        mr_results_list: list(dict), each dict is {
            "gt_moment": tensor [gt_moment_num=1, 2],
            "m_duration": float,
            "ranked_preds": tensor (#preds, 3)
        }
        all_video_ids_list: [val_len]
        all_music_ids_list: [val_len]
    '''
    RankIoU_metrics = {
        "R1_iou0.5": 0.0,
        "R10_iou0.5": 0.0,
        "R50_iou0.5": 0.0,
        "R100_iou0.5": 0.0,
        "R1_iou0.7": 0.0,
        "R10_iou0.7": 0.0,
        "R50_iou0.7": 0.0,
        "R100_iou0.7": 0.0,
        "R1_miou": 0.0,
        "R10_miou": 0.0,
        "R50_miou": 0.0,
        "R100_miou": 0.0,
    }
    R1_num, R10_num, R50_num, R100_num = 0, 0, 0, 0
    for idx, (ret_rank, IoU, pred_dict, video_id, music_id) in enumerate(zip(ret_rank_list, IoU_list, mr_results_list, all_video_ids_list, all_music_ids_list)):
        
        rank = ret_rank + 1
        if rank == 1:
            RankIoU_metrics["R1_iou0.5"] += IoU > 0.5
            RankIoU_metrics["R1_iou0.7"] += IoU > 0.7
            RankIoU_metrics["R1_miou"] += IoU
            R1_num += 1
        if rank <= 10:
            RankIoU_metrics["R10_iou0.5"] += IoU > 0.5
            RankIoU_metrics["R10_iou0.7"] += IoU > 0.7
            RankIoU_metrics["R10_miou"] += IoU
            R10_num += 1
        if rank <= 50:
            RankIoU_metrics["R50_iou0.5"] += IoU > 0.5
            RankIoU_metrics["R50_iou0.7"] += IoU > 0.7
            RankIoU_metrics["R50_miou"] += IoU
            R50_num += 1
        if rank <= 100:
            RankIoU_metrics["R100_iou0.5"] += IoU > 0.5
            RankIoU_metrics["R100_iou0.7"] += IoU > 0.7
            RankIoU_metrics["R100_miou"] += IoU
            R100_num += 1
    for key in RankIoU_metrics:
        RankIoU_metrics[key] /= len(ret_rank_list)
        if "0." in key:
            RankIoU_metrics[key] *= 100
    RankIoU_metrics["R1_miou"] = RankIoU_metrics["R1_miou"] / R1_num if R1_num > 0 else 0.0
    RankIoU_metrics["R10_miou"] = RankIoU_metrics["R10_miou"] / R10_num if R10_num > 0 else 0.0
    RankIoU_metrics["R50_miou"] = RankIoU_metrics["R50_miou"] / R50_num if R50_num > 0 else 0.0
    RankIoU_metrics["R100_miou"] = RankIoU_metrics["R100_miou"] / R100_num if R100_num > 0 else 0.0
    return RankIoU_metrics


def uni_save_results_json(ret_results_list, loc_results_list, iou_list, save_path):
    """
    Input:
        ret_results_list: list(dict)
        loc_results_list: list(dict)
        save_path: str, json file path
    """
    uni_save_results_list = []
    for idx, (ret_dict, loc_dict, iou) in enumerate(zip(ret_results_list, loc_results_list, iou_list)):
        assert ret_dict["music_id"] == loc_dict["music_id"]
        uni_pred_map = dict(
            video_id = loc_dict["video_id"],
            music_id = ret_dict["music_id"],
            topk_mids = ret_dict["topk_music_ids"],
            gt_mid_rank = ret_dict["rank"],
            iou = round(iou.item(), 4),  # tensor to float
            m_duration = loc_dict["m_duration"],
            gt_st = round(loc_dict["gt_moment"][0][0], 3),
            gt_ed = round(loc_dict["gt_moment"][0][1], 3),
            pred_st = round(max(loc_dict["pred_st"], 0), 3),
            pred_ed = round(min(loc_dict["pred_ed"], 240), 3),
        )
        uni_save_results_list.append(uni_pred_map)
    with open(save_path, "w") as f:
        json.dump(uni_save_results_list, f, indent=4)