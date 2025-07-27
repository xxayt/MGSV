import torch
from math import fabs

def span_se_to_cw(se_spans):
    """ start-end turn to center-width
    Inputs:
        se_spans: [#windows, 2] (tensor)
    Returns:
        cw_spans: [#windows, 2] (tensor)
    """
    center = se_spans.sum(-1) * 0.5
    width = se_spans[:, 1] - se_spans[:, 0]
    return torch.stack([center, width], dim=-1)

def span_cw_to_se(cw_spans):
    """ center-width turn to start-end
    Inputs:
        cw_spans: [#windows, 2] (tensor)
    Returns:
        se_spans: [#windows, 2] (tensor)
    """
    start = cw_spans[:, 0] - 0.5 * cw_spans[:, 1]
    end = cw_spans[:, 0] + 0.5 * cw_spans[:, 1]
    return torch.stack([start, end], dim=-1)

# def span_c_to_se(c_spans, w_spans):
#     """ center-width turn to start-end
#     Inputs:
#         c_spans: [#windows] (tensor)
#         w_spans: [#windows] (tensor)
#     Returns:
#         se_spans: [#windows, 2] (tensor)
#     """
#     start = c_spans - 0.5 * w_spans
#     end = c_spans + 0.5 * w_spans
#     return torch.stack([start, end], dim=-1)


def temporal_iou(spans1, spans2):
    """
    Inputs:
        spans1: (N, 2) torch.Tensor, each row defines a span [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        iou: (N, M) torch.Tensor
        union: (N, M) torch.Tensor
    >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
    >>> temporal_iou(test_spans1, test_spans2)
    (tensor([[0.6667, 0.2000],
         [0.0000, 0.5000]]),
     tensor([[0.3000, 1.0000],
             [0.8000, 1.0000]]))
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N, )
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M, )

    left = torch.max(spans1[:, None, 0], spans2[:, 0])  # (N, M)
    right = torch.min(spans1[:, None, 1], spans2[:, 1])  # (N, M)

    inter = (right - left).clamp(min=0)  # (N, M)
    union = areas1[:, None] + areas2 - inter  # (N, M)

    iou = inter / union
    return iou, union


def temporal_intersection_over_pred(gt_spans, pred_spans):
    """ intersection over the second input spans
    Args:
        gt_spans: (N, 2),
        pred_spans: (M, 2)

    Returns:

    """
    left = torch.max(gt_spans[:, None, 0], pred_spans[:, 0])
    right = torch.min(gt_spans[:, None, 1], pred_spans[:, 1])

    inter = (right - left).clamp(min=0)  # (N, M)
    inter_over_pred = inter / (pred_spans[:, 1] - pred_spans[:, 0])
    return inter_over_pred


def generalized_temporal_iou(spans1, spans2):
    """
    Generalized IoU from https://giou.stanford.edu/
    Also reference to DETR implementation of generalized_box_iou
    https://github.com/facebookresearch/detr/blob/master/util/box_ops.py#L40

    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span in xx format [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        giou: (N, M) torch.Tensor

    >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
    >>> generalized_temporal_iou(test_spans1, test_spans2)
    tensor([[ 0.6667,  0.2000],
        [-0.2000,  0.5000]])
    """
    spans1 = spans1.float()
    spans2 = spans2.float()
    assert (spans1[:, 1] >= spans1[:, 0]).all()
    assert (spans2[:, 1] >= spans2[:, 0]).all()
    iou, union = temporal_iou(spans1, spans2)

    left = torch.min(spans1[:, None, 0], spans2[:, 0])  # (N, M)
    right = torch.max(spans1[:, None, 1], spans2[:, 1])  # (N, M)
    enclosing_area = (right - left).clamp(min=0)  # (N, M)

    return iou - (enclosing_area - union) / enclosing_area



def individual_IoU_tensor(gt_st, gt_ed, gt_m_duration, pred_st, pred_ed, discounted=False):
    '''
    Input:
        gt_st, gt_ed, pred_st, pred_ed: torch.Tensor
    Output:
        iou: torch.Tensor
    '''
    if torch.any(gt_st >= gt_ed):
        return torch.tensor(0.0)
    pred_st = torch.clamp(pred_st, min=0)
    pred_ed = torch.clamp(pred_ed, max=gt_m_duration)
    inter_start = torch.max(gt_st, pred_st)
    inter_end = torch.min(gt_ed, pred_ed)
    # Intersection including Non-negative overlap score
    segments_intersection = torch.max(inter_end - inter_start, torch.tensor(0))
    # Segment union.
    segments_union = (pred_ed - pred_st) + (gt_ed - gt_st) - segments_intersection
    if torch.any(segments_union <= 0):
        return torch.tensor(0.0)
    # Compute overlap as the ratio of the intersection over union of two segments
    iou = segments_intersection / segments_union
    if discounted:
        # discounted ratio
        alpha_st = 1 - torch.abs(gt_st - pred_st) / gt_m_duration
        alpha_ed = 1 - torch.abs(gt_ed - pred_ed) / gt_m_duration
        iou = iou * alpha_st * alpha_ed  # discounted IoU
    return iou

def detr_iou(args, mr_results_list):
    '''
    Input:
        mr_results_list: list(dict), each dict is {
            "gt_moment": tensor [gt_moment_num=1, 2],
            "m_duration": float,
            "ranked_preds": tensor [#preds, 3]
        }
    '''
    IoU_list = []
    for idx, pred_dict in enumerate(mr_results_list):
        ranked_preds = pred_dict["ranked_preds"][0]
        pred_st, pred_ed = ranked_preds[0], ranked_preds[1]
        pred_st = torch.clamp(pred_st, min=0)
        pred_ed = torch.clamp(pred_ed, max=args.max_m_duration)
        gt_moment = pred_dict["gt_moment"]  # [1, 2]
        m_duration = pred_dict["m_duration"]
        
        iou = torch.tensor(0.0)
        gt_m = gt_moment[0]
        gt_st, gt_ed = gt_m[0], gt_m[1]
        iou = individual_IoU_tensor(gt_st, gt_ed, m_duration, pred_st, pred_ed)
        IoU_list.append(iou)
    return IoU_list