# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from music_detr.span_utils import generalized_temporal_iou, span_cw_to_se


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, args, cost_class: float = 1, cost_span: float = 1, cost_giou: float = 1,
                 span_loss_type: str = "l1", snippet_num: int = 100):
        """Creates the matcher
        Inputs:
            cost_span: This is the relative weight of the L1 error of the span coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the spans in the matching cost
        """
        super().__init__()
        self.args = args
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.span_loss_type = span_loss_type
        self.snippet_num = snippet_num
        self.foreground_label = 0 if self.args.fb_label == "01" else 1
        assert cost_class != 0 or cost_span != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Inputs:
            outputs: (dict)
                    "pred_logits": Tensor of dim [bs, num_queries, num_classes=2] with the classification logits
                    "pred_spans": Tensor of dim [bs, num_queries, 2] with the predicted span coordinates, in normalized (c, w) format
            targets: (list), len(targets) = bs, where each target is a dict containing:
                    "spans": Tensor of dim [num_target_spans, 2] containing the target span coordinates. The spans are in normalized (cx, w) format
        Returns:
            A list of size bs, containing tuples of (index_i, index_j) where:
                index_i is the indices of the selected predictions (in order)
                index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_spans)
        """
        bs, num_queries = outputs["pred_spans"].shape[:2]
        # targets = targets["span_labels"]



        # Also concat the target labels and spans
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [bs * #queries, num_classes]
        moment_mask = (targets[:, :, 1] != 0)  # [bs, gt_moment_num]
        tgt_spans = targets[moment_mask]  # [num_target_spans in batch, 2]
        sizes = moment_mask.sum(dim=1).tolist()  # [num_target_spans in batch]

        tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)   # [total #spans in the batch]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]  # [bs * #queries, total #spans]

        if self.span_loss_type == "l1":
            # We flatten to compute the cost matrices in a batch
            out_spans = outputs["pred_spans"].flatten(0, 1)  # [bs * num_queries, 2]

            # Compute the L1 cost between spans
            cost_span = torch.cdist(out_spans.float(), tgt_spans.float(), p=1)  # [bs * #queries, total #spans]

            # Compute the giou cost between spans
            cost_giou = - generalized_temporal_iou(span_cw_to_se(out_spans), span_cw_to_se(tgt_spans))  # [bs * #queries, total #spans]
        else:
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, snippet_num * 2)
            pred_spans = pred_spans.view(bs * num_queries, 2, self.snippet_num).softmax(-1)  # (bsz * #queries, 2, snippet_num)
            cost_span = - pred_spans[:, 0][:, tgt_spans[:, 0]] - \
                pred_spans[:, 1][:, tgt_spans[:, 1]]  # (bsz * #queries, #spans)
            # giou
            cost_giou = 0

        # Final cost matrix
        C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()  # [bs, num_queries, total #spans in the batch]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]  # i is pred_idx, j is target_idx


def build_matcher(args):
    return HungarianMatcher(
        args,
        cost_span=10,
        cost_giou=1,
        cost_class=4,
        span_loss_type=args.span_loss_type,
        snippet_num=args.max_snippet_num
    )
