import torch
import torch.nn.functional as F
from torch import nn

from music_detr.span_utils import generalized_temporal_iou, span_cw_to_se
from music_detr.misc import accuracy
from music_detr.matcher import build_matcher



class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, args, eos_coef, temperature):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            snippet_num: int,
        """
        super().__init__()
        self.args = args
        self.matcher = build_matcher(args)
        self.temperature = temperature
        self.span_loss_type = args.span_loss_type  # "l1"
        self.snippet_num = args.max_snippet_num

        self.weight_dict = {"loss_span": 4,
                            "loss_giou": 1,
                            "loss_label": 0.8}
        if args.contrastive_align_loss:
            self.weight_dict.update({"loss_contrastive_align": 0.2})
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.detr_dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in self.weight_dict.items() if k != "loss_saliency"})
            self.weight_dict.update(aux_weight_dict)

        # foreground and background classification
        if args.fb_label == "01":
            self.foreground_label = 0
            self.background_label = 1
        elif args.fb_label == "10":
            self.foreground_label = 1
            self.background_label = 0
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        # empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        empty_weight[self.background_label] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        # (src, tgt) = indices[0]
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs

        # 根据indices获取对应取出pred_spans和tgt_spans
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # [total #spans, 2]
        losses = {}
        tgt_spans = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)  # [total #spans, 2]
        if self.args.l1_loss:
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            losses['loss_span'] = loss_span.mean()
        loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cw_to_se(src_spans), span_cw_to_se(tgt_spans)))
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # [bs, #queries, #classes=2]
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)  # [total #spans = bs]
        target_classes = torch.full(src_logits.shape[:2], self.background_label, dtype=torch.int64, device=src_logits.device)  # [bs, #queries]
        target_classes[idx] = self.foreground_label  # 1

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses


    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_vit_embed = outputs["proj_vid_mem"]  # [bs, max_v_frames, con_dim]  frame tokens
        normalized_music_embed = outputs["proj_queries"]  # [bs, #queries, con_dim]
        logits = torch.einsum("bmd,bnd->bmn", normalized_music_embed, normalized_vit_embed)  # [bs, #queries, max_v_frames]
        logits = logits.sum(2) / self.temperature  # [bs, #queries]
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)  # [bs, #queries]
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)  # [bs, #queries] 只有正样本的地方有值，其他地方为0

        pos_term = positive_logits.sum(1)  # [bs]  计算正样本的值
        num_pos = positive_map.sum(1)  # [bs] 计算正样本的个数
        neg_term = logits.logsumexp(1)  # [bs] 计算所有样本的值
        loss_nce = - pos_term / num_pos + neg_term  # [bs]
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Inputs:
            outputs: dict of tensors, auxiliary losses is optional
                "pred_logits": [bs, #queries, #classes=2]
                "pred_spans": [bs, #queries, 2]
                "aux_outputs": list of dicts, including 'pred_logits', 'pred_spans'
            targets: [bs, gt_moment_nums, 2]
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)
        # indices: [(sample0 prediction, sample0 target), (sample1 prediction, sample1 target), ...]
        indices = self.matcher(outputs_without_aux, targets)  # list of tensors, each tensor is the indices of the selected predictions/targets

        # Compute all the requested losses
        loss_map = {}
        self.loss_type_map = {
            "spans": self.loss_spans,  # return "loss_span", "loss_giou"
            "labels": self.loss_labels,  # return "loss_label", "class_error"
            "contrastive": self.loss_contrastive_align,  # return "loss_contrastive_align"
        }
        for loss_name, loss_func in self.loss_type_map.items():
            if loss_name == "contrastive" and self.args.contrastive_align_loss == 0 and self.args.moment_loss == 0:
                continue
            loss_map.update(loss_func(outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):  # #layers
                indices = self.matcher(aux_outputs, targets)
                for loss_name, loss_func in self.loss_type_map.items():
                    if loss_name == "contrastive" and self.args.contrastive_align_loss == 0 and self.args.moment_loss == 0:
                        continue
                    kwargs = {}
                    l_dict = loss_func(aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    loss_map.update(l_dict)
        return loss_map