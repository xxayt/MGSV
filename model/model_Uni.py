import torch
import numpy as np
import torch.nn as nn
import time
import torch.nn.functional as F
from modules.loss import *
from music_detr.transformer import MLP
from music_detr.loss_detr import SetCriterion
from model.model_Base import Base_model
from model.model_Base import *
from modules.transformer import Transformer_XA
from modules.metrics import sim_matrix_music_pooling, sim_matrix_video_pooling, sim_matrix_both_pooling

class Uni_model(Base_model):
    def __init__(self, args, device=None, logger=None):
        super(Uni_model, self).__init__(args, device, logger)
        self.args = args
        self.device = device
        self.logger = logger
        assert args.hidden_dim == self.dim_input, "hidden_dim must equal to dim_input"

        # X-Pool for Retrieval
        if "XA" in self.args.vmr_fusion:
            if "music" in self.args.vmr_fusion:
                self.video_guided_to_music_pooling_cross_transformer = Transformer_XA(self.args)
            if "video" in self.args.vmr_fusion:
                self.music_guided_to_video_pooling_cross_transformer = Transformer_XA(self.args)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.temperature_init_value))

        # MML Fusion
        if "CA" in self.args.mml_fusion:
            # cross-attention transformer for Localization
            self.video_music_fusion_cross_transformer = CrossTransformer(
                dim_in = self.dim_input,
                depth = 1,
                heads = 8,
                dim_head = 128,
                mlp_dim = self.cross_hidden_dim,
                dim_out = self.dim_input,
                dropout = 0.8,
                init_method = self.init_method
            )

        # encoder-decoder
        hidden_dim = self.detr_transformer.d_model
        self.num_moment_queries = self.args.num_moment_queries  # 1
        self.decoder_query_embed = nn.Embedding(self.num_moment_queries, hidden_dim)  # object query
        # Localization
        if "detr" in self.args.mml_localization:
            # predict boundary
            span_pred_dim = 1 if self.args.predict_center == 1 else 2
            self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
            self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
            if self.args.moment_loss:
                self.moment_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
            if self.args.contrastive_align_loss:
                contrastive_hdim = self.args.contrastive_dim
                if self.args.audio_short_cut:
                    contrastive_hdim = hidden_dim
                self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
                self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)
            self.aux_loss = args.aux_loss
            # loss
            self.criterion = SetCriterion(self.args, eos_coef = 0.1, temperature = 0.07).to(self.device)
        elif "regression" in self.args.mml_localization:
            span_pred_dim = 1 if self.args.predict_center == 1 else 2
            reg_input_dim = self.dim_input
            self.reg_mlp = MLP(reg_input_dim, 256, span_pred_dim, 3)



    def get_temporal_parameter(self):
        params = []
        params += self.get_projection_parameter()
        params += self.get_SA_parameter()
        return params


    def get_matching_parameter(self):
        params = []
        if "XA" in self.args.vmr_fusion:
            if "music" in self.args.vmr_fusion:
                params += list(self.video_guided_to_music_pooling_cross_transformer.parameters())
            if "video" in self.args.vmr_fusion:
                params += list(self.music_guided_to_video_pooling_cross_transformer.parameters())
        # logit_scale
        params += [self.logit_scale]
        return params


    def get_detection_parameter(self):
        params = []
        # MMD Fusion
        if "CA" in self.args.mml_fusion:
            # cross-attention transformer
            params += list(self.video_music_fusion_cross_transformer.parameters())
        
        # Localization
        if "detr" in self.args.mml_localization:
            # DETR
            params += list(self.detr_transformer.parameters())
            # predict boundary
            params += list(self.span_embed.parameters())
            params += list(self.class_embed.parameters())
            # mlp
            if self.args.moment_loss:
                params += list(self.moment_embed.parameters())
            if self.args.contrastive_align_loss:
                params += list(self.contrastive_align_projection_query.parameters())
                params += list(self.contrastive_align_projection_vid.parameters())
        elif "regression" in self.args.mml_localization:
            params += list(self.reg_mlp.parameters())
        return params


    def calc_output(self, memory, decoder_hidden_states, video_feats, video_feats_mean, audio_feats_mean, width_propotion=None):
        '''
        Inputs:
            memory: torch.tensor, [bs, L_music+L_video or L_music, dim]
            decoder_hidden_states: torch.tensor, [#layers, bs, #Q, dim]
            video_feats: torch.tensor, [bs, max_v_frames, dim]
            video_feats_mean: torch.tensor, [bs, dim]
            audio_feats_mean: torch.tensor, [bs, dim]
            width_propotion: torch.tensor, [bs, #Q, 1]
        Returns:
            output_map: dict
        '''
        output_map = {}
        # class loss
        outputs_class = self.class_embed(decoder_hidden_states)  # [#layers, bs, #Q, #classes=2]
        output_map["pred_logits"] = outputs_class[-1]  # [bs, #Q, #classes=2]
        # spans loss
        outputs_coord = self.span_embed(decoder_hidden_states)  # [#layers, bs, #Q, 2 or 1]
        outputs_coord = outputs_coord.sigmoid()  # span_loss_type = "l1"
        if self.args.predict_center == 1:
            width_propotion = width_propotion.unsqueeze(0).repeat(outputs_coord.shape[0], 1, 1, 1)  # [#layers, bs, #Q, 1]
            outputs_coord = torch.cat([outputs_coord, width_propotion], dim=-1)  # [#layers, bs, #Q, 2]
        output_map["pred_spans"] = outputs_coord[-1]  # [bs, #Q, 2 or 1]
        # contrastive loss
        if self.args.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(decoder_hidden_states), p=2, dim=-1)  # [#layers, bs, #Q, con_dim]
            if self.args.audio_short_cut:
                proj_queries = proj_queries + audio_feats_mean.unsqueeze(1)
                proj_queries = F.normalize(proj_queries, p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(video_feats), p=2, dim=-1)  # [bs, max_v_frames, con_dim]
            output_map.update(dict(
                proj_queries=proj_queries[-1],  # [bs, #Q, con_dim]
                proj_vid_mem=proj_vid_mem  # [bs, max_v_frames, con_dim]
            ))
        # XR want to try
        if self.args.moment_loss:
            moment_feats = self.moment_embed(decoder_hidden_states[-1])  # [bs, #Q, dim]
            moment_feats = F.normalize(moment_feats, p=2, dim=-1)
            if self.args.audio_short_cut:
                moment_feats = moment_feats + audio_feats_mean.unsqueeze(1)
                moment_feats = F.normalize(moment_feats, p=2, dim=-1)
            output_map["moment_feats"] = moment_feats
            output_map["video_feats"] = video_feats_mean
        # auxilary loss
        if self.aux_loss:
            output_map["aux_outputs"] = [{
                "pred_logits": a,
                "pred_spans": b
            } for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            if self.args.contrastive_align_loss:
                assert proj_queries is not None
                for idx, d in enumerate(proj_queries[:-1]):
                    if self.args.audio_short_cut:
                        d = d + audio_feats_mean.unsqueeze(1)
                        d = F.normalize(d, p=2, dim=-1)
                    output_map["aux_outputs"][idx].update(dict(proj_queries=d, proj_vid_mem=proj_vid_mem))
        return output_map



    def forward(self, frame_feats, segment_feats, frame_masks, segment_masks, spans_target, v_duration=None, video_ids=None, music_ids=None, is_train=False):
        '''
        Inputs:
            # videos: [bs, max_v_frames, 3, 224, 224]
            # audios: [bs, max_snippet_num, target_length, mel_bins]

            frame_feats: [bs, max_v_frames, 512]
            frame_masks: [bs, max_v_frames], 1 as valid & 0 as padding
            segment_feats: [bs, max_snippet_num, 768]
            segment_masks: [bs, max_snippet_num]
            spans_target: [bs, 1, 2]
            v_duration: [bs]
            video_ids: [bs]
            music_ids: [bs]
        Returns:
            output_map: dict of tensors
        '''
        # Multi-modal Embedding
        frame_feats, video_feats, frame_masks = self.forward_video_encoder_feature(frame_feats=frame_feats, frame_masks=frame_masks, video_ids=video_ids)  # [bs, max_v_frames, dim_input], [bs, dim_input]
        segment_feats, music_feats, segment_masks = self.forward_audio_encoder_feature(segment_feats=segment_feats, segment_masks=segment_masks, music_ids=music_ids)  # [bs, max_snippet_num, dim_input], [bs, dim_input]

        # Video-to-Music Matching: X-Pool
        if "XA" in self.args.vmr_fusion:
            if "music" in self.args.vmr_fusion:
                music_feats_pooled = self.video_guided_to_music_pooling_cross_transformer(video_feats, segment_feats, segment_masks if self.args.fusion_mask==1 else None)  # [bs_m, bs_v, dim]
            if "video" in self.args.vmr_fusion:
                video_feats_pooled = self.music_guided_to_video_pooling_cross_transformer(music_feats, frame_feats, frame_masks if self.args.fusion_mask==1 else None)  # [bs_v, bs_m, dim]

        # Music Moment Detection
        if "concat" in self.args.mml_fusion:
            segment_feats_fusion = torch.cat([frame_feats, segment_feats], dim=1)  # [bs, max_v_frames+max_snippet_num, dim_input]
            segment_masks_fusion = torch.cat([frame_masks, segment_masks], dim=1)  # [bs, max_v_frames+max_snippet_num]
        elif "CA" in self.args.mml_fusion:
            segment_feats_fusion, attention = self.video_music_fusion_cross_transformer(segment_feats, frame_feats, q_mask=segment_masks, kv_mask=frame_masks)  # [bs, max_snippet_num, dim]
            segment_feats_fusion = segment_feats_fusion.masked_fill(segment_masks.unsqueeze(-1) == 0, 0)  # [bs, max_snippet_num, dim]
            segment_masks_fusion = segment_masks

        # prepare input for DETR, L = max_snippet_num
        detr_kv_input, detr_mask = segment_feats_fusion, segment_masks_fusion
        detr_pos = self.music_position_embedding(segment_feats_fusion, segment_masks_fusion)  # [bs, max_v_frames+max_snippet_num, dim]
        if self.args.moment_query_type == "video":  # decoder content query embed
            moment_query = video_feats.unsqueeze(0).repeat(self.num_moment_queries, 1, 1)  # [#Q, bs, dim_input]
        elif self.args.moment_query_type == "music":
            moment_query = music_feats.unsqueeze(0).repeat(self.num_moment_queries, 1, 1)  # [#Q, bs, dim_input]
        elif self.args.moment_query_type == "xpool":
            moment_query = music_feats_pooled.mean(dim=1).unsqueeze(0).repeat(self.num_moment_queries, 1, 1)  # [#Q, bs, dim_input]
        elif (self.args.moment_query_type == "zero" or self.args.moment_query_type == "random"):
            moment_query = None
        
        decoder_hidden_states, memory = self.detr_transformer(src=detr_kv_input, mask=~(detr_mask.bool()), pos_embed=detr_pos, 
                                                                target=moment_query, query_embed=self.decoder_query_embed.weight)  # [#layers, bs, #Q, dim_input], [bs, max_v_frames+max_snippet_num, dim_input]
        if "regression" in self.args.mml_localization:
            fusion_feats = memory.sum(axis=1) / segment_masks_fusion.sum(axis=1).unsqueeze(-1)  # [bs, 256]
            outputs_coord = self.reg_mlp(fusion_feats)  # [bs, 1 or 2]
            outputs_coord = outputs_coord.sigmoid()
            output_map = dict(pred_spans=outputs_coord.unsqueeze(1))  # [bs, 1, 2]


        # get matching loss
        retrieval_loss = 0
        if self.args.vmr_loss == "dual":
            dual_sim = cal_distance(video_feats, music_feats)
            dual_loss = CLIPLoss(dual_sim, self.logit_scale)
            retrieval_loss = dual_loss * self.args.dual_single_loss_weight
        elif self.args.vmr_loss == "single" and "XA" in self.args.vmr_fusion:
            single_sim = torch.zeros(video_feats.shape[0], music_feats.shape[0]).to(self.device)
            if "oneloss" in self.args.vmr_loss:
                single_sim += sim_matrix_both_pooling(video_feats_pooled, music_feats_pooled)
            else:
                if "music" in self.args.vmr_fusion:
                    bs_music_sim = sim_matrix_music_pooling(video_feats, music_feats_pooled)
                    single_sim += bs_music_sim
                if "video" in self.args.vmr_fusion:
                    bs_video_sim = sim_matrix_video_pooling(video_feats_pooled, music_feats)
                    single_sim += bs_video_sim
            single_loss = CLIPLoss(single_sim, self.logit_scale)
            retrieval_loss = single_loss * self.args.dual_single_loss_weight
        elif self.args.vmr_loss == "dual_single_loss_fuse" and "XA" in self.args.vmr_fusion:
            dual_sim = cal_distance(video_feats, music_feats)
            dual_loss, _, _ = InfoNCELoss(dual_sim, self.logit_scale, audio_id=None, args=self.args, is_train=is_train)
            # dual_loss = CLIPLoss(dual_sim, self.logit_scale)
            single_sim = torch.zeros(video_feats.shape[0], music_feats.shape[0]).to(self.device)
            bs_music_sim = sim_matrix_music_pooling(video_feats, music_feats_pooled)
            single_sim += bs_music_sim
            single_loss = CLIPLoss(single_sim, self.logit_scale)
            retrieval_loss = dual_loss * 1.0 + single_loss * 1.0
        elif self.args.vmr_loss == "dual_single_sim_fuse" and "XA" in self.args.vmr_fusion:
            dual_sim = cal_distance(video_feats, music_feats)
            single_sim = sim_matrix_music_pooling(video_feats, music_feats_pooled)
            overall_sim = dual_sim * 1.0 + single_sim * 1.0
            retrieval_loss = CLIPLoss(overall_sim, self.logit_scale) * self.args.dual_single_loss_weight
        elif self.args.vmr_loss == "dual_single_feature_fuse" and "XA" in self.args.vmr_fusion:
            music_feats_pooled_fused = (music_feats_pooled + music_feats.unsqueeze(1)) * 0.5  # [bs_m, bs_v, dim]
            single_sim = torch.zeros(video_feats.shape[0], music_feats.shape[0]).to(self.device)
            bs_music_sim = sim_matrix_music_pooling(video_feats, music_feats_pooled_fused)
            single_sim += bs_music_sim
            retrieval_loss = CLIPLoss(single_sim, self.logit_scale) * self.args.dual_single_loss_weight
        else:
            raise ValueError(f"Error: vmr_loss={self.args.vmr_loss} and vmr_fusion={self.args.vmr_fusion} is not supported in VMR_model")


        # get detection loss
        localization_loss = 0
        loss_dict = {}
        width_propotion = None
        if self.args.predict_center == 1:
            width_propotion = (v_duration / self.args.max_m_duration).unsqueeze(1).unsqueeze(1)
            width_propotion = width_propotion.repeat(1, self.args.num_moment_queries, 1)  # [bs, #Q, 1]
        if "detr" in self.args.mml_localization:
            output_map = self.calc_output(memory, decoder_hidden_states, frame_feats, video_feats, music_feats, width_propotion=width_propotion)
            loss_dict = self.criterion(output_map, spans_target)
            weight_dict = self.criterion.weight_dict
            localization_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        elif "regression" in self.args.mml_localization:
            if self.args.predict_center == 1:
                output_map["pred_spans"] = torch.cat([output_map["pred_spans"], width_propotion], dim=-1)  # [bs, 1, 2]
            src_spans = output_map["pred_spans"]  # [bs, 1 or 2]
            assert src_spans.shape[-1] == 2 and src_spans.shape[-2] == 1, "pred_spans.shape[-1] must be 2 and shape[-2] must be 1"
            assert spans_target.shape == src_spans.shape, f"spans_target.shape {spans_target.shape} must equal to src_spans.shape f{src_spans.shape}"
            loss_dict['loss_span'] = F.l1_loss(src_spans, spans_target, reduction='mean')
            loss_dict['loss_giou'] = 0
            loss_dict['loss_label'] = 0
            loss_dict['class_error'] = 0
            localization_loss = loss_dict['loss_span'] * 20 + loss_dict['loss_giou'] * 5


        feat_map = {
            "video_feats": video_feats,  # [bs_v, dim]
            "music_feats": music_feats,  # [bs_m, dim]
            "frame_feats": frame_feats,  # [bs_v, max_v_frames, dim]
            "segment_feats": segment_feats,  # [bs_m, max_snippet_num, dim]
        }
        mask_map = {
            "frame_masks": frame_masks,  # [bs_v, max_v_frames]
            "segment_masks": segment_masks  # [bs_m, max_snippet_num]
        }
        id_map = {
            "video_ids": video_ids,  # [bs_v]
            "music_ids": music_ids  # [bs_m]
        }
        loss_map = {
            "retrieval_loss": retrieval_loss,
            "localization_loss": localization_loss,
            "localization_loss_dict": loss_dict
        }
        return output_map, loss_map, feat_map, mask_map, id_map