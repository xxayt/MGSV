import os
from torch.utils.data import Dataset
import torch
import pandas as pd

class MGSV_EC_DataLoader(Dataset):
    def __init__(
            self,
            csv_path,
            args=None,
    ):
        self.args = args
        self.csv_data = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.csv_data)

    def get_cw_propotion(self, gt_spans, max_m_duration):
        '''
        Inputs:
            gt_spans: [1, 2]
            max_m_duration: float
        '''
        gt_spans[:, 1] = torch.clamp(gt_spans[:, 1], max=max_m_duration)
        center_propotion = (gt_spans[:, 0] + gt_spans[:, 1]) / 2.0 / max_m_duration  # [1]
        width_propotion = (gt_spans[:, 1] - gt_spans[:, 0]) / max_m_duration  # [1]
        return torch.stack([center_propotion, width_propotion], dim=-1)  # [1, 2]

    def __getitem__(self, idx):
        # id
        video_id = self.csv_data['video_id'].to_numpy()[idx]
        music_id = self.csv_data['music_id'].to_numpy()[idx]
        # duration
        # v_duration = self.csv_data['video_total_duration'].to_numpy()[idx]
        m_duration = self.csv_data['music_total_duration'].to_numpy()[idx]
        m_duration = float(m_duration)
        # video moment st, ed
        video_start_time = self.csv_data['video_start'].to_numpy()[idx]
        video_end_time = self.csv_data['video_end'].to_numpy()[idx]
        # music moment
        music_start_time = self.csv_data['music_start'].to_numpy()[idx]
        music_end_time = self.csv_data['music_end'].to_numpy()[idx]
        gt_windows_list = [(music_start_time, music_end_time)]
        gt_windows = torch.Tensor(gt_windows_list)  # [1, 2]
        # time map
        meta_map = {
            "video_id": str(video_id),
            "music_id": str(music_id),
            "v_duration": torch.tensor(video_end_time - video_start_time),
            "m_duration": torch.tensor(m_duration),
            "gt_moment": gt_windows,  # [1, 2]
        }
        # target spans
        spans_target = self.get_cw_propotion(gt_windows, self.args.max_m_duration)  # [1, 2]

        # extract features
        video_feature_path = os.path.join(self.args.frame_frozen_feature_path, 'vit_feature', f'{video_id}.pt')
        video_mask_path = os.path.join(self.args.frame_frozen_feature_path, 'vit_mask', f'{video_id}.pt')
        frame_feats = torch.load(video_feature_path, map_location='cpu')
        frame_mask = torch.load(video_mask_path, map_location='cpu')
        frame_feats = frame_feats.masked_fill(frame_mask.unsqueeze(-1) == 0, 0)  # [bs, max_frame_num, 512]

        music_feature_path = os.path.join(self.args.music_frozen_feature_path, 'ast_feature', f'{music_id}.pt')
        music_mask_path = os.path.join(self.args.music_frozen_feature_path, 'ast_mask', f'{music_id}.pt')
        segment_feats = torch.load(music_feature_path, map_location='cpu')
        segment_mask = torch.load(music_mask_path, map_location='cpu')  
        segment_feats = segment_feats.masked_fill(segment_mask.unsqueeze(-1) == 0, 0)  # [bs, max_snippet_num, 768]

        data_map = {
            "frame_feats": frame_feats,
            "frame_mask": frame_mask,
            "segment_feats": segment_feats,
            "segment_mask": segment_mask,
        }
        return data_map, meta_map, spans_target