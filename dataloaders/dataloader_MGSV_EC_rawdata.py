import os
from torch.utils.data import Dataset
import torch
import math
import torchaudio
from torchvision.transforms import Compose, Resize, InterpolationMode, CenterCrop, ToTensor, Normalize
from PIL import Image
import numpy as np
import pandas as pd
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform_clip(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


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
    
    # Padding 0 for max_v_frames
    def get_clip_frame(self, video_id, frame_path, video_start_time, video_end_time, max_v_frames=50):
        '''
        Loads video frames from the specified directory, selects a subset of frames based on the given time range,
        and pads the frames to a fixed number of frames (`max_v_frames`). It applies a transformation to each image to prepare
        them for model input.

        Args:
            video_id (str): The unique identifier for the video.
            frame_path (str): The path to the directory containing the video frames (as JPEG images).
            video_start_time (float): The start time (in seconds) for the frame selection.
            video_end_time (float): The end time (in seconds) for the frame selection.
            max_v_frames (int): The maximum number of frames to be selected from the video. If there are fewer frames, they will be padded to this number. Default is 50.

        Returns:
            video (Tensor): A tensor containing the transformed frames. Shape is [max_v_frames, 3, 224, 224].
            video_mask (Tensor): A tensor representing the presence of valid frames (1 for valid frames, 0 for padded frames). Shape is [max_v_frames].

        Example:
            video, video_mask = self.get_clip_frame(video_id, frame_path, video_start_time, video_end_time, max_v_frames=self.args.max_v_frames)
        '''

        transform = _transform_clip(self.image_resolution)
        path_frame_num = len(os.listdir(frame_path))  # 0 ~ path_frame_num-1
        # choose frames
        video_start_time = math.floor(video_start_time)  # Because all video_start_time < 0.5
        video_end_time = math.floor(video_end_time)
        video_end_time = min(video_end_time, min(path_frame_num-1, max_v_frames-1))
        assert video_end_time - video_start_time + 1 <= max_v_frames, f"video_end_time - video_start_time+1: {video_end_time - video_start_time + 1} > max_v_frames: {max_v_frames}, video_id: {video_id}"
        indices = [i for i in range(video_start_time, video_end_time + 1)]
        video_mask = torch.zeros(max_v_frames)
        # load images
        images = []
        for idx, i in enumerate(indices):
            image_name = f"{i}.jpg"
            if i == path_frame_num - 1 and os.path.exists(os.path.join(frame_path, image_name)) is False and os.path.exists(os.path.join(frame_path, "end.jpg")) is True:
                image_name = "end.jpg"
            image_path = os.path.join(frame_path, image_name)
            if not os.path.exists(image_path):
                raise RuntimeError(f"{frame_path} Failed to read image: {image_path}")
            else:
                image = Image.open(image_path)
            image = transform(image)  # [3, 224, 224]
            images.append(image)
            video_mask[idx] = 1
        # padding 0
        already_frames = len(images)
        for i in range(already_frames+1, max_v_frames+1):
            image = Image.new('RGB', (self.image_resolution, self.image_resolution), (0, 0, 0))
            images.append(transform(image))
        assert len(images) == max_v_frames, f"len(images): {len(images)} != max_v_frames: {max_v_frames}, video_id: {video_id}"
        video = torch.stack(images)  # [max_v_frames, 3, 224, 224]
        return video, video_mask  # [max_v_frames, 3, 224, 224], [max_v_frames]

    # Sliding Window with Overlap
    def get_ast_rawaudio(self, music_path, stride=2.5, filter=10.0, padding=0, max_m_duration=240, mel_bins=128, target_length=1024):
        '''
        Loads an audio file, resamples it to a fixed sample rate, and splits it into overlapping segments using a sliding window.
        Each segment is then transformed into a Mel-spectrogram feature with a fixed target length.

        Args:
            music_path (str): Path to the audio file (e.g., WAV, MP3).
            stride (float): The step size (in seconds) for the sliding window. Default is 2.5 seconds.
            filter (float): The window size for each segment (in seconds). Default is 10.0 seconds.
            padding (float): Amount of padding (in seconds) to add before and after each segment. Default is 0 seconds.
            max_m_duration (float): Maximum duration (in seconds) of the audio. If the audio is shorter, it will be padded. Default is 240 seconds.
            mel_bins (int): The number of Mel filter banks to use for feature extraction. Default is 128.
            target_length (int): The target length of each Mel-spectrogram feature after padding or trimming. Default is 1024 (approx. 10.26 seconds).

        Returns:
            audio (Tensor): A tensor containing the Mel-spectrogram features of the audio segments. Shape is [max_snippet_num, target_length, mel_bins].
            audio_mask (Tensor): A tensor representing the presence of valid segments (1 for valid, 0 for padded). Shape is [max_snippet_num].
        
        Example:
            audio, audio_mask = self.get_ast_rawaudio(music_path, stride=self.args.stride, filter=self.args.filter, padding=self.args.padding, max_m_duration=self.args.max_m_duration)
        '''
        
        waveform, origin_sample_rate = torchaudio.load(music_path)
        # fixed sample rate
        target_sample_rate = 16000
        if target_sample_rate != origin_sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=origin_sample_rate, new_freq=target_sample_rate)
        # fixed audio length
        m_duration = waveform.shape[1] / target_sample_rate
        if m_duration < max_m_duration:
            pad_sample_num = int(target_sample_rate * max_m_duration) - waveform.shape[1]
            waveform = torch.cat([waveform, torch.zeros(waveform.shape[0], pad_sample_num)], 1)
        else:
            waveform = waveform[:, 0:int(target_sample_rate * max_m_duration)]
        
        # snippet/segment audio
        audio_segments = []
        assert self.args.max_snippet_num == int(max_m_duration / stride), f"args.max_snippet_num: {self.args.max_snippet_num} != max_m_duration // stride: {max_m_duration // stride}"
        audio_mask = torch.zeros(self.args.max_snippet_num)
        for snippet_num, center in enumerate(np.arange(0, max_m_duration, stride)):
            start = max(0 - padding, center - filter / 2)
            end = min(max_m_duration + padding, center + filter / 2)
            # set audio mask
            if center <= m_duration:
                audio_mask[snippet_num] = 1
            start_sample_num = int(target_sample_rate * start)
            end_sample_num = int(target_sample_rate * end)
            waveform_snippet = waveform[:, start_sample_num:end_sample_num]  # snippet: [start, end]
            # mel feature extraction
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform_snippet, htk_compat=True, sample_frequency=target_sample_rate, use_energy=False,
                window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)  # 默认 frame_length=25. final dim(1024)=1+(duration*16000-400)/160, 其中400=25ms*16000/1000, 160=10ms*16000/1000
            # fixed target length by AST: 1024 samples = 10.26s
            if target_length > fbank.shape[0]:  # equ to: 10.26s > filter
                m = torch.nn.ZeroPad2d((0, 0, 0, target_length - fbank.shape[0]))
                fbank = m(fbank)
            elif target_length < fbank.shape[0]:
                fbank = fbank[0:target_length, :]
            assert fbank.shape[0] == target_length, f'fbank length {fbank.shape[0]} must be equal to target_length {target_length}'
            # normalize
            fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)  # [target_length, mel_bins]
            audio_segments.append(fbank)
        audio = torch.stack(audio_segments)
        return audio, audio_mask  # [max_snippet_num, target_length, mel_bins], [max_snippet_num]


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