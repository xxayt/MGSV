import torch as th
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# pip install opencv-python
import cv2


class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=-1, ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)
        # self.transform_test_video_aug = self._transform_test_video_aug(self.size)

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),#caz 
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.
        #VideoCapture是用于从视频文件、图片序列、摄像头捕获视频的类
        # print(video_file)
        cap = cv2.VideoCapture(video_file)
        #统计frame_num
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #统计帧率：每秒显示的帧数
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1

        inds = [ind for ind in np.arange(0, fps, interval)]
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        ret = True
        #start 切帧
        images, included = [], []

        for sec in np.arange(start_sec, end_sec + 1):
            if not ret: break
            sec_base = int(sec * fps)
            for ind in inds:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #Image.fromarray：实现array到image的转换
                images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))  # num_frames * [3, 224, 224]
                
        #         print('frame_rgb.shape {}'.format(frame_rgb.shape))
        #         print(Image.fromarray(frame_rgb).convert("RGB"))
        # print('*'*20)    
        # print(len(images))
        # print('*'*20)

        cap.release()

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        #video返回的是一个tensor,大小为：[frame_nums,h,w,3]
        return {'video': video_data}
    #caz
    def sort_str(self,img_name_list):#caz
        dict_img = {}
        for item in img_name_list:
            # print(item )
            frame_index = item.split('_')[-1].split('.jpg')[0]
            dict_img[item]=int(frame_index)

        dict_img = dict(sorted(dict_img.items(), key=lambda x:x[1],reverse = False))
        sort_img_name_list = list(dict_img.keys())
        return sort_img_name_list

    # 从frame_path中读取全部图片，然后转换为tensor
    def image_to_tensor(self, frame_path, preprocess, start_time=None, end_time=None):
        frame_imgs = os.listdir(frame_path)
        frame_imgs = self.sort_str(frame_imgs)
        if len(frame_imgs)>256:
            frame_imgs = np.array(frame_imgs)
            sample_indx = np.linspace(0, len(frame_imgs)-1, num=128, dtype=int)
            frame_imgs =frame_imgs[sample_indx]

        images, included = [], []
        for frame in frame_imgs:
            raw_frame = cv2.imread(os.path.join(frame_path,frame)) 
            raw_frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            images.append(preprocess(Image.fromarray(raw_frame_rgb).convert("RGB")))#caz
        
        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return {'video': video_data}
    

    def get_video_data(self, video_path, start_time=None, end_time=None):
        #if 判断video_path是不是一个文件夹，是一个文件夹的话就用image_to_tensor
        temp_flag = os.path.isdir(video_path)  # 判断是否为文件夹
        if temp_flag:
            if(len(os.listdir(video_path))>0):
                try:
                    image_input = self.image_to_tensor(video_path, self.transform)
                except Exception as e:
                    image_input = None
            else:
                image_input = None
            # print('image_input is {}'.format(len(image_input['video']))) 
        else:
            try:
                image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
                # print('image_input is {}'.format(len(image_input['video'])))
            except Exception as e:
                print("!!!!{} cv read error".format(video_path))
                image_input = None
            
        return image_input


    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2