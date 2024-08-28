import os
import pickle
import glob
import numpy as np
import torch
import torchvision
from PIL import ImageFile
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd

from opensora.registry import DATASETS

from .utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video, temporal_random_crop

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_FPS = 120

#configure
MAX_NUM_VIDEOS = 100

@DATASETS.register_module()
class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        num_frames=16,
        frame_interval=1,
        image_size=(256, 256),
        transform_name="center",
    ):
        self.data_path = data_path
        self.data = self._load_pickle_files(data_path)
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }

    # TODO: @levi make this shit faster bruh
    def _load_pickle_files(self, data_path):
        data = []
        pkl_files = []
        
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith('.pkl'):
                    pkl_files.append(os.path.join(root, file))
        
        for file_path in tqdm(pkl_files, desc="Loading .pkl files", unit="file"):
            with open(file_path, "rb") as f:
                data.append(pickle.load(f))
            if len(data) >= MAX_NUM_VIDEOS:
                break
        
        df = pd.DataFrame(data)
        return df

    def _print_data_number(self):
        num_videos = len(self.data)
        print(f"Dataset contains {num_videos} video entries.")

    # only supporting videos for now
    def get_type(self, video_info):
        return "video"

    def getitem(self, index):
        sample = self.data.iloc[index]
        video_info = sample['video_info']
        video_path = sample['video_path']
        text = video_info.get('caption', '')

        file_type = self.get_type(video_info)

        if file_type == "video":
            rel_path = video_path.split("nba-plus-statvu-dataset")[1]
            path = os.path.join('/mnt/mir/fan23j/data/nba-plus-statvu-dataset', rel_path)
            # loading
            video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 30
            vframes, _, _ = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")

            # Sampling video frames
            video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)

            # transform
            transform = self.transforms["video"]
            video = transform(video)  # T C H W
        else:
            image = pil_loader(video_path)
            video_fps = IMG_FPS
            transform = self.transforms["image"]
            image = transform(image)
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        return {"video": video, "text": text, "fps": video_fps}

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                video_path = self.data.iloc[index]['video_path']
                print(f"data {video_path}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)

@DATASETS.register_module()
class VariableVideoTextDataset(VideoTextDataset):
    def __init__(
        self,
        data_path,
        num_frames=None,
        frame_interval=1,
        image_size=None,
        transform_name=None,
    ):
        super().__init__(data_path, num_frames, frame_interval, image_size, transform_name=None)
        self.transform_name = transform_name
        # for i, item in enumerate(self.data):
        #     item['id'] = i
        self.data["id"] = np.arange(len(self.data))

    def get_data_info(self, index):
        video_info = self.data.iloc[index]['video_info']
        H = video_info.get('height', 0)
        W = video_info.get('width', 0)
        T = len(self.data.iloc[index]['frames'])
        # H = 720
        # W = 1280
        return T, H, W

    def getitem(self, index):
        # a hack to pass in the (time, height, width) info from sampler
        index, num_frames, height, width = [int(val) for val in index.split("-")]

        sample = self.data.iloc[index]
        video_info = sample['video_info']
        video_path = sample['video_path']
        text = video_info.get('caption', '')
        file_type = self.get_type(video_info)
        ar = height / width

        video_fps = video_info.get('fps', 30)  # default fps to 30 if not present
        if file_type == "video":
            rel_path = video_path.split("filtered_clips")[1]
            path = os.path.join('/mnt/mir/fan23j/data/nba-plus-statvu-dataset/filtered_clips', rel_path)
            # loading
            vframes, _, infos = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
            if "fps" in infos:
                video_fps = infos["fps"]

            # Sampling video frames
            video = temporal_random_crop(vframes, num_frames, self.frame_interval)

            # transform
            transform = get_transforms_video(self.transform_name, (height, width))
            video = transform(video)  # T C H W
        else:
            image = pil_loader(video_path)
            video_fps = IMG_FPS

            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        return {
            "video": video,
            "text": text,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
        }