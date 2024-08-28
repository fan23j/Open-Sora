import os
import numpy as np
import torch
import torchvision

from nba.src.statvu_align.entities.clip_dataset import FilteredClipDataset
from nba.src.statvu_align.entities.clip_annotations import (
    Bbox,
    ClipAnnotationWrapper,
    Frame,
)
from typing import Optional, List, Tuple, Any, Dict
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from opensora.utils.sampler_entities import MicroBatch
from opensora.registry import DATASETS
from .utils import (
    VID_EXTENSIONS,
    get_transforms_image,
    get_transforms_video,
    read_file,
    temporal_random_crop,
    extract_conditions,
)

IMG_FPS = 120
DATASET_DIR = (
    "/mnt/mir/levlevi/nba-plus-statvu-dataset/filtered-clip-annotations-40-bbx-ratios"
)
TYPE_VIDEO = "video"
TYPE_IMAGE = "image"


class NBAClipsConditions:
    def __init__(self, data: Dict):
        self.bbox_ratios: np.ndarray = data["bbox_ratios"]


class NBAClipsSample:
    def __init__(self, data: Dict):
        self.video: torch.Tensor = data["video"]
        self.text: str = data["text"]
        self.num_frames: int = data["num_frames"]
        self.height: int = data["height"]
        self.width: int = data["width"]
        self.ar: float = data["ar"]
        self.fps: int = data["fps"]
        self.conditions: NBAClipsConditions = NBAClipsConditions(data["conditions"])


class NBAClipsDataset(torch.utils.data.Dataset):
    """
    Dataset for NBAClips.
    """

    def __init__(
        self,
        data_path: str,
        num_frames: int = 16,
        frame_interval: int = 1,
        image_size: Tuple[int, int] = (256, 256),
        transform_name: Optional[str] = "center",
    ):
        self.data_path: str = data_path
        self.filtered_ds = FilteredClipDataset(DATASET_DIR)
        
        # HACK: using one sample
        self.annotation_file_paths: List[str] = (
            self.filtered_ds.filtered_clip_annotations_file_paths
        )
        self.annotation_file_paths: List[str] = self.annotation_file_paths[:1]
        
        self.num_frames: int = num_frames
        self.frame_interval: int = frame_interval
        self.image_size: Tuple[int, int] = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }
        self.ann: ClipAnnotationWrapper = ClipAnnotationWrapper(self.annotation_file_paths[0])

    def __getitem__(self, index: int):

        sample = self.annotation_file_paths[index]
        sample_ann_wrapper = ClipAnnotationWrapper(sample)
        self.ann = sample_ann_wrapper.clip_annotation
        path: str = sample_ann_wrapper.video_fp
        text = self.ann.video_info.caption
        file_type = self.ann.video_info.file_type

        if file_type == TYPE_VIDEO:
            # loading
            vframes, _, _ = torchvision.io.read_video(
                filename=path, pts_unit="sec", output_format="TCHW"
            )
            # Sampling video frames
            video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)
            # transform
            transform = self.transforms["video"]
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)
            # transform
            transform = self.transforms["image"]
            image = transform(image)
            # repeat
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        return {"video": video, "text": text}

    def __len__(self) -> int:
        return len(self.annotation_file_paths)


class VariableNBAClipsDataset(NBAClipsDataset):
    def __init__(
        self,
        data_path: str,
        num_frames: Optional[int] = None,
        frame_interval: int = 1,
        image_size: Optional[Tuple[int, int]] = None,
        transform_name: Optional[str] = None,
    ):
        super().__init__(
            data_path, num_frames, frame_interval, image_size, transform_name=None
        )
        self.transform_name = transform_name
        self.bbxs: List[List[Bbox]] = [
            frame.bbox for frame in self.ann.clip_annotation.frames
        ]
        bbx_ratios = []
        for bbx_arr in self.bbxs:
            ratios_arr = []
            for bbx in bbx_arr:
                ratios_arr.append(bbx.bbox_ratio)
            bbx_ratios.append(ratios_arr)
        self.bbx_ratios: List[List[np.ndarray]] = bbx_ratios

    def __getitem__(self, index: MicroBatch) -> NBAClipsSample:

        sample_index = index.index
        num_frames = index.num_frames
        height, width = index.height, index.width

        path = self.ann.video_fp
        text: str = self.ann.clip_annotation.video_info.caption
        file_type: str = self.ann.clip_annotation.video_info.file_type
        ar: float = height / width
        video_fps = self.ann.clip_annotation.video_info.video_fps

        if file_type == TYPE_VIDEO:
            # loading
            vframes, _, infos = torchvision.io.read_video(
                filename=path, pts_unit="sec", output_format="TCHW"
            )
            if "video_fps" in infos:
                video_fps = infos["video_fps"]
            # sampling video frames
            video, _ = temporal_random_crop(vframes, num_frames, self.frame_interval)
            # conditions
            assert type(sample_index) == int, f"{sample_index}"
            conditions = {
                "bbox_ratios": self.bbx_ratios[sample_index],
            }
            # transform
            transform = get_transforms_video(self.transform_name, (height, width))
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS
            # transform
            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)

            # repeat
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        video: torch.Tensor = video.permute(1, 0, 2, 3)
        return {
            "video": video,
            "text": text,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
            "conditions": conditions,
        }


@DATASETS.register_module()
class VideoTextDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        data_path: str,
        num_frames: int = 16,
        frame_interval: int = 1,
        image_size: Tuple[int, int] = (256, 256),
        transform_name="center",
    ):
        self.data_path: str = data_path
        self.data = read_file(data_path)
        self.num_frames: int = num_frames
        self.frame_interval: int = frame_interval
        self.image_size: Tuple[int, int] = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }

    def _print_data_number(self):
        num_videos = 0
        num_images = 0
        for path in self.data["path"]:
            if self.get_type(path) == "video":
                num_videos += 1
            else:
                num_images += 1
        print(f"Dataset contains {num_videos} videos and {num_images} images.")

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]
        text = sample["text"]
        file_type = self.get_type(path)

        if file_type == "video":
            # loading
            vframes, _, _ = torchvision.io.read_video(
                filename=path, pts_unit="sec", output_format="TCHW"
            )

            # Sampling video frames
            video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)

            # transform
            transform = self.transforms["video"]
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)

            # transform
            transform = self.transforms["image"]
            image = transform(image)

            # repeat
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        return {"video": video, "text": text}

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                path = self.data.iloc[index]["path"]
                print(f"data {path}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)


@DATASETS.register_module()
class VariableVideoTextDataset(VideoTextDataset):
    def __init__(
        self,
        data_path: str,
        num_frames: Optional[int] = None,
        frame_interval=1,
        image_size=None,
        transform_name=None,
    ):
        super().__init__(
            data_path, num_frames, frame_interval, image_size, transform_name=None
        )
        self.transform_name = transform_name
        self.data["id"] = np.arange(len(self.data))

    def get_data_info(self, index):
        T = self.data.iloc[index]["num_frames"]
        H = self.data.iloc[index]["height"]
        W = self.data.iloc[index]["width"]
        return T, H, W

    def getitem(self, index):
        # a hack to pass in the (time, height, width) info from sampler
        index, num_frames, height, width = [int(val) for val in index.split("-")]

        sample = self.data.iloc[index]
        path = sample["path"]
        text = sample["text"]
        file_type = self.get_type(path)
        ar = height / width

        video_fps = 24  # default fps
        if file_type == "video":
            # loading
            path = f"/mnt/mir/fan23j/data/nba-plus-statvu-dataset/filtered-clips/{path}"
            vframes, _, infos = torchvision.io.read_video(
                filename=path, pts_unit="sec", output_format="TCHW"
            )
            if "video_fps" in infos:
                video_fps = infos["video_fps"]

            # sampling video frames
            video, frame_indices = temporal_random_crop(
                vframes, num_frames, self.frame_interval
            )
            # conditions
            conditions = extract_conditions(sample, frame_indices)

            # transform
            transform = get_transforms_video(self.transform_name, (height, width))
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)

            # repeat
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        video: torch.Tensor = video.permute(1, 0, 2, 3)
        return {
            "video": video,
            "text": text,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
            "conditions": conditions,
        }
