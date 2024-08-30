from json import load
import os
import numpy as np
import torch
import torchvision

from nba.src.entities.clip_dataset import FilteredClipDataset
from nba.src.entities.clip_annotations import (
    BoundingBox,
    ClipAnnotation,
    ClipAnnotationWrapper,
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
DATASET_DIR = "/playpen-storage/levlevi/nba-plus-statvu-dataset/filtered-clip-annotations-40-bbx-ratios"
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
        num_frames: int = 16,
        frame_interval: int = 1,
        image_size: Tuple[int, int] = (256, 256),
        transform_name: Optional[str] = "center",
    ):

        # breakpoint()
        # dataset wrapper
        self.filtered_dataset = FilteredClipDataset(DATASET_DIR)

        # HACK: using one sample
        self.annotation_file_paths: List[str] = (
            self.filtered_dataset.filtered_clip_annotations_file_paths
        )
        self.annotation_file_paths: List[str] = self.annotation_file_paths[:100]

        # wrapper for the current sample
        self.ann_wrapper: ClipAnnotationWrapper = ClipAnnotationWrapper(
            self.annotation_file_paths[0], load_clip_annotation=True
        )

        # clip annotation object
        self.clip_annotation: ClipAnnotation = self.ann_wrapper.clip_annotation

        # number of frames in each generated video
        self.num_frames: int = num_frames

        # sample rate / step size
        self.frame_interval: int = frame_interval

        # resolution of output videos
        self.image_size: Tuple[int, int] = image_size

        # MARK: we only support videos
        self.transform: str = transform_name

        # breakpoint()
        # TODO: this is pretty janky
        # (T, 10)
        self.bbxs: List[List[BoundingBox]] = [
            frame.bbox for frame in self.clip_annotation.frames
        ]
        bbx_ratios = []
        for bbx_arr in self.bbxs:
            ratios_arr = []
            for bbx in bbx_arr:
                ratios_arr.append(bbx.bbox_ratio)
            bbx_ratios.append(ratios_arr)
        # (T, 10, 4)
        self.bbx_ratios: List[List[np.ndarray]] = bbx_ratios

    def load_annotation(self, index: int) -> None:

        assert (
            index < len(self.annotation_file_paths) and index >= 0
        ), f"Error: index {index} out of range"
        fp = self.annotation_file_paths[index]
        self.ann_wrapper = ClipAnnotationWrapper(fp, load_clip_annotation=True)

        # try again if we encounter a bad video
        if self.ann_wrapper.clip_annotation.video_path is None:
            self.load_annotation(index + 1)

        # update the current clip annotation obj
        self.clip_annotation = self.ann_wrapper.clip_annotation

    def __getitem__(self, index: MicroBatch) -> Dict:
        """
        Returns a single sample (i.e., video clip and related data)
        """

        # load the next valid annotation object
        sample_index: int = index.index
        self.load_annotation(sample_index)

        num_frames: int = index.num_frames
        height, width = index.height, index.width
        video_path = self.ann_wrapper.video_fp
        text: str = self.clip_annotation.video_info.caption
        ar: float = height / width
        video_fps: int = self.clip_annotation.video_info.video_fps

        # TODO: add image support
        vframes, _, infos = torchvision.io.read_video(
            filename=video_path, pts_unit="sec", output_format="TCHW"
        )
        if "video_fps" in infos:
            video_fps = infos["video_fps"]

        # take a random temporal crop from video of `num_frames` spaced by `frame_interval`
        video, frame_indices = temporal_random_crop(
            vframes, num_frames, self.frame_interval
        )

        # select the corresponding bbx rations
        frame_indices_set = set(frame_indices)
        selected_bbxs = [
            bbx for idx, bbx in enumerate(self.bbx_ratios) if idx in frame_indices_set
        ]

        assert type(sample_index) == int, f"{sample_index}"
        # convert conditions to tensor obj
        conditions = {
            "bbox_ratios": torch.tensor(selected_bbxs),
        }

        # transform
        transform = get_transforms_video(self.transform, (height, width))
        video = transform(video)  # T C H W
        # breakpoint()

        # TCHW -> CTHW
        video: torch.Tensor = video.permute(1, 0, 2, 3)
        return {
            "clip_annotation_idx": torch.tensor(sample_index),
            "video": video,
            "num_frames": torch.tensor(num_frames),
            "height": torch.tensor(height),
            "width": torch.tensor(width),
            "ar": torch.tensor(ar),
            "fps": torch.tensor(video_fps),
            "conditions": conditions,
            # "text": text,
        }

    def __len__(self) -> int:
        return len(self.annotation_file_paths)
