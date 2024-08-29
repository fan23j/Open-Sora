import os
import re

import numpy as np
import pandas as pd
import requests
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from torchvision.io import write_video
from torchvision.utils import save_image

from . import video_transforms

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

regex = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def is_url(url):
    return re.match(regex, url) is not None


def read_file(input_path):
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    elif input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {input_path}")


def download_url(input_path):
    output_dir = "cache"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, base_name)
    img_data = requests.get(input_path).content
    with open(output_path, "wb") as handler:
        handler.write(img_data)
    print(f"URL {input_path} downloaded to {output_path}")
    return output_path


def temporal_random_crop(vframes, num_frames, frame_interval):
    temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
    total_frames = len(vframes)
    start_frame_ind, end_frame_ind = temporal_sample(total_frames)
    assert end_frame_ind - start_frame_ind >= num_frames
    frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, num_frames, dtype=int)
    video = vframes[frame_indice]
    return video, frame_indice


def get_transforms_video(name="center", image_size=(256, 256)):
    if name is None:
        return None
    elif name == "center":
        assert image_size[0] == image_size[1], "image_size must be square for center crop"
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                # video_transforms.RandomHorizontalFlipVideo(),
                video_transforms.UCFCenterCropVideo(image_size[0]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "resize_crop":
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                video_transforms.ResizeCrop(image_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform_video


def get_transforms_image(name="center", image_size=(256, 256)):
    if name is None:
        return None
    elif name == "center":
        assert image_size[0] == image_size[1], "Image size must be square for center crop"
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size[0])),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "resize_crop":
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: resize_crop_to_fill(pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform


def read_image_from_path(path, transform=None, transform_name="center", num_frames=1, image_size=(256, 256)):
    image = pil_loader(path)
    if transform is None:
        transform = get_transforms_image(image_size=image_size, name=transform_name)
    image = transform(image)
    video = image.unsqueeze(0).repeat(num_frames, 1, 1, 1)
    video = video.permute(1, 0, 2, 3)
    return video


def read_video_from_path(path, transform=None, transform_name="center", image_size=(256, 256)):
    vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
    if transform is None:
        transform = get_transforms_video(image_size=image_size, name=transform_name)
    video = transform(vframes)  # T C H W
    video = video.permute(1, 0, 2, 3)
    return video


def read_from_path(path, image_size, transform_name="center"):
    if is_url(path):
        path = download_url(path)
    ext = os.path.splitext(path)[-1].lower()
    if ext.lower() in VID_EXTENSIONS:
        return read_video_from_path(path, image_size=image_size, transform_name=transform_name)
    else:
        assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
        return read_image_from_path(path, image_size=image_size, transform_name=transform_name)


def save_sample(x, fps=8, save_path=None, normalize=True, value_range=(-1, 1), force_video=False):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    assert x.ndim == 4

    if not force_video and x.shape[1] == 1:  # T = 1: save as image
        save_path += ".png"
        x = x.squeeze(1)
        save_image([x], save_path, normalize=normalize, value_range=value_range)
    else:
        save_path += ".mp4"
        if normalize:
            low, high = value_range
            x.clamp_(min=low, max=high)
            x.sub_(low).div_(max(high - low, 1e-5))

        x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)
        write_video(save_path, x, fps=fps, video_codec="h264")
    print(f"Saved to {save_path}")
    return save_path


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def resize_crop_to_fill(pil_image, image_size):
    w, h = pil_image.size  # PIL is (W, H)
    th, tw = image_size
    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        image = pil_image.resize((sw, sh), Image.BICUBIC)
        i = 0
        j = int(round((sw - tw) / 2.0))
    else:
        sh, sw = round(h * rw), tw
        image = pil_image.resize((sw, sh), Image.BICUBIC)
        i = int(round((sh - th) / 2.0))
        j = 0
    arr = np.array(image)
    assert i + th <= arr.shape[0] and j + tw <= arr.shape[1]
    return Image.fromarray(arr[i : i + th, j : j + tw])



def bounding_box_string_to_tensor(bbox_string, frame_indices, num_instances=10):
    """
    Convert a string of bounding box coordinates to a tensor, extracting only the specified frames.
    
    Args:
    bbox_string (str): A string of comma-separated floating-point numbers representing
                       bounding box coordinates in the format "x1,y1,w1,h1,x2,y2,w2,h2,...".
    frame_indices (list): List of frame indices to extract.
    num_instances (int, optional): Number of object instances. Defaults to 10.
    
    Returns:
    torch.Tensor: A tensor of shape [num_instances, len(frame_indices), 4] containing the extracted bounding box coordinates.
    """
    # Clean and parse the string
    # cleaned_string = bbox_string.strip('[]() ').strip()
    # bbox_values = [float(x) for x in cleaned_string.split(",")]
    bbox_values = eval(bbox_string, {"array": np.array})
    
    # Calculate the total number of frames in the original data
    total_frames = len(bbox_values)
    
    # Extract only the specified frames
    extracted_values = []
    for frame in frame_indices:
        extracted_values.extend(bbox_values[frame])
    
    # Reshape into [num_instances, len(frame_indices), 4]
    num_frames = len(frame_indices)
    extracted_values_array = np.array(extracted_values)
    return torch.from_numpy(extracted_values_array).reshape(num_instances, num_frames, 4)

def extract_conditions(sample, frame_indices):
    conditions = {}
    conditions["bbox_ratios"] = bounding_box_string_to_tensor(
        sample["bbox_ratios"], 
        frame_indices
    )

    conditions["text"] = sample["text"]
    
    return conditions

basketball_actions = [
    "A basketball player missing a three-point shot",
    "A basketball player assisting on a play",
    "A basketball player setting a screen",
    "A basketball player grabbing a rebound",
    "A basketball player committing a turnover",
    "A basketball player making a free throw",
    "A basketball player missing a free throw",
    "A basketball player scoring and being fouled",
    "A basketball player missing a two-point shot",
    "A basketball player making a two-point shot",
    "A basketball player committing a foul",
    "A basketball player executing a pick and roll",
    "A basketball player posting up",
    "A basketball player stealing the ball",
    "A basketball player receiving a technical foul",
    "A basketball player making a three-point shot",
    "A basketball player committing their second foul",
    "A basketball player committing their third foul",
    "A basketball player committing an unsportsmanlike foul",
    "A basketball player making a three-pointer and being fouled",
    "A basketball player getting a second chance opportunity",
    "A basketball player making two free throws",
    "A basketball player missing two free throws",
    "A basketball player making three free throws",
    "A basketball player missing three free throws",
    "A basketball player committing a disqualifying foul"
]

action_to_index = {action: i for i, action in enumerate(basketball_actions)}

def get_embeddings_for_prompts(prompts, embeddings, mask):
    device = embeddings.device  # Get the device of the embeddings tensor

    # Convert list of prompts to indices
    indices = np.array([action_to_index[prompt] for prompt in prompts])
    assert len(indices) == len(prompts), f"Not all prompts found in action_to_index. Missing: {set(prompts) - set(action_to_index.keys())}"
    
    # Convert indices to a torch tensor on the correct device
    indices_tensor = torch.from_numpy(indices).to(device)
    
    # Use index_select with the device-specific indices
    filtered_embeddings = torch.index_select(embeddings, 0, indices_tensor)
    filtered_y_null = embeddings[26].unsqueeze(0).expand(len(prompts), -1, -1, -1)
    filtered_mask = torch.index_select(mask, 0, indices_tensor)
    
    # Concatenate along the first dimension
    combined_embeddings = torch.cat([filtered_embeddings, filtered_y_null], dim=0)
    
    return combined_embeddings, filtered_mask