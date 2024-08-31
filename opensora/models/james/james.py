import torch.nn as nn
from .adapters import MultiTrajEncoder
from opensora.models.layers.blocks import MultiHeadCrossAttention
import torch 

class JAMES(nn.Module):
    """
    JAMES: Joint Adaptive Motion and Entity Sequencer
    """
    def __init__(self):
        super().__init__()

        # adapter modules
        self.multi_traj_adapter = MultiTrajEncoder()
        self.pose_traj_adapter = None
        self.motion_track_adapter = None

    def forward(self, conditions, **kwargs):
        bbox_features = self.multi_traj_adapter(conditions["bbox_ratios"])
        conditions["bbox_features"] = bbox_features
        return conditions


