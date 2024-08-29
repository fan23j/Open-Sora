import torch.nn as nn
from .adapters import MultiTrajEncoder
from opensora.models.layers.blocks import MultiHeadCrossAttention
import torch 

class JAMES(nn.Module):
    """
    JAMES: Joint Adaptive Motion and Entity Sequencer
    """
    def __init__(self, ca_hidden_size=1152, ca_num_heads=16):
        super().__init__()

        # adapter modules
        self.multi_traj_adapter = MultiTrajEncoder()
        self.pose_traj_adapter = None
        self.motion_track_adapter = None

    #TODO: do better.
    def forward(self, conditions, **kwargs):
        bbox_features = self.multi_traj_adapter(conditions["bbox_ratios"])
        conditions["bbox_ratios"] = bbox_features
        return conditions


