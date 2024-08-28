import torch.nn as nn
from .adapters import MultiTrajEncoder
from opensora.models.layers.blocks import MultiHeadCrossAttention


class JAMES(nn.Module):
    """
    JAMES: Joint Adaptive Motion and Entity Sequencer
    """

    def __init__(self, ca_hidden_size=1152, ca_num_heads=16):
        super().__init__()

        self.cross_attn = MultiHeadCrossAttention(ca_hidden_size, ca_num_heads)

        # adapter modules
        self.multi_traj_adapter = MultiTrajEncoder()
        self.text_encoder = None
        self.pose_traj_adapter = None
        self.motion_track_adapter = None

    def forward(self, x, conditions, **kwargs):
        bbox_features = self.multi_traj_adapter(conditions["bbox_ratios"])
        x = x + self.cross_attn(x, bbox_features)
        return x
