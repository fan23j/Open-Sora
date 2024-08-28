import torch.nn as nn
from .adapters import MultiTrajEncoder
from opensora.models.layers.blocks import MultiHeadCrossAttention
import torch 

class JAMES(nn.Module):
    """
    JAMES: Joint Adaptive Motion and Entity Sequencer
    """
    def __init__(self, ca_hidden_size=1152, ca_num_heads=16, y_embedder=None):
        super().__init__()

        self.y_embedder = y_embedder

        self.bbox_cross_attn = MultiHeadCrossAttention(ca_hidden_size, ca_num_heads)

        self._2d_pos_cross_attn = MultiHeadCrossAttention(ca_hidden_size, ca_num_heads)
        
        self.text_cross_attn = MultiHeadCrossAttention(ca_hidden_size, ca_num_heads)

        # adapter modules
        self.multi_traj_adapter = MultiTrajEncoder()
        self.text_embeddings = torch.load('/mnt/mir/fan23j/data/nba-plus-statvu-dataset/__scripts__/text_embeddings_bfloat16.pth')
        self.pose_traj_adapter = None
        self.motion_track_adapter = None

    #TODO: do better.
    def forward(self, x, conditions, inject_bbox=False, inject_2d_pos=False, inject_text=False, **kwargs):
        if inject_bbox:
            bbox_features = self.multi_traj_adapter(conditions["bbox_ratios"])
            x = x + self.bbox_cross_attn(x, bbox_features)
        if inject_2d_pos:
            pos_features = self.multi_traj_adapter(conditions["2d_pos"])
            x = x + self._2d_pos_cross_attn(x, pos_features)
        if inject_text:
            # do better
            y = self.y_embedder(self.text_embeddings["y"], x.shape[0] != 1)
            mask = self.text_embeddings["mask"]
            if mask is not None:
                if mask.shape[0] != y.shape[0]:
                    mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
                mask = mask.squeeze(1).squeeze(1)
                y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
                y_lens = mask.sum(dim=1).tolist()
            else:
                y_lens = [y.shape[2]] * y.shape[0]
                y = y.squeeze(1).view(1, -1, x.shape[-1])
            x = x + self.text_cross_attn(x, y)
        return x


