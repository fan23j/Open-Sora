import numpy as np
import torch
import torch.nn as nn
import os
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.models.layers.blocks import (
    Attention,
    CaptionEmbedder,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    PositionEmbedding2D,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_2d_sincos_pos_embed,
    get_layernorm,
    t2i_modulate,
)
from opensora.registry import MODELS
from transformers import PretrainedConfig, PreTrainedModel
from opensora.utils.ckpt_utils import load_checkpoint
from TrailBlazer.CrossAttn.InjectorProc import InjectorProcessor

class STDiT2Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        rope=None,
        qk_norm=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn
        self._enable_sequence_parallelism = enable_sequence_parallelism
        bbox_per_frame = [[0.5, 0.35, 1.0, 0.65], [0.4920634920634921, 0.35, 0.9920634920634921, 0.65], [0.48412698412698413, 0.35, 0.9841269841269842, 0.65], [0.47619047619047616, 0.35, 0.9761904761904762, 0.65], [0.46825396825396826, 0.35, 0.9682539682539683, 0.65], [0.46031746031746035, 0.35000000000000003, 0.9603174603174603, 0.6500000000000001], [0.4523809523809524, 0.35, 0.9523809523809523, 0.65], [0.4444444444444444, 0.3499999999999999, 0.9444444444444444, 0.6499999999999999], [0.4365079365079365, 0.35, 0.9365079365079365, 0.65], [0.4285714285714286, 0.35, 0.9285714285714286, 0.65], [0.42063492063492064, 0.35, 0.9206349206349207, 0.65], [0.4126984126984127, 0.35, 0.9126984126984127, 0.65], [0.40476190476190477, 0.35, 0.9047619047619048, 0.65], [0.39682539682539686, 0.35, 0.8968253968253969, 0.65], [0.3888888888888889, 0.35, 0.8888888888888888, 0.6499999999999999], [0.38095238095238093, 0.3499999999999999, 0.8809523809523809, 0.65], [0.373015873015873, 0.35, 0.873015873015873, 0.65], [0.3650793650793651, 0.35000000000000003, 0.8650793650793651, 0.65], [0.35714285714285715, 0.35, 0.8571428571428572, 0.65], [0.3492063492063492, 0.35, 0.8492063492063492, 0.65], [0.3412698412698413, 0.35, 0.8412698412698413, 0.65], [0.33333333333333337, 0.35, 0.8333333333333334, 0.6500000000000001], [0.3253968253968254, 0.35, 0.8253968253968254, 0.65], [0.31746031746031744, 0.35, 0.8174603174603174, 0.6499999999999999], [0.30952380952380953, 0.35, 0.8095238095238095, 0.65], [0.3015873015873016, 0.35, 0.8015873015873016, 0.65], [0.29365079365079366, 0.35, 0.7936507936507937, 0.65], [0.2857142857142857, 0.35, 0.7857142857142857, 0.65], [0.2777777777777778, 0.35, 0.7777777777777778, 0.65], [0.2698412698412699, 0.35, 0.7698412698412699, 0.6500000000000001], [0.2619047619047619, 0.35, 0.7619047619047619, 0.65], [0.25396825396825395, 0.35, 0.753968253968254, 0.6499999999999999], [0.24603174603174605, 0.35, 0.746031746031746, 0.65], [0.23809523809523808, 0.35, 0.7380952380952381, 0.65], [0.23015873015873017, 0.35, 0.7301587301587302, 0.65], [0.2222222222222222, 0.35, 0.7222222222222222, 0.65], [0.2142857142857143, 0.35, 0.7142857142857143, 0.65], [0.20634920634920634, 0.35, 0.7063492063492063, 0.65], [0.19841269841269843, 0.35, 0.6984126984126984, 0.65], [0.19047619047619047, 0.35, 0.6904761904761905, 0.65], [0.18253968253968256, 0.35, 0.6825396825396826, 0.65], [0.1746031746031746, 0.35, 0.6746031746031746, 0.65], [0.16666666666666669, 0.35, 0.6666666666666667, 0.65], [0.15873015873015872, 0.35, 0.6587301587301587, 0.65], [0.1507936507936508, 0.35, 0.6507936507936508, 0.65], [0.14285714285714285, 0.35, 0.6428571428571428, 0.65], [0.13492063492063494, 0.35, 0.6349206349206349, 0.65], [0.12698412698412698, 0.35, 0.626984126984127, 0.65], [0.11904761904761907, 0.35, 0.6190476190476191, 0.65], [0.1111111111111111, 0.35, 0.6111111111111112, 0.6499999999999999], [0.1031746031746032, 0.35, 0.6031746031746033, 0.6499999999999999], [0.09523809523809523, 0.35, 0.5952380952380952, 0.65], [0.08730158730158732, 0.35, 0.5873015873015873, 0.65], [0.07936507936507936, 0.35, 0.5793650793650793, 0.65], [0.07142857142857145, 0.35, 0.5714285714285714, 0.65], [0.06349206349206349, 0.35, 0.5634920634920635, 0.65], [0.05555555555555558, 0.35, 0.5555555555555556, 0.65], [0.047619047619047616, 0.35, 0.5476190476190477, 0.65], [0.03968253968253971, 0.35, 0.5396825396825398, 0.65], [0.031746031746031744, 0.35, 0.5317460317460317, 0.65], [0.023809523809523836, 0.35, 0.5238095238095238, 0.65], [0.015873015873015872, 0.35, 0.5158730158730158, 0.65], [0.007936507936507964, 0.35, 0.5079365079365079, 0.65], [0.0, 0.35, 0.5, 0.65]]
        # bbox injector
        self.injector = InjectorProcessor(
                    bbox_per_frame=bbox_per_frame,
                    strengthen_scale=0.125,
                    weaken_scale=0.001,
                    is_spatial=True,
                )
        # spatial branch
        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flash_attn=enable_flash_attn,
            qk_norm=qk_norm,
        )
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        # cross attn
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)

        # mlp branch
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # temporal branch
        self.norm_temp = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)  # new
        self.attn_temp = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flash_attn=self.enable_flash_attn,
            rope=rope,
            qk_norm=qk_norm,
        )
        self.scale_shift_table_temporal = nn.Parameter(torch.randn(3, hidden_size) / hidden_size**0.5)  # new

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(self, x, y, t, t_tmp, mask=None, x_mask=None, t0=None, t0_tmp=None, T=None, S=None):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        shift_tmp, scale_tmp, gate_tmp = (self.scale_shift_table_temporal[None] + t_tmp.reshape(B, 3, -1)).chunk(
            3, dim=1
        )
        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                self.scale_shift_table[None] + t0.reshape(B, 6, -1)
            ).chunk(6, dim=1)
            shift_tmp_zero, scale_tmp_zero, gate_tmp_zero = (
                self.scale_shift_table_temporal[None] + t0_tmp.reshape(B, 3, -1)
            ).chunk(3, dim=1)

        # modulate
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa) #[2, 58880, 1152]
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # spatial branch
        x_s = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
        x_s = self.attn(x_s)
        x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=T, S=S)
        # x_s = rearrange(x_m, "B (T H W) C -> (B T) H W C", B=2, T=T, H=23, W=40)
        # x_s= self.injector(x_s, 40, 23)
        # x_s = rearrange(x_s, "(B T) H W C -> (B T) (H W) C", B=2, T=T, H=23, W=40)
        # x_s = self.attn(x_s)
        # x_s = rearrange(x_s, "(B T) (H W) C -> B (T H W) C", B=2, T=T, H=23, W=40)
        if x_mask is not None:
            x_s_zero = gate_msa_zero * x_s
            x_s = gate_msa * x_s
            x_s = self.t_mask_select(x_mask, x_s, x_s_zero, T, S)
        else:
            x_s = gate_msa * x_s
        x = x + self.drop_path(x_s)

        # modulate
        x_m = t2i_modulate(self.norm_temp(x), shift_tmp, scale_tmp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm_temp(x), shift_tmp_zero, scale_tmp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # temporal branch
        # x_t = rearrange(x_m, "B (T H W) C -> (B T) H W C", B=2, T=T, H=23, W=40)
        # x_t= self.injector(x_t, 40, 23)
        # x_t = rearrange(x_t, "(B T) H W C -> (B T) (H W) C", B=2, T=T, H=23, W=40)
        # x_t = self.attn(x_t)
        # x_t = rearrange(x_t, "(B T) (H W) C -> B (T H W) C", B=2, T=T, H=23, W=40)
        x_t = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
        x_t = self.attn_temp(x_t)
        x_t = rearrange(x_t, "(B S) T C -> B (T S) C", T=T, S=S)
        # x_t = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S) #[1840, 64, 1152]
        # x_t = self.attn_temp(x_t)
        # x_t = rearrange(x_t, "(B H W) T C -> (B T) H W C", B=2, T=T, H=23, W=40)
        # x_t= self.injector(x_t, 40, 23)
        # x_t = rearrange(x_t, "(B T) H W C -> B (T H W) C", B=2, T=T, H=23, W=40)
        if x_mask is not None:
            x_t_zero = gate_tmp_zero * x_t
            x_t = gate_tmp * x_t
            x_t = self.t_mask_select(x_mask, x_t, x_t_zero, T, S)
        else:
            x_t = gate_tmp * x_t
        x = x + self.drop_path(x_t)

        # cross attn
        x = x + self.cross_attn(x, y, mask)

        # modulate
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # mlp
        x_mlp = self.mlp(x_m)
        if x_mask is not None:
            x_mlp_zero = gate_mlp_zero * x_mlp
            x_mlp = gate_mlp * x_mlp
            x_mlp = self.t_mask_select(x_mask, x_mlp, x_mlp_zero, T, S)
        else:
            x_mlp = gate_mlp * x_mlp
        x = x + self.drop_path(x_mlp)

        return x


class STDiT2Config(PretrainedConfig):
    
    model_type = "STDiT2"

    def __init__(
        self,
        input_size=(None, None, None),
        input_sq_size=32,
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        no_temporal_pos_emb=False,
        caption_channels=4096,
        model_max_length=120,
        freeze=None,
        qk_norm=False,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        **kwargs,
    ):
        self.input_size = input_size
        self.input_sq_size = input_sq_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.pred_sigma = pred_sigma
        self.drop_path = drop_path
        self.no_temporal_pos_emb = no_temporal_pos_emb
        self.caption_channels = caption_channels
        self.model_max_length = model_max_length
        self.freeze = freeze
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        super().__init__(**kwargs)


@MODELS.register_module()
class STDiT2(PreTrainedModel):

    config_class = STDiT2Config

    def __init__(
        self,
        config
    ):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.no_temporal_pos_emb = config.no_temporal_pos_emb
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_layernorm_kernel = config.enable_layernorm_kernel

        # support dynamic input
        self.patch_size = config.patch_size
        self.input_size = config.input_size
        self.input_sq_size = config.input_sq_size
        self.pos_embed = PositionEmbedding2D(config.hidden_size)

        self.x_embedder = PatchEmbed3D(config.patch_size, config.in_channels, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True))
        self.t_block_temp = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True))  # new
        self.y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )

        drop_path = [x.item() for x in torch.linspace(0, config.drop_path, config.depth)]
        self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads)  # new
        self.blocks = nn.ModuleList(
            [
                STDiT2Block(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flash_attn=self.enable_flash_attn,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    rope=self.rope.rotate_queries_or_keys,
                    qk_norm=config.qk_norm,
                )
                for i in range(self.depth)
            ]
        )
        self.final_layer = T2IFinalLayer(config.hidden_size, np.prod(self.patch_size), self.out_channels)

        # multi_res
        assert self.hidden_size % 3 == 0, "hidden_size must be divisible by 3"
        self.csize_embedder = SizeEmbedder(self.hidden_size // 3)
        self.ar_embedder = SizeEmbedder(self.hidden_size // 3)
        self.fl_embedder = SizeEmbedder(self.hidden_size)  # new
        self.fps_embedder = SizeEmbedder(self.hidden_size)  # new

        # init model
        self.initialize_weights()
        self.initialize_temporal()
        if config.freeze is not None:
            assert config.freeze in ["not_temporal", "text"]
            if config.freeze == "not_temporal":
                self.freeze_not_temporal()
            elif config.freeze == "text":
                self.freeze_text()

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def forward(
        self, x, timestep, y, mask=None, x_mask=None, num_frames=None, height=None, width=None, ar=None, fps=None
    ):
        """
        Forward pass of STDiT.
        Args:
            x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
            timestep (torch.Tensor): diffusion time steps; of shape [B]
            y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
            mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]

        Returns:
            x (torch.Tensor): output latent representation; of shape [B, C, T, H, W]
        """
        B = x.shape[0]
        dtype = self.x_embedder.proj.weight.dtype
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        # === process data info ===
        # 1. get dynamic size
        hw = torch.cat([height[:, None], width[:, None]], dim=1)
        rs = (height[0].item() * width[0].item()) ** 0.5
        csize = self.csize_embedder(hw, B)

        # 2. get aspect ratio
        ar = ar.unsqueeze(1)
        ar = self.ar_embedder(ar, B)
        data_info = torch.cat([csize, ar], dim=1)

        # 3. get number of frames
        fl = num_frames.unsqueeze(1)
        fps = fps.unsqueeze(1)
        fl = self.fl_embedder(fl, B)
        fl = fl + self.fps_embedder(fps, B)

        # === get dynamic shape size ===
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)
        S = H * W
        scale = rs / self.input_sq_size
        base_size = round(S**0.5)
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # embedding
        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb
        x = rearrange(x, "B T S C -> B (T S) C")

        # prepare adaIN
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        t_spc = t + data_info  # [B, C]
        t_tmp = t + fl  # [B, C]
        t_spc_mlp = self.t_block(t_spc)  # [B, 6*C]
        t_tmp_mlp = self.t_block_temp(t_tmp)  # [B, 3*C]
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0_spc = t0 + data_info
            t0_tmp = t0 + fl
            t0_spc_mlp = self.t_block(t0_spc)
            t0_tmp_mlp = self.t_block_temp(t0_tmp)
        else:
            t0_spc = None
            t0_tmp = None
            t0_spc_mlp = None
            t0_tmp_mlp = None

        # prepare y
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # blocks
        for _, block in enumerate(self.blocks):
            x = auto_grad_checkpoint(
                block,
                x,
                y,
                t_spc_mlp,
                t_tmp_mlp,
                y_lens,
                x_mask,
                t0_spc_mlp,
                t0_tmp_mlp,
                T,
                S,
            )
            # x.shape: [B, N, C]

        # final process
        x = self.final_layer(x, t, x_mask, t0_spc, T, S)  # [B, N, C=T_p * H_p * W_p * C_out]
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)  # [B, C_out, T, H, W]

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x

    def unpatchify_old(self, x):
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = rearrange(x, "n t h w r p q c -> n c t r h p w q")
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def get_spatial_pos_embed(self, H, W, scale=1.0, base_size=None):
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (H, W),
            scale=scale,
            base_size=base_size,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def freeze_not_temporal(self):
        for n, p in self.named_parameters():
            if "attn_temp" not in n:
                p.requires_grad = False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "cross_attn" in n:
                p.requires_grad = False

    def initialize_temporal(self):
        for block in self.blocks:
            nn.init.constant_(block.attn_temp.proj.weight, 0)
            nn.init.constant_(block.attn_temp.proj.bias, 0)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        nn.init.normal_(self.t_block_temp[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


@MODELS.register_module("STDiT2-XL/2")
def STDiT2_XL_2(from_pretrained=None, **kwargs):
    if from_pretrained is not None:
        if os.path.isdir(from_pretrained) or os.path.isfile(from_pretrained):
            # if it is a directory or a file, we load the checkpoint manually
            config = STDiT2Config(
                depth=28,
                hidden_size=1152,
                patch_size=(1, 2, 2),
                num_heads=16, **kwargs
            )
            model = STDiT2(config)
            load_checkpoint(model, from_pretrained)
            return model
        else:
            # otherwise, we load the model from hugging face hub
            return STDiT2.from_pretrained(from_pretrained)
    else:
        # create a new model
        config = STDiT2Config(
            depth=28,
            hidden_size=1152,
            patch_size=(1, 2, 2),
            num_heads=16, **kwargs
        )
        model = STDiT2(config)
    return model
