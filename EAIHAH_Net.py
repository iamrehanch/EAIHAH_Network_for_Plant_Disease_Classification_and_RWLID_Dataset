
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from timm.models.layers import trunc_normal_, DropPath as TimmDropPath
from timm.models.registry import register_model
from pytorch_wavelets import DWTForward


# =============================================================================
# Basic ConvNeXt block
# =============================================================================
class Block(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = TimmDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


# =============================================================================

class FFGBlock(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.local_path = nn.Sequential(
            nn.Conv2d(
                in_ch, in_ch, kernel_size=3, padding=2, dilation=2,
                groups=in_ch, bias=False
            ),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.local_gate = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        sobel_x = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feat = self.local_path(x)

        b, c, h, w = x.shape
        x_reshaped = x.reshape(b * c, 1, h, w)
        grad_x = F.conv2d(x_reshaped, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_reshaped, self.sobel_y, padding=1)
        edge_feat = (grad_x + grad_y).reshape(b, c, h, w)

        dense_local_gate = torch.sigmoid(self.local_gate(local_feat + edge_feat))
        channel_gate = self.channel_gate(x)

        local_refined = x * dense_local_gate
        channel_refined = x * channel_gate
        return local_refined + channel_refined


# =============================================================================

class FixedPCAReduction(nn.Module):
    def __init__(
        self,
        in_ch: int,
        k: int = 57,
        momentum: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.k = min(k, in_ch)
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(in_ch))
        self.register_buffer("running_cov", torch.eye(in_ch))
        self.register_buffer("basis", torch.eye(in_ch))

    @torch.no_grad()
    def update_basis(self, x: torch.Tensor) -> None:
        _, c, _, _ = x.shape
        feats = x.detach().permute(0, 2, 3, 1).reshape(-1, c).float()

        batch_mean = feats.mean(dim=0)
        centered = feats - batch_mean.unsqueeze(0)
        batch_cov = (centered.T @ centered) / max(feats.shape[0] - 1, 1)

        self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * batch_mean)
        self.running_cov.mul_(1.0 - self.momentum).add_(self.momentum * batch_cov)

        cov = 0.5 * (self.running_cov + self.running_cov.T)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        order = torch.argsort(eigvals, descending=True)
        eigvecs = eigvecs[:, order]
        self.basis.copy_(eigvecs.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.update_basis(x)

        x_centered = x - self.running_mean.view(1, self.in_ch, 1, 1)
        basis_k = self.basis[:, :self.k].contiguous()
        proj_weight = basis_k.T.contiguous().view(self.k, self.in_ch, 1, 1)
        return F.conv2d(x_centered, proj_weight)


def kl_divergence_left_neighbor(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

    p = F.softmax(x[:, :, :, 1:], dim=1)
    q = F.softmax(x[:, :, :, :-1], dim=1)
    log_p = torch.log(p.clamp(min=eps))
    log_q = torch.log(q.clamp(min=eps))
    return (p * (log_p - log_q)).sum(dim=1)


class ModifiedInvolution(nn.Module):
    def __init__(
        self,
        channels: int = 96,
        kernel_size: int = 3,
        groups: int = 1,
        stride: int = 1,
        pca_k: int = 57,
        kl_threshold: float = 0.5,
    ):
        super().__init__()
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")

        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = stride
        self.kl_threshold = kl_threshold

        self.pca_reduction = FixedPCAReduction(in_ch=channels, k=pca_k)
        self.kernel_gen = nn.Conv2d(
            self.pca_reduction.k,
            kernel_size * kernel_size * groups,
            kernel_size=1,
            bias=False,
        )
        self.unfold = nn.Unfold(
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ksize = self.kernel_size
        groups = self.groups
        c_per_group = c // groups

        x_k = self.pca_reduction(x)
        raw_kernel = self.kernel_gen(x_k)  # (B, G*K*K, H, W)

        if w > 1:
            divergence = kl_divergence_left_neighbor(x_k)
            reuse_mask = (divergence < self.kl_threshold).unsqueeze(1)  # (B,1,H,W-1)

            kernel = raw_kernel.clone()
            kernel[:, :, :, 1:] = torch.where(
                reuse_mask,
                raw_kernel[:, :, :, :-1],
                raw_kernel[:, :, :, 1:],
            )
        else:
            kernel = raw_kernel

        h_out, w_out = kernel.shape[-2], kernel.shape[-1]
        x_unfolded = self.unfold(x)
        x_unfolded = x_unfolded.view(b, groups, c_per_group, ksize * ksize, h_out, w_out)

        kernel = kernel.view(b, groups, ksize * ksize, h_out, w_out)
        kernel = F.softmax(kernel, dim=2)

        out = (x_unfolded * kernel.unsqueeze(2)).sum(dim=3)
        out = out.view(b, c, h_out, w_out)
        return out


class EAIBlock(nn.Module):
    def __init__(
        self,
        channels: int = 96,
        pca_k: int = 57,
        kl_threshold: float = 0.5,
        kernel_size: int = 3,
        groups: int = 1,
    ):
        super().__init__()
        self.involution = ModifiedInvolution(
            channels=channels,
            kernel_size=kernel_size,
            groups=groups,
            pca_k=pca_k,
            kl_threshold=kl_threshold,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.involution(x)


# =============================================================================
# Grad-CAM++ utilities for HAT block guidance
# =============================================================================
class GradCAMPlusPlusEngine:
    def __init__(self, target_layer: nn.Module, eps: float = 1e-7):
        self._activation: Optional[torch.Tensor] = None
        self._gradient: Optional[torch.Tensor] = None
        self._hooks: List = []
        self.eps = eps

        self._hooks.append(target_layer.register_forward_hook(self._save_activation))
        self._hooks.append(target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, inputs, output):
        self._activation = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradient = grad_output[0].detach()

    def compute(
        self,
        score_c: torch.Tensor,
        cam_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        score_c.sum().backward(retain_graph=False)

        if self._activation is None or self._gradient is None:
            raise RuntimeError("Grad-CAM++ hooks did not capture activation/gradient.")

        cam_norm, cam_mask = self._post_process(self._activation, self._gradient, cam_threshold)
        self._activation = None
        self._gradient = None
        return cam_norm, cam_mask

    def _post_process(
        self,
        activation: torch.Tensor,
        gradient: torch.Tensor,
        cam_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grad2 = gradient ** 2
        grad3 = gradient ** 3
        sum_grad3 = grad3.sum(dim=[2, 3], keepdim=True)
        denom = 2.0 * grad2 + activation * sum_grad3
        alpha_pixel = grad2 / (denom + self.eps)

        alpha_k = (torch.relu(gradient) * alpha_pixel).sum(dim=[2, 3])
        cam = (alpha_k.unsqueeze(-1).unsqueeze(-1) * activation).sum(dim=1)
        cam = torch.relu(cam)

        b, h, w = cam.shape
        cam_flat = cam.view(b, -1)
        cam_min = cam_flat.min(dim=1)[0].view(b, 1, 1)
        cam_max = cam_flat.max(dim=1)[0].view(b, 1, 1)
        cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam_mask = cam_norm >= cam_threshold
        return cam_norm, cam_mask

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def extract_padded_bboxes(
    mask: torch.Tensor,
    pad: int = 1,
    H: int = 14,
    W: int = 14,
) -> List[Tuple[int, int, int, int]]:
    bboxes = []
    for b in range(mask.shape[0]):
        m = mask[b]
        row_fg = m.any(dim=1)
        col_fg = m.any(dim=0)

        if not row_fg.any():
            bboxes.append((0, 0, H, W))
            continue

        rows = row_fg.nonzero(as_tuple=True)[0]
        cols = col_fg.nonzero(as_tuple=True)[0]
        r1 = max(0, int(rows[0]) - pad)
        r2 = min(H, int(rows[-1]) + 1 + pad)
        c1 = max(0, int(cols[0]) - pad)
        c2 = min(W, int(cols[-1]) + 1 + pad)
        bboxes.append((r1, c1, r2, c2))
    return bboxes


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return self.proj_drop(out)


class HAHMSA(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        H: int,
        W: int,
        bbox_pad: int = 1,
        token_threshold: float = 0.5,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.bbox_pad = bbox_pad
        self.token_threshold = token_threshold
        self.msa = MultiHeadSelfAttention(dim, num_heads, attn_drop, proj_drop)

        self.last_bboxes: List[Tuple[int, int, int, int]] = []
        self.last_fg_ratio: float = 0.0
        self.last_selected_ratio: float = 0.0

    def forward(
        self,
        x: torch.Tensor,
        cam_mask: torch.Tensor,
        cam_norm: torch.Tensor,
    ) -> torch.Tensor:
        b, h, w, c = x.shape
        bboxes = extract_padded_bboxes(cam_mask, pad=self.bbox_pad, H=h, W=w)
        self.last_bboxes = bboxes

        self.last_fg_ratio = sum(
            (r2 - r1) * (c2 - c1) / float(h * w) for r1, c1, r2, c2 in bboxes
        ) / max(b, 1)

        out = x.clone()
        crops, n_tokens, sel_indices = [], [], []

        for batch_idx, (r1, c1, r2, c2) in enumerate(bboxes):
            crop = x[batch_idx, r1:r2, c1:c2, :]
            cam_crop = cam_norm[batch_idx, r1:r2, c1:c2]
            h_b, w_b = crop.shape[:2]

            crop_flat = crop.reshape(h_b * w_b, c)
            cam_flat = cam_crop.reshape(-1)

            sel_mask = cam_flat >= self.token_threshold
            if sel_mask.sum() == 0:
                sel_mask = torch.ones_like(sel_mask, dtype=torch.bool)

            sel_idx = sel_mask.nonzero(as_tuple=False).squeeze(1)
            sel_toks = crop_flat[sel_idx]

            crops.append(sel_toks)
            n_tokens.append(sel_toks.shape[0])
            sel_indices.append((batch_idx, r1, c1, r2, c2, sel_idx, h_b, w_b))

        total_bbox_tokens = sum((r2 - r1) * (c2 - c1) for _, r1, c1, r2, c2, _, _, _ in sel_indices)
        self.last_selected_ratio = sum(n_tokens) / max(total_bbox_tokens, 1)

        max_n = max(n_tokens) if n_tokens else 1
        padded = x.new_zeros(b, max_n, c)
        pad_mask = torch.ones(b, max_n, dtype=torch.bool, device=x.device)

        for i, (tokens, n) in enumerate(zip(crops, n_tokens)):
            padded[i, :n] = tokens
            pad_mask[i, :n] = False

        attn_out = self.msa(padded, key_padding_mask=pad_mask)

        for i, (batch_idx, r1, c1, r2, c2, sel_idx, h_b, w_b) in enumerate(sel_indices):
            n = n_tokens[i]
            flat = out[batch_idx, r1:r2, c1:c2, :].contiguous().reshape(h_b * w_b, c)
            flat[sel_idx] = attn_out[i, :n]
            out[batch_idx, r1:r2, c1:c2, :] = flat.reshape(h_b, w_b, c)

        return out


class FFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HATBlock(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 6,
        H: int = 14,
        W: int = 14,
        bbox_pad: int = 1,
        token_threshold: float = 0.5,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.feature_adapt = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.norm1 = nn.LayerNorm(dim)
        self.hah_msa = HAHMSA(
            dim=dim,
            num_heads=num_heads,
            H=H,
            W=W,
            bbox_pad=bbox_pad,
            token_threshold=token_threshold,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim=dim, mlp_ratio=mlp_ratio, drop=proj_drop)
        self.drop_path = TimmDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.H = H
        self.W = W

        self._cam_mask: Optional[torch.Tensor] = None
        self._cam_norm: Optional[torch.Tensor] = None

    def set_cam_mask(
        self,
        mask: Optional[torch.Tensor],
        norm: Optional[torch.Tensor] = None,
    ) -> None:
        self._cam_mask = mask
        self._cam_norm = norm

    def _resize_cam(self, cam: torch.Tensor, h: int, w: int, mode: str) -> torch.Tensor:
        cam = cam.unsqueeze(1)
        cam = F.interpolate(cam.float(), size=(h, w), mode=mode, align_corners=False if mode != "nearest" else None)
        return cam.squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_adapt(x)
        b, c, h, w = x.shape

        if self._cam_mask is None:
            cam_mask = torch.ones(b, h, w, dtype=torch.bool, device=x.device)
            cam_norm = torch.ones(b, h, w, dtype=x.dtype, device=x.device)
        else:
            cam_mask = self._cam_mask
            cam_norm = self._cam_norm if self._cam_norm is not None else cam_mask.float()

            if cam_mask.shape[-2:] != (h, w):
                cam_mask = self._resize_cam(cam_mask, h, w, mode="nearest") > 0.5
            if cam_norm.shape[-2:] != (h, w):
                cam_norm = self._resize_cam(cam_norm, h, w, mode="bilinear")

        x_hwc = x.permute(0, 2, 3, 1).contiguous()

        residual = x_hwc
        x_hwc = self.norm1(x_hwc)
        x_hwc = residual + self.drop_path(self.hah_msa(x_hwc, cam_mask, cam_norm))

        residual = x_hwc
        x_hwc = self.norm2(x_hwc)
        x_hwc = residual + self.drop_path(self.ffn(x_hwc))

        return x_hwc.permute(0, 3, 1, 2).contiguous()

    @property
    def last_bboxes(self):
        return self.hah_msa.last_bboxes

    @property
    def last_fg_ratio(self):
        return self.hah_msa.last_fg_ratio

    @property
    def last_selected_ratio(self):
        return self.hah_msa.last_selected_ratio


class GradCAMController:
    def __init__(
        self,
        model: nn.Module,
        box_threshold: float = 0.5,
        token_threshold: float = 0.5,
    ):
        self.model = model
        self.box_threshold = box_threshold
        self.token_threshold = token_threshold

        self.hat_block = self._find_hat_block()
        self.target_layer = self._find_pre_hat_target_layer()
        self.gradcam = GradCAMPlusPlusEngine(target_layer=self.target_layer)

        self.last_images = None
        self.last_cam_norm = None
        self.last_cam_mask = None
        self.last_bboxes = None

    def _find_hat_block(self) -> HATBlock:
        for stage in self.model.stages:
            for module in stage.modules():
                if isinstance(module, HATBlock):
                    return module
        raise RuntimeError("No HATBlock found in model.stages.")

    def _find_pre_hat_target_layer(self) -> nn.Module:
        for stage in self.model.stages:
            modules = list(stage.children())
            for idx, module in enumerate(modules):
                if isinstance(module, HATBlock):
                    if idx == 0:
                        raise RuntimeError("HATBlock is the first block in its stage. No pre-HAT target layer found.")
                    return modules[idx - 1]
        raise RuntimeError("Could not find a layer immediately before HATBlock.")

    def two_pass_forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        optimizer.zero_grad()

        with torch.enable_grad():
            self.hat_block.set_cam_mask(None, None)
            logits_pass1 = self.model(images)
            score_c = logits_pass1[torch.arange(len(labels), device=labels.device), labels]
            cam_norm, cam_mask = self.gradcam.compute(score_c, cam_threshold=self.box_threshold)

        optimizer.zero_grad()

        self.hat_block.set_cam_mask(cam_mask, cam_norm)
        logits_pass2 = self.model(images)
        self.hat_block.set_cam_mask(None, None)

        self.last_images = images.detach().cpu()
        self.last_cam_norm = cam_norm.detach().cpu()
        self.last_cam_mask = cam_mask.detach().cpu()
        self.last_bboxes = extract_padded_bboxes(
            cam_mask.detach().cpu(),
            pad=self.hat_block.hah_msa.bbox_pad,
            H=cam_mask.shape[-2],
            W=cam_mask.shape[-1],
        )
        return logits_pass2

    def remove_hooks(self):
        self.gradcam.remove()


# =============================================================================

class FDSFBlock(nn.Module):
    def __init__(self, in_channel: int, features: int, bias: bool = True, wave: str = "haar"):
        super().__init__()
        if features % 2 != 0:
            raise ValueError("features must be even")

        half = features // 2

        self.initial_dwconv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channel,
            bias=bias,
        )

        self.spatial_dwconv = nn.Conv2d(
            in_channels=half,
            out_channels=half,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=half,
            bias=bias,
        )
        self.spatial_pwconv = nn.Conv2d(
            in_channels=half,
            out_channels=half,
            kernel_size=1,
            stride=1,
            bias=bias,
        )
        self.spatial_bn = nn.BatchNorm2d(half)

        self.dwt = DWTForward(J=1, wave=wave, mode="zero")
        self.hp_reduce = nn.Conv2d(
            in_channels=3 * half,
            out_channels=half,
            kernel_size=1,
            stride=1,
            bias=bias,
        )
        self.freq_dilated_conv = nn.Conv2d(
            in_channels=half,
            out_channels=half,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            bias=bias,
        )

        self.act = nn.GELU()
        self.final_fuse = nn.Conv2d(
            in_channels=features,
            out_channels=features,
            kernel_size=1,
            bias=bias,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        _, _, h, w = x.shape

        x_hat = self.initial_dwconv(x)
        x_s, x_f = x_hat.chunk(2, dim=1)

        spatial_feat = self.spatial_dwconv(x_s)
        spatial_feat = self.spatial_pwconv(spatial_feat)
        spatial_feat = self.spatial_bn(spatial_feat)
        spatial_feat = self.act(spatial_feat)
        y_s = x_s + spatial_feat

        _, yh = self.dwt(x_f)
        lh = yh[0][:, :, 0, :, :]
        hl = yh[0][:, :, 1, :, :]
        hh = yh[0][:, :, 2, :, :]

        high_freq = torch.cat([lh, hl, hh], dim=1)
        high_freq = self.hp_reduce(high_freq)
        high_freq = F.interpolate(high_freq, size=(h, w), mode="bilinear", align_corners=False)
        high_freq = self.freq_dilated_conv(high_freq)
        high_freq = self.act(high_freq)
        y_f = x_f + high_freq

        fused = torch.cat([y_s, y_f], dim=1)
        fused = self.final_fuse(fused)
        return residual + fused


# =============================================================================

class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
        eai_pca_k: int = 57,
        eai_kl_threshold: float = 0.5,
        hat_num_heads: int = 6,
        hat_box_threshold: float = 0.5,
        hat_token_threshold: float = 0.5,
    ):
        super().__init__()
        self.pre_stage0_ffg = FFGBlock(in_ch=dims[0])

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            blocks = [
                Block(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                )
                for j in range(depths[i])
            ]
            stage = nn.Sequential(*blocks)

            if i == 0:
                stage.add_module(
                    "eai_block",
                    EAIBlock(
                        channels=dims[i],
                        pca_k=eai_pca_k,
                        kl_threshold=eai_kl_threshold,
                    ),
                )

            if i == 2:
                stage.add_module(
                    "hat_block",
                    HATBlock(
                        dim=dims[i],
                        num_heads=hat_num_heads,
                        H=14,
                        W=14,
                        bbox_pad=1,
                        token_threshold=hat_token_threshold,
                        attn_drop=0.0,
                        proj_drop=0.0,
                        drop_path_rate=0.0,
                    ),
                )

            if i == 3:
                stage.add_module(
                    "fdsf_block",
                    FDSFBlock(
                        in_channel=dims[i],
                        features=dims[i],
                    ),
                )

            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.hat_box_threshold = hat_box_threshold
        self.hat_token_threshold = hat_token_threshold

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm)):
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample_layers[0](x)
        x = self.pre_stage0_ffg(x)
        x = self.stages[0](x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = x.mean(dim=(-2, -1))
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.head(x)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError("data_format must be channels_last or channels_first")

        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


def _load_pretrained(model: nn.Module, url: str):
    checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    return model


@register_model
def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls["convnext_tiny_22k"] if in_22k else model_urls["convnext_tiny_1k"]
        model = _load_pretrained(model, url)
    return model


@register_model
def convnext_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls["convnext_small_22k"] if in_22k else model_urls["convnext_small_1k"]
        model = _load_pretrained(model, url)
    return model


@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls["convnext_base_22k"] if in_22k else model_urls["convnext_base_1k"]
        model = _load_pretrained(model, url)
    return model


@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls["convnext_large_22k"] if in_22k else model_urls["convnext_large_1k"]
        model = _load_pretrained(model, url)
    return model


@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        if not in_22k:
            raise AssertionError("Only ImageNet-22K pre-trained ConvNeXt-XL is available; set in_22k=True.")
        url = model_urls["convnext_xlarge_22k"]
        model = _load_pretrained(model, url)
    return model
