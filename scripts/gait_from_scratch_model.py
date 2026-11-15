"""Random-initialized assessment TCN; no pose extractor or pretrained encoder.

This module does not read datasets, fit references, launch jobs, or load weights.
It consumes the separately audited 30-frame H36M-17 tensor and declared context.
Outputs describe the observed full window, never an event in the future.
"""
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


OUTCOMES = (
    "knee_rom95_mean", "knee_rom95_asymmetry", "knee_p95_mean",
    "knee_temporal_sd_mean", "trunk_frontal_rom95", "trunk_side_rom95",
    "hip_line_rom95_proxy", "shoulder_line_rom95", "arm_rom95_mean",
    "arm_rom95_asymmetry",
)
CLASSES = tuple(f"{name}_high" for name in OUTCOMES) + (
    "knee_rom95_mean_low", "knee_p95_mean_low", "arm_rom95_mean_low",
)


@dataclass(frozen=True)
class TCNConfig:
    frames: int = 30
    joints: int = 17
    channels: tuple = ("normalized_x", "normalized_y", "source_confidence", "valid")
    width: int = 96
    dilations: tuple = (1, 2, 4, 8)
    dropout: float = 0.1
    readout_width: int = 128
    views: tuple = ("front_rear", "side", "oblique", "mixed", "unknown")
    environments: tuple = ("overground", "treadmill", "other")
    numeric_methods: tuple = ("sapiens2_1b", "publisher")
    outcomes: tuple = OUTCOMES
    classes: tuple = CLASSES

    @property
    def context_names(self):
        return tuple("view_" + x for x in self.views) + tuple(
            "environment_" + x for x in self.environments
        )

    def to_dict(self):
        return dict(asdict(self), context_names=self.context_names,
                    initialization="PyTorch random Linear/Conv1d defaults; no checkpoint",
                    temporal_scope="full observed window; noncausal convolutions")


class TemporalBlock(nn.Module):
    """Two dilated convolutions, with LayerNorm over channels at each frame."""

    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
            for _ in range(2)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, frame_mask):
        residual = x
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x.transpose(1, 2)).transpose(1, 2)
            x = self.dropout(F.gelu(norm(x)))
            # Missing frames cannot become extrema through convolution biases.
            x = x * frame_mask.unsqueeze(-1)
        return (residual + x) * frame_mask.unsqueeze(-1)


class GaitTCN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or TCNConfig()
        if c.channels != TCNConfig().channels:
            raise ValueError("This version requires the declared 2D/confidence/mask channels")
        self.frame_embedding = nn.Linear(c.joints * len(c.channels), c.width)
        self.blocks = nn.ModuleList([
            TemporalBlock(c.width, d, c.dropout) for d in c.dilations
        ])
        self.readout = nn.Sequential(
            nn.Linear(2 * c.width + len(c.context_names), c.readout_width),
            nn.GELU(), nn.Dropout(c.dropout),
        )
        self.classifier = nn.Linear(c.readout_width, len(c.classes))
        self.numeric_heads = nn.ModuleList([
            nn.Linear(c.readout_width, len(c.outcomes)) for _ in c.numeric_methods
        ])
        self.evaluability = nn.Linear(c.readout_width, len(c.outcomes))
        if self.parameter_count >= 500_000:
            raise ValueError("The declared assessment-model size budget is exceeded")

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, pose: Tensor, context: Tensor):
        c = self.config
        if pose.ndim != 4 or tuple(pose.shape[1:]) != (c.frames, c.joints, len(c.channels)):
            raise ValueError("Expected B x 30 x H36M-17 x 4 pose tensor")
        if tuple(context.shape) != (pose.shape[0], len(c.context_names)):
            raise ValueError("Context must contain only the declared view/environment channels")
        if not torch.isfinite(pose).all() or not torch.isfinite(context).all():
            raise ValueError("Model tensors require finite placeholders and explicit masks")
        valid = pose[..., 3]
        if not ((valid == 0) | (valid == 1)).all():
            raise ValueError("The joint-validity channel must be binary")
        joint_mask = valid.bool()
        frame_mask = joint_mask.any(dim=-1)
        if not frame_mask.any(dim=1).all():
            raise ValueError("Exclude all-invalid windows before model inference")
        # Arbitrary finite placeholders at invalid joints must not affect a prediction.
        pose = torch.where(joint_mask.unsqueeze(-1), pose, torch.zeros_like(pose))
        x = F.gelu(self.frame_embedding(pose.flatten(2)))
        x = x * frame_mask.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, frame_mask)
        mean = x.sum(dim=1) / frame_mask.sum(dim=1, keepdim=True)
        maximum = x.masked_fill(~frame_mask.unsqueeze(-1), -torch.inf).amax(dim=1)
        z = self.readout(torch.cat((mean, maximum, context), dim=-1))
        return dict(
            class_logits=self.classifier(z),
            numeric=torch.stack([head(z) for head in self.numeric_heads], dim=1),
            evaluability_logits=self.evaluability(z),
        )


def masked_multitask_loss(outputs, targets, masks, method_index, *,
                          positive_class_weight=None, sample_weight=None):
    """Mask unknown labels explicitly; numeric targets are standardized upstream.

    The caller fits normalization/positive weights on training data only and masks
    disabled classes. Absence of a reference is never converted into a movement
    negative or an evaluability negative here. Method selects only a numeric head;
    it is not an input to the assessment network.
    """
    batch = outputs["numeric"].shape[0]
    if method_index.dtype != torch.long or tuple(method_index.shape) != (batch,):
        raise ValueError("Expected one integer numeric-method index per window")
    if (method_index < 0).any() or (method_index >= outputs["numeric"].shape[1]).any():
        raise ValueError("Unknown numeric pose method")
    predictions = dict(
        classification=outputs["class_logits"],
        regression=outputs["numeric"][torch.arange(batch, device=method_index.device), method_index],
        evaluability=outputs["evaluability_logits"],
    )
    if sample_weight is None:
        sample_weight = predictions["regression"].new_ones(batch)
    if tuple(sample_weight.shape) != (batch,) or not torch.isfinite(sample_weight).all() or (sample_weight < 0).any():
        raise ValueError("Sample weights must be finite, nonnegative and per-window")
    losses = {}
    for name, prediction in predictions.items():
        mask = masks[name]
        target = targets[name]
        if mask.dtype != torch.bool or mask.shape != prediction.shape or target.shape != prediction.shape:
            raise ValueError(f"Target/mask shape or type mismatch: {name}")
        if not torch.isfinite(target[mask]).all():
            raise ValueError(f"An available target is nonfinite: {name}")
        safe_target = torch.where(mask, target, torch.zeros_like(target))
        if name == "regression":
            element_loss = F.huber_loss(prediction, safe_target, reduction="none", delta=1.0)
        else:
            if not ((target[mask] == 0) | (target[mask] == 1)).all():
                raise ValueError(f"Available binary targets must be zero or one: {name}")
            element_loss = F.binary_cross_entropy_with_logits(
                prediction, safe_target, reduction="none",
                pos_weight=positive_class_weight if name == "classification" else None,
            )
        weight = mask.to(prediction.dtype) * sample_weight[:, None]
        losses[name] = (element_loss * weight).sum() / weight.sum().clamp_min(
            torch.finfo(prediction.dtype).tiny
        )
    losses["total"] = losses["classification"] + .5 * losses["regression"] + .2 * losses["evaluability"]
    return losses
