"""Dual-stream YOLO11 with cross-attention between RGB and thermal."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from mm_yolo.attention import CrossModalAttention

# YOLO11 backbone layers whose maps the neck reads: P3, P4, P5.
FUSE_INDEXES = (4, 6, 10)
BACKBONE_LEN = 11

# Checkpoint from the full FASDD smoke run. Reused when it is on disk.
DEFAULT_RGB_WEIGHTS = Path("runs/detect/smoke_full_100ep/weights/best.pt")


class _DetectRef:
    """Indexable tail so v8DetectionLoss can read model.model[-1]."""

    def __init__(self, detect: nn.Module):
        self._detect = detect

    def __getitem__(self, index: int) -> nn.Module:
        if index not in (-1, 0):
            raise IndexError(index)
        return self._detect


def resolve_rgb_weights(path: Path | None) -> Path:
    """Use the finished smoke run when this machine already has it."""
    if path is not None:
        candidate = path.expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise SystemExit(
            f"Missing RGB weights: {candidate}\n"
            "Pass the YOLO11s smoke checkpoint, usually "
            "runs/detect/smoke_full_100ep/weights/best.pt"
        )
    if DEFAULT_RGB_WEIGHTS.is_file():
        print(f"Reusing existing YOLO results: {DEFAULT_RGB_WEIGHTS.resolve()}")
        return DEFAULT_RGB_WEIGHTS.resolve()
    raise SystemExit(
        "No RGB checkpoint found at runs/detect/smoke_full_100ep/weights/best.pt.\n"
        "This setup reuses that trained smoke model instead of starting over.\n"
        "On the machine that already ran it, pull the latest code and train from that file."
    )


def _load_base(weights: Path):
    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(weights, map_location="cpu")
    model = ckpt["model"] if isinstance(ckpt, dict) else ckpt
    model = model.float().eval()
    return model


class RGBThermalYOLO(nn.Module):
    """YOLO11 RGB stream + YOLO11 thermal stream, fused at P3, P4, and P5.

    The RGB backbone and, by default, the neck and detection head are copies of
    the already-trained smoke checkpoint and stay frozen. Thermal features enter
    through cross-attention. At initialization the fusion passes RGB through, so
    the detector matches that checkpoint before any thermal updates.
    """

    def __init__(
        self,
        rgb: nn.ModuleList,
        thermal: nn.ModuleList,
        neck: nn.ModuleList,
        fuse: nn.ModuleDict,
        stride: torch.Tensor,
        names: dict,
        args: SimpleNamespace,
        scale: str,
        freeze_rgb: bool,
        freeze_head: bool,
    ):
        super().__init__()
        self.rgb = rgb
        self.thermal = thermal
        self.neck = neck
        self.fuse = fuse
        self.names = names
        self.args = args
        self.scale = scale
        self.nc = int(neck[-1].nc)
        self.freeze_rgb = freeze_rgb
        self.freeze_head = freeze_head
        self.stride = stride
        self._apply_freeze()

    @property
    def model(self) -> _DetectRef:
        return _DetectRef(self.neck[-1])

    def _apply_freeze(self) -> None:
        # Ultralytics checkpoints are saved with requires_grad False.
        for param in self.parameters():
            param.requires_grad = True
        if self.freeze_rgb:
            for param in self.rgb.parameters():
                param.requires_grad = False
            self.rgb.eval()
        if self.freeze_head:
            for param in self.neck.parameters():
                param.requires_grad = False
            self.neck.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_rgb:
            self.rgb.eval()
        if self.freeze_head:
            self.neck.eval()
            # Loss needs the Detect head's training return value. Leave its
            # BatchNorm children in eval so the reused statistics stay put.
            self.neck[-1].training = mode
        return self

    def _apply(self, fn):
        self = super()._apply(fn)
        detect = self.neck[-1]
        if isinstance(detect.stride, torch.Tensor):
            detect.stride = fn(detect.stride)
        for name in ("anchors", "strides"):
            value = getattr(detect, name, None)
            if isinstance(value, torch.Tensor) and value.numel():
                setattr(detect, name, fn(value))
        return self

    def _stem(self, layers: nn.ModuleList, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        for layer in layers:
            if layer.f != -1:
                x = features[layer.f] if isinstance(layer.f, int) else [features[j] for j in layer.f]
            x = layer(x)
            features.append(x)
        return features

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor):
        if self.freeze_rgb:
            self.rgb.eval()
        if self.freeze_head:
            self.neck.eval()
        detect_training = getattr(self, "_force_detect_train", None)
        self.neck[-1].training = self.training if detect_training is None else detect_training

        rgb_feats = self._stem(self.rgb, rgb)
        thermal_feats = self._stem(self.thermal, thermal)
        features = list(rgb_feats)
        for index in FUSE_INDEXES:
            features[index] = self.fuse[str(index)](rgb_feats[index], thermal_feats[index])

        x = features[-1]
        for layer in self.neck:
            if layer.f != -1:
                x = (
                    features[layer.f]
                    if isinstance(layer.f, int)
                    else [x if j == -1 else features[j] for j in layer.f]
                )
            x = layer(x)
            features.append(x)
        return x

    @classmethod
    def from_checkpoint(
        cls,
        weights: Path,
        freeze_rgb: bool = True,
        freeze_head: bool = True,
    ) -> "RGBThermalYOLO":
        base = _load_base(weights)
        scale = str(getattr(base, "yaml", {}).get("scale") or "s")
        raw_args = base.args if isinstance(base.args, dict) else vars(base.args)
        args = SimpleNamespace(
            box=float(raw_args.get("box", 7.5)),
            cls=float(raw_args.get("cls", 0.5)),
            dfl=float(raw_args.get("dfl", 1.5)),
        )
        names = dict(base.names)
        return cls._from_base(base, freeze_rgb, freeze_head, names, scale, args)

    @classmethod
    def from_scale(
        cls,
        scale: str,
        nc: int,
        freeze_rgb: bool = True,
        freeze_head: bool = True,
        args: SimpleNamespace | None = None,
    ) -> "RGBThermalYOLO":
        from ultralytics.nn.tasks import DetectionModel

        base = DetectionModel(f"yolo11{scale}.yaml", ch=3, nc=nc, verbose=False)
        names = {i: f"{i}" for i in range(nc)}
        if args is None:
            args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        return cls._from_base(base, freeze_rgb, freeze_head, names, scale, args)

    @classmethod
    def _from_base(cls, base, freeze_rgb, freeze_head, names, scale, args) -> "RGBThermalYOLO":
        layers = list(base.model)
        if len(layers) <= BACKBONE_LEN:
            raise RuntimeError(f"Expected a YOLO11 detect model, got {len(layers)} layers")
        # Clone so this module does not share parameters with the loaded checkpoint.
        rgb = nn.ModuleList(_clone(layer) for layer in layers[:BACKBONE_LEN])
        thermal = nn.ModuleList(_clone(layer) for layer in layers[:BACKBONE_LEN])
        neck = nn.ModuleList(_clone(layer) for layer in layers[BACKBONE_LEN:])
        channels = _backbone_channels(rgb)
        fuse = nn.ModuleDict({str(i): CrossModalAttention(channels[i]) for i in FUSE_INDEXES})
        stride = base.stride.detach().float().cpu().clone()
        neck[-1].stride = stride.clone()
        return cls(rgb, thermal, neck, fuse, stride, names, args, scale, freeze_rgb, freeze_head)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def _clone(module: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(module)


def _backbone_channels(rgb: nn.ModuleList) -> dict[int, int]:
    was_training = rgb.training
    rgb.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, 64, 64)
        features: list[torch.Tensor] = []
        for layer in rgb:
            x = layer(x)
            features.append(x)
    rgb.train(was_training)
    return {index: int(features[index].shape[1]) for index in FUSE_INDEXES}
