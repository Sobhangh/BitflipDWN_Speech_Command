import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dwn as dwn


class STEResidualAdd(nn.Module):
    """Residual addition that enforces a binary signed activation range [-1, 1]."""

    def __init__(self):
        super().__init__()

    def forward(self, x, residual):
        if x.shape != residual.shape:
            raise ValueError(
                f"Residual shape mismatch: x={tuple(x.shape)}, residual={tuple(residual.shape)}"
            )

        merged = x + residual
        merged = torch.clamp(merged, -1.0, 1.0)
        out = dwn.STEFunction.apply(merged)
        return out * 2.0 - 1.0


class DWNResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        depth=2,
        lut_rank=4,
        receptive_field=3,
        channels_per_group=4,
        debug=False,
    ):
        super().__init__()

        if not torch.cuda.is_available():
            raise RuntimeError("DWNConvLayer requires CUDA.")

        pad = (receptive_field - 1) // 2

        self.conv1 = dwn.DWNConvLayer(
            in_channels=in_channels,
            depth=depth,
            lut_rank=lut_rank,
            kernels=out_channels,
            receptive_field=receptive_field,
            stride=stride,
            channels_per_group=channels_per_group,
            padding=pad,
            ste=True,
            flatten_output=False,
            random_kernel_groups=False,
            learnable_connections=False,
            debug=debug,
        )
        self.conv2 = dwn.DWNConvLayer(
            in_channels=out_channels,
            depth=depth,
            lut_rank=lut_rank,
            kernels=out_channels,
            receptive_field=receptive_field,
            stride=1,
            channels_per_group=channels_per_group,
            padding=pad,
            ste=True,
            flatten_output=False,
            random_kernel_groups=False,
            learnable_connections=False,
            debug=debug,
        )
        self.residual_add = STEResidualAdd()

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = dwn.DWNConvLayer(
                in_channels=in_channels,
                depth=depth,
                lut_rank=lut_rank,
                kernels=out_channels,
                receptive_field=receptive_field,
                stride=stride,
                channels_per_group=channels_per_group,
                padding=pad,
                ste=True,
                flatten_output=False,
                random_kernel_groups=False,
                learnable_connections=False,
                debug=debug,
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_add(out, residual)
        return out


class DWNResNetCIFAR(nn.Module):
    # CIFAR-10 input: [N, 9, 32, 32]
    # Stem:           [N, 32, 32, 32]
    # Block1:         [N, 32, 32, 32]
    # Block2:         [N, 64, 16, 16]   (stride=2 downsample)
    # Block3:         [N, 64, 16, 16]
    # Flatten:        [N, 64 * 16 * 16]
    # LUT1:          [N, hidden]
    # LUT2:          [N, hidden]
    # GroupSum:      [N, 10]
    def __init__(self, num_classes=10, base_channels=32, debug=False):
        super().__init__()

        self.debug = debug
        self.stem = dwn.DWNConvLayer(
            in_channels=9,
            depth=2,
            lut_rank=4,
            kernels=base_channels,
            receptive_field=3,
            stride=1,
            channels_per_group=3,
            padding=(3 - 1) // 2,
            ste=True,
            flatten_output=False,
            random_kernel_groups=False,
            learnable_connections=False,
            debug=debug,
        )

        self.block1 = DWNResidualBlock(
            in_channels=base_channels,
            out_channels=base_channels,
            stride=1,
            depth=2,
            lut_rank=4,
            receptive_field=3,
            channels_per_group=4,
            debug=debug,
        )
        self.block2 = DWNResidualBlock(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            stride=2,
            depth=2,
            lut_rank=4,
            receptive_field=3,
            channels_per_group=4,
            debug=debug,
        )
        self.block3 = DWNResidualBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels * 2,
            stride=1,
            depth=2,
            lut_rank=4,
            receptive_field=3,
            channels_per_group=4,
            debug=debug,
        )

        flat_dim = base_channels * 2 * 16 * 16
        self.lut1 = dwn.LUTLayer(flat_dim, flat_dim // 2, n=4)
        self.lut2 = dwn.LUTLayer(flat_dim // 2, flat_dim // 4, n=4)
        self.classifier = dwn.GroupSum(k=num_classes, tau=1 / 0.1)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected CIFAR-10 input [N, 9, 32, 32], got {tuple(x.shape)}")
        if x.shape[1] != 9 or x.shape[2:] != (32, 32):
            raise ValueError(f"Expected CIFAR-10 input [N, 9, 32, 32], got {tuple(x.shape)}")

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.lut1(x)
        x = self.lut2(x)
        return self.classifier(x)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("This demo expects a CUDA-enabled GPU for DWNConvLayer.")

    model = DWNResNetCIFAR(num_classes=10)
    model = model.cuda()
    x = torch.randn(2, 9, 32, 32, device='cuda')
    y = model(x)
    print(f"Input: {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
