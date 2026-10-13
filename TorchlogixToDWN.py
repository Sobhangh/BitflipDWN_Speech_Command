"""Convert trained torchlogix models into DWN (``torch_dwn``) models.

Supported torchlogix layers and their DWN counterparts:

=====================================  ==========================================
torchlogix                             torch_dwn
=====================================  ==========================================
``LogicConv2d`` (fixed connections)    ``DWNConvLayer`` (+ ``ConstantPad2d``)
``LogicDense`` (fixed / learnable)     ``LUTLayer``
``OrPooling2d``                        ``LogicalORPool2d`` (+ ``ConstantPad2d``)
``GroupSum``                           ``torch_dwn.utils.GroupSum``
anything else without logic (Flatten,  deep-copied unchanged
binarization, ...)
=====================================  ==========================================

Conventions that make the conversion exact:

* torchlogix LUT truth tables are indexed MSB-first (input 0 is the most
  significant address bit); ``LUTLayer`` addresses LSB-first (mapped input 0 is
  address bit 0). Every truth table is therefore bit-reversed on copy.
* torchlogix LUT entries are booleans; ``LUTLayer`` stores values in [-1, 1]
  and outputs ``value > 0``. True -> +1.0, False -> -1.0.
* A torchlogix conv kernel samples leaves over *all* channels (or a sliding
  channel window), which does not match the disjoint channel groups of
  ``DWNConvLayer``. The converted layer therefore uses a single group and
  stores the absolute channel of every leaf in ``Cc``.
* Padding is zero padding in both libraries; it is emitted as an explicit
  ``ConstantPad2d`` in front of the converted layer.
"""
import copy
import os
import sys

import torch
from torch import nn

# The cloned repo (./torchlogix) uses a src layout; make it importable when
# torchlogix is not pip-installed (otherwise ./torchlogix is seen as a bare folder).
#_TORCHLOGIX_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torchlogix", "src")
# if os.path.isdir(_TORCHLOGIX_SRC) and _TORCHLOGIX_SRC not in sys.path:
#     sys.path.insert(0, _TORCHLOGIX_SRC)

from torch_dwn.conv import DWNConvLayer, LogicalORPool2d
from torch_dwn.lut_layer import LUTLayer
from torch_dwn.utils import GroupSum as DWNGroupSum

from torchlogix.connections import FixedConvConnections
from torchlogix.layers import GroupSum, LogicConv2d, LogicDense, OrPooling2d, OrPooling3d
from torchlogix.layers.base import LogicBase


# --------------------------------------------------------------------------- #
# LUT helpers
# --------------------------------------------------------------------------- #
def _bit_reversal_permutation(rank: int) -> torch.Tensor:
    """``perm[a]`` = ``a`` with its ``rank`` address bits reversed."""
    idx = torch.arange(1 << rank, dtype=torch.int64)
    rev = torch.zeros_like(idx)
    for bit in range(rank):
        rev |= ((idx >> bit) & 1) << (rank - 1 - bit)
    return rev


def _to_dwn_luts(tl_luts: torch.Tensor, rank: int) -> torch.Tensor:
    """Convert boolean torchlogix truth tables ``(..., 2**rank)`` to DWN tables."""
    perm = _bit_reversal_permutation(rank).to(tl_luts.device)
    dwn_luts = tl_luts.bool()[..., perm]
    return dwn_luts.float() * 2.0 - 1.0


# --------------------------------------------------------------------------- #
# Layer converters
# --------------------------------------------------------------------------- #
def _convert_conv2d(layer: LogicConv2d) -> nn.Module:
    conn = layer.connections
    if type(conn) is not FixedConvConnections:
        raise NotImplementedError(
            f"Only FixedConvConnections can be converted, got {type(conn).__name__}."
        )
    rf_h, rf_w = layer.receptive_field_size
    if rf_h != rf_w:
        raise ValueError(f"DWNConvLayer needs a square receptive field, got {(rf_h, rf_w)}.")

    L, K, depth = layer.lut_rank, layer.num_kernels, layer.tree_depth
    S = L ** (depth - 1)  # nodes in the first tree level

    # indices[0]: (L, K, P, S, 3) absolute (h, w, c) of every window position P.
    abs_idx = conn.indices[0].detach().cpu()
    P = abs_idx.shape[2]
    if P != layer.n_kernel_positions:
        raise ValueError("Connection tensor does not match the layer's kernel positions.")

    # Window p only shifts (h, w) by its start offset, so p = 0 holds the
    # per-kernel relative leaf positions (_get_random_receptive_field_tensor).
    # Verify that the sliding window matches the DWN convention
    # (row-major positions, offset = position * stride).
    out_h, out_w = layer.kernel_positions
    ph = torch.arange(out_h, dtype=abs_idx.dtype).repeat_interleave(out_w) * layer.stride
    pw = torch.arange(out_w, dtype=abs_idx.dtype).repeat(out_h) * layer.stride
    expected_shift = torch.stack([ph, pw, torch.zeros_like(ph)], dim=-1).view(1, 1, P, 1, 3)
    if not torch.equal(abs_idx - abs_idx[:, :, :1], expected_shift.expand_as(abs_idx)):
        raise ValueError("Sliding-window offsets differ from the DWN convention.")

    rel = abs_idx[:, :, 0]  # (L, K, S, 3)
    # Leaf f = s * L + l is input l of first-level node s -> (K, S * L, 3)
    leaves = rel.permute(1, 2, 0, 3).reshape(K, S * L, 3)
    ch, cw, cc = leaves[..., 0], leaves[..., 1], leaves[..., 2]
    if int(ch.max()) >= rf_h or int(cw.max()) >= rf_w:
        raise ValueError("Leaf positions fall outside the receptive field.")
    if int(cc.max()) >= layer.channels:
        raise ValueError("Leaf channel index out of range.")

    dwn = DWNConvLayer(
        in_channels=layer.channels,
        depth=depth,
        lut_rank=L,
        kernels=K,
        receptive_field=rf_h,
        stride=layer.stride,
        groups=1,  # one group: Cc holds absolute channels
        ste=True,
        flatten_output=False,
        random_kernel_groups=False,
        learnable_connections=False,
    )

    with torch.no_grad():
        dwn.kernel_groups.zero_()
        dwn.Cc.copy_(cc)
        dwn.Ch.copy_(ch)
        dwn.Cw.copy_(cw)

        tl_luts = layer.get_luts()  # [level][node] -> (K, 2**L)
        for level, lut_layer in enumerate(dwn.lut_layers):
            stacked = torch.stack([t.detach().cpu() for t in tl_luts[level]])  # (nodes, K, 2**L)
            nodes = dwn.level_node_counts[level]
            if stacked.shape[0] != nodes:
                raise ValueError(f"Level {level}: expected {nodes} nodes, got {stacked.shape[0]}.")
            per_kernel_major = stacked.permute(1, 0, 2).reshape(K * nodes, 1 << L)
            lut_layer.luts.copy_(_to_dwn_luts(per_kernel_major, L))

    if layer.padding > 0:
        return nn.Sequential(nn.ConstantPad2d(layer.padding, 0.0), dwn)
    return dwn


def _convert_dense(layer: LogicDense) -> LUTLayer:
    conn = layer.connections
    if hasattr(conn, "resolve_connections"):
        indices = conn.resolve_connections()
    else:
        indices = conn.indices
    indices = indices.detach().cpu()  # (L, out_dim)
    L = layer.lut_rank
    if indices.shape != (L, layer.out_dim):
        raise ValueError(f"Unexpected dense connection shape {tuple(indices.shape)}.")

    mapping = indices.t().contiguous().to(torch.int32)  # (out_dim, L), column l = input l
    dwn = LUTLayer(
        input_size=layer.in_dim,
        output_size=layer.out_dim,
        n=L,
        mapping=mapping,
        ste=True,
    )
    with torch.no_grad():
        dwn.luts.copy_(_to_dwn_luts(layer.get_luts().detach().cpu(), L))
    return dwn


def _convert_or_pool(layer: OrPooling2d) -> nn.Module:
    pool = LogicalORPool2d(layer.kernel_size, layer.stride)
    padding = layer.padding
    if isinstance(padding, (tuple, list)):
        if len(set(padding)) != 1:
            raise ValueError(f"Only symmetric padding is supported, got {padding}.")
        padding = padding[0]
    if padding > 0:
        # Zero is the OR identity, so zero padding equals max_pool's implicit -inf padding.
        return nn.Sequential(nn.ConstantPad2d(int(padding), 0.0), pool)
    return pool


def _convert_group_sum(layer: GroupSum) -> DWNGroupSum:
    if layer.beta != 0.0:
        raise NotImplementedError("GroupSum with beta != 0 has no DWN equivalent.")
    return DWNGroupSum(layer.k, tau=layer.tau)


# --------------------------------------------------------------------------- #
# Model conversion
# --------------------------------------------------------------------------- #
def convert_module(module: nn.Module) -> nn.Module:
    """Recursively convert ``module``; the input module is left untouched."""
    if isinstance(module, LogicConv2d):
        return _convert_conv2d(module)
    if isinstance(module, LogicDense):
        return _convert_dense(module)
    if isinstance(module, OrPooling2d):
        return _convert_or_pool(module)
    if isinstance(module, GroupSum):
        return _convert_group_sum(module)
    if isinstance(module, (LogicBase, OrPooling3d)):
        raise NotImplementedError(f"No DWN conversion for {type(module).__name__}.")

    # children = list(module.named_children())
    # converted = copy.deepcopy(module)
    # for name, child in children:
    #     setattr(converted, name, convert_module(child))
    # return converted
    return module


def convert_torchlogix_to_dwn(model: nn.Module) -> nn.Module:
    """Convert a (convolutional) torchlogix model into a DWN model.

    The returned model is on CPU. Move it to CUDA (``.cuda()``) before use, as
    ``LUTLayer`` only has a CUDA kernel. Inputs must be binary (0/1) tensors,
    exactly as for the torchlogix model in eval mode.
    """
    return convert_module(model)


def convert_layers_to_dwn_sequential(sequential: nn.Sequential) -> nn.Sequential:
    """Convert a torchlogix ``nn.Sequential`` into a DWN ``nn.Sequential``.

    Each child is converted with :func:`convert_module`, in order. Layers that
    need explicit zero padding (``LogicConv2d`` / ``OrPooling2d`` with
    ``padding > 0``) expand into ``ConstantPad2d`` followed by the DWN layer, so
    the result can have more modules than the input. Non-logic modules such as
    ``Flatten`` or binarization are copied unchanged.

    Example:
        dwn_model = convert_layers_to_dwn_sequential(model.net).cuda()
    """
    if not isinstance(sequential, nn.Sequential):
        raise TypeError(f"Expected nn.Sequential, got {type(sequential).__name__}.")
    converted = []
    for layer in sequential:
        new_layer = convert_module(layer)
        if type(new_layer) is nn.Sequential:
            converted.extend(new_layer.children())
        else:
            converted.append(new_layer)
    return nn.Sequential(*converted)


@torch.no_grad()
def check_equivalence(
    tl_model: nn.Module,
    dwn_model: nn.Module,
    x: torch.Tensor,
    device: str = "cuda",
    atol: float = 1e-5,
) -> bool:
    """Compare eval-mode outputs of both models on ``x`` (same input for both)."""
    tl_model = copy.deepcopy(tl_model).to(device).eval()
    dwn_model = dwn_model.to(device).eval()
    x = x.to(device)
    ref = tl_model(x).float()
    out = dwn_model(x).float()
    return bool(torch.allclose(ref, out, atol=atol))
