"""Native-torch replacements for the ``torch_scatter`` ops used by this package.

``torch_scatter`` ships as a compiled extension that must be built against an
exact torch/CUDA pair, and it is by a wide margin the most common install
failure for users of this project. It is needed here for exactly two functions,
both of which modern ``torch`` can express natively.

The contract below deliberately mirrors ``torch_scatter`` rather than improving
on it, because the published checkpoints were trained against its semantics:

* ``dim`` defaults to ``-1``, not ``0``. ``model.py`` calls ``scatter_mean``
  without a ``dim``, which is only equivalent to ``dim=0`` because ``src`` there
  happens to be 1-D.
* When ``dim_size`` is ``None`` the output length along ``dim`` is
  ``int(index.max()) + 1`` -- and ``0`` for an empty ``index``. No call site in
  this package passes ``dim_size``, so if a batch's trailing elements contribute
  no entries the output is genuinely shorter than the batch. That is upstream
  behaviour and must not be "fixed" here; doing so would silently change tensor
  shapes downstream.
* ``scatter_mean`` divides by a per-index count clamped to a minimum of 1, so
  output rows that no input touched come out as ``0``. Note that
  ``Tensor.scatter_reduce_(reduce="mean", include_self=False)`` instead leaves
  such rows *untouched*, which is why it is not a drop-in replacement.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["broadcast", "scatter_sum", "scatter_mean"]


def broadcast(src: Tensor, other: Tensor, dim: int) -> Tensor:
    """Expand ``src`` so it can index/align against ``other`` along ``dim``."""
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand(other.size())


def _output_size(
    src: Tensor, index: Tensor, dim: int, dim_size: int | None
) -> list[int]:
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    return size


def scatter_sum(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Tensor | None = None,
    dim_size: int | None = None,
) -> Tensor:
    """Sum ``src`` into buckets given by ``index`` along ``dim``.

    Equivalent to ``torch_scatter.scatter_sum``.
    """
    index = broadcast(index, src, dim)
    if out is not None:
        return out.scatter_add_(dim, index, src)
    size = _output_size(src, index, dim, dim_size)
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


def scatter_mean(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Tensor | None = None,
    dim_size: int | None = None,
) -> Tensor:
    """Mean of ``src`` per bucket given by ``index`` along ``dim``.

    Equivalent to ``torch_scatter.scatter_mean``. Buckets with no contributing
    element yield ``0`` rather than ``NaN``.
    """
    summed = scatter_sum(src, index, dim, out, dim_size)
    resolved_dim_size = summed.size(dim)

    # The count is scattered with the *un-broadcast* index, along the axis of
    # index that corresponds to `dim`.
    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=summed.dtype, device=summed.device)
    count = scatter_sum(ones, index, index_dim, None, resolved_dim_size)
    count = count.clamp(min=1)
    count = broadcast(count, summed, dim)

    # Upstream divides in place, which is observable when the caller supplies
    # `out`. Kept identical rather than "improved" to a functional divide.
    if summed.is_floating_point():
        return summed.true_divide_(count)
    return summed.div_(count, rounding_mode="floor")
