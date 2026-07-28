"""Contract tests for the native replacements of ``torch_scatter``.

``torch_scatter`` no longer ships a wheel for modern torch (the PyG index
carries only ``pyg_lib``, and PyPI has an sdist-only 2.1.2), so a direct A/B
against the real library is not available. Instead each op is checked against a
brute-force reference built from explicit Python loops. That is independent
evidence rather than a restatement of the implementation.

The edge cases below are the ones that actually bite, because no call site in
this package passes ``dim_size``:

* an index that leaves output rows unreferenced,
* an index whose maximum is below the batch size, so the output is genuinely
  shorter than the batch,
* an empty index,
* ``scatter_mean`` over unreferenced rows, which must yield 0 and not NaN.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from deepbioisostere.scatter import broadcast, scatter_mean, scatter_sum


def ref_scatter(
    src: torch.Tensor, index: torch.Tensor, dim: int, dim_size, reduce: str
):
    """Brute-force reference: accumulate element by element with Python loops."""
    if dim < 0:
        dim = src.dim() + dim
    idx = broadcast(index, src, dim)

    if dim_size is not None:
        out_len = dim_size
    elif idx.numel() == 0:
        out_len = 0
    else:
        out_len = int(idx.max()) + 1

    out_shape = list(src.shape)
    out_shape[dim] = out_len
    out = torch.zeros(out_shape, dtype=src.dtype)
    counts = torch.zeros(out_shape, dtype=torch.long)

    for coord in itertools.product(*[range(s) for s in src.shape]):
        target = list(coord)
        target[dim] = int(idx[coord])
        out[tuple(target)] += src[coord]
        counts[tuple(target)] += 1

    if reduce == "sum":
        return out
    return out / counts.clamp(min=1).to(out.dtype)


# (name, src, index, dim, dim_size)
CASES = [
    (
        "1d contiguous",
        torch.randn(10),
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
        -1,
        None,
    ),
    ("1d with gap", torch.randn(6), torch.tensor([0, 0, 3, 3, 5, 5]), -1, None),
    ("1d single bucket", torch.randn(4), torch.tensor([0, 0, 0, 0]), -1, None),
    ("2d dim0", torch.randn(8, 5), torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]), 0, None),
    ("2d dim0 gap", torch.randn(6, 3), torch.tensor([0, 2, 2, 5, 5, 5]), 0, None),
    (
        "explicit dim_size larger",
        torch.randn(5, 2),
        torch.tensor([0, 0, 1, 1, 2]),
        0,
        7,
    ),
    ("unsorted index", torch.randn(7, 3), torch.tensor([3, 1, 0, 2, 1, 3, 0]), 0, None),
]


@pytest.mark.parametrize(
    "name,src,index,dim,dim_size", CASES, ids=[c[0] for c in CASES]
)
def test_scatter_sum_matches_reference(name, src, index, dim, dim_size):
    got = scatter_sum(src, index, dim, dim_size=dim_size)
    want = ref_scatter(src, index, dim, dim_size, "sum")
    assert got.shape == want.shape
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize(
    "name,src,index,dim,dim_size", CASES, ids=[c[0] for c in CASES]
)
def test_scatter_mean_matches_reference(name, src, index, dim, dim_size):
    got = scatter_mean(src, index, dim, dim_size=dim_size)
    want = ref_scatter(src, index, dim, dim_size, "mean")
    assert got.shape == want.shape
    torch.testing.assert_close(got, want)


def test_output_length_is_index_max_plus_one_not_batch_size():
    """The contract that silently changes shapes if ported carelessly.

    With no ``dim_size``, an index whose maximum is below the batch size
    produces a *shorter* output. Downstream code depends on this, so a port
    that "helpfully" pads to the batch size would be wrong.
    """
    src = torch.randn(6, 4)
    index = torch.tensor([0, 0, 1, 1, 2, 2])  # batch is nominally 10
    out = scatter_sum(src, index, dim=0)
    assert out.shape == (3, 4)


def test_empty_index_yields_empty_output():
    src = torch.randn(0, 4)
    index = torch.empty(0, dtype=torch.long)
    out = scatter_sum(src, index, dim=0)
    assert out.shape == (0, 4)
    assert scatter_mean(src, index, dim=0).shape == (0, 4)


def test_mean_of_unreferenced_rows_is_zero_not_nan():
    """Rows no input touched must be 0.

    ``Tensor.scatter_reduce_(reduce="mean", include_self=False)`` leaves such
    rows untouched instead, which is why it is not a drop-in replacement.
    """
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    index = torch.tensor([0, 3])  # rows 1 and 2 are never written
    out = scatter_mean(src, index, dim=0)
    assert out.shape == (4, 2)
    assert torch.isfinite(out).all(), "unreferenced rows produced NaN/inf"
    torch.testing.assert_close(out[1], torch.zeros(2))
    torch.testing.assert_close(out[2], torch.zeros(2))


def test_default_dim_is_minus_one():
    """model.py calls scatter_mean without a dim; upstream defaults to -1."""
    src = torch.randn(3, 6)
    index = torch.tensor([0, 0, 1, 1, 2, 2])
    torch.testing.assert_close(scatter_sum(src, index), scatter_sum(src, index, dim=-1))
    assert scatter_sum(src, index).shape == (3, 3)


def test_gradients_flow_to_src():
    src = torch.randn(6, 3, requires_grad=True)
    index = torch.tensor([0, 0, 1, 1, 2, 2])
    scatter_sum(src, index, dim=0).sum().backward()
    assert src.grad is not None
    torch.testing.assert_close(src.grad, torch.ones_like(src))


def test_mean_gradient_is_weighted_by_bucket_count():
    src = torch.randn(4, 1, requires_grad=True)
    index = torch.tensor([0, 0, 0, 1])  # bucket 0 has 3 members, bucket 1 has 1
    scatter_mean(src, index, dim=0).sum().backward()
    expected = torch.tensor([[1 / 3], [1 / 3], [1 / 3], [1.0]])
    torch.testing.assert_close(src.grad, expected)


def test_dtype_and_shape_preserved_for_float64():
    src = torch.randn(6, 2, dtype=torch.float64)
    index = torch.tensor([0, 0, 1, 1, 2, 2])
    out = scatter_sum(src, index, dim=0)
    assert out.dtype == torch.float64


def test_real_call_site_shapes():
    """Mirror the four production call sites: 1-D index, dim=0, no dim_size."""
    # layers.py:200  h_f = scatter_sum(src=x_n_emb, index=data.x_f, dim=0)
    x_n_emb = torch.randn(40, 128)
    x_f = torch.randint(0, 7, (40,))
    x_f[-1] = 6  # ensure the max is realised
    assert scatter_sum(src=x_n_emb, index=x_f, dim=0).shape == (7, 128)

    # model.py:265  nPosLoss = scatter_mean(src=<1-D>, index=<1-D>)
    neg = torch.rand(12)
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3])
    assert scatter_mean(src=(1 - neg + 1e-10).log(), index=batch).shape == (4,)
