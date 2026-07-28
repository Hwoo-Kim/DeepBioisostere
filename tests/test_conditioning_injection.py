"""Contract tests for ``Generator._inject_conditioning``.

This method is the single place where a property target enters the network. It
exists because there used to be four copies of the same block -- three in
``generate.py`` and one in ``baseline_generator.py`` -- and the fourth had
drifted to a different scheme: it appended the condition to the *fragment*
embeddings *after* the AMPN instead of splicing it into the *atom* features
*before* it. Against the published checkpoints that path could not run at all;
``model.ampn`` raised

    mat1 and mat2 shapes cannot be multiplied (55x66 and 68x128)

because the first linear layer wants ``mol_node_features + cond_dim`` inputs.

So the contract under test is a shape-and-placement contract, and it is worth
pinning precisely because getting it wrong the *other* way -- appending at the
end rather than splicing at ``mol_node_features`` -- would not raise. It would
feed the network a silently permuted feature vector.

No model, no fragment library, no GPU: the method only reads
``self.model.mol_node_features``, ``self.conditioner`` and ``self.properties``,
so a stub supplies them.
"""

from __future__ import annotations

import torch

from deepbioisostere.generate import Generator


class _StubModel:
    def __init__(self, mol_node_features: int):
        self.mol_node_features = mol_node_features


class _StubData:
    """Just enough of a PyG Batch: atom features and the atom->fragment map."""

    def __init__(self, x_n: torch.Tensor, x_f: torch.Tensor):
        self.x_n = x_n
        self.x_f = x_f


def _generator(mol_node_features: int, properties, conditioner=object()):
    """A Generator whose __init__ is bypassed -- it would load a 7 GB library."""
    gen = Generator.__new__(Generator)
    gen.model = _StubModel(mol_node_features)
    gen.conditioner = conditioner
    gen.properties = properties
    return gen


def _batch(num_frags: int, properties):
    # One scalar column per property, distinct per fragment so a wrong gather
    # is visible rather than accidentally right.
    return {
        prop: torch.arange(num_frags, dtype=torch.float32).unsqueeze(1) + 10 * i
        for i, prop in enumerate(properties)
    }


def test_condition_is_spliced_at_mol_node_features():
    """The condition lands between the molecule features and the tail, not at the end."""
    mol_node_features, tail, num_atoms, num_frags = 4, 3, 5, 2
    properties = ["logp", "mw"]

    head = torch.arange(num_atoms * mol_node_features, dtype=torch.float32).reshape(
        num_atoms, mol_node_features
    )
    tail_block = torch.full((num_atoms, tail), -1.0)
    x_n = torch.cat([head, tail_block], dim=1)
    x_f = torch.tensor([0, 0, 1, 1, 1])

    gen = _generator(mol_node_features, properties)
    out = gen._inject_conditioning(_batch(num_frags, properties), _StubData(x_n, x_f))

    assert out.x_n.shape == (num_atoms, mol_node_features + len(properties) + tail)
    # Head untouched, tail untouched, condition in the middle.
    torch.testing.assert_close(out.x_n[:, :mol_node_features], head)
    torch.testing.assert_close(out.x_n[:, -tail:], tail_block)

    expected_cond = torch.stack(
        [
            torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]),  # logp: frag index
            torch.tensor([10.0, 10.0, 11.0, 11.0, 11.0]),  # mw: 10 + frag index
        ],
        dim=1,
    )
    torch.testing.assert_close(
        out.x_n[:, mol_node_features : mol_node_features + len(properties)],
        expected_cond,
    )


def test_condition_is_broadcast_per_atom_via_x_f():
    """Every atom gets its own fragment's condition, not the batch's first one."""
    properties = ["qed"]
    x_n = torch.zeros(4, 2)
    x_f = torch.tensor([1, 0, 1, 0])  # deliberately not sorted

    gen = _generator(2, properties)
    out = gen._inject_conditioning({"qed": torch.tensor([[7.0], [9.0]])},
                                   _StubData(x_n, x_f))
    torch.testing.assert_close(out.x_n[:, 2], torch.tensor([9.0, 7.0, 9.0, 7.0]))


def test_properties_are_concatenated_in_self_properties_order():
    """Column order follows self.properties, which Generator.__init__ sorts."""
    properties = ["logp", "mw", "qed"]
    gen = _generator(1, properties)
    batch = {
        "logp": torch.tensor([[1.0]]),
        "mw": torch.tensor([[2.0]]),
        "qed": torch.tensor([[3.0]]),
    }
    out = gen._inject_conditioning(batch, _StubData(torch.zeros(1, 1),
                                                   torch.tensor([0])))
    torch.testing.assert_close(out.x_n[0, 1:4], torch.tensor([1.0, 2.0, 3.0]))


def test_no_conditioner_is_a_no_op():
    x_n = torch.randn(3, 5)
    gen = _generator(5, ["logp"], conditioner=None)
    out = gen._inject_conditioning({"logp": torch.tensor([[1.0]])},
                                   _StubData(x_n.clone(), torch.tensor([0, 0, 0])))
    torch.testing.assert_close(out.x_n, x_n)


def test_empty_properties_is_a_no_op():
    """properties=None is what Generator.__init__ stores when none are given."""
    x_n = torch.randn(3, 5)
    gen = _generator(5, None)
    out = gen._inject_conditioning({}, _StubData(x_n.clone(), torch.tensor([0, 0, 0])))
    torch.testing.assert_close(out.x_n, x_n)


def test_width_matches_what_the_first_linear_layer_expects():
    """The regression itself: skipping injection leaves the AMPN short by cond_dim.

    This is the shape that produced
    ``mat1 and mat2 shapes cannot be multiplied (55x66 and 68x128)`` -- 66 real
    columns against a layer built for 68.
    """
    mol_node_features, properties = 66, ["mw", "qed"]
    x_n = torch.zeros(55, mol_node_features)
    gen = _generator(mol_node_features, properties)
    out = gen._inject_conditioning(_batch(1, properties),
                                   _StubData(x_n, torch.zeros(55, dtype=torch.long)))
    assert out.x_n.shape[1] == mol_node_features + len(properties) == 68
