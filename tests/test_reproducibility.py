"""Numerical regression against the pre-refactor environment.

The values below were produced by the *original* code and environment
(rdkit 2022.03, torch 1.11) and are recorded as saved cell outputs in
``example.ipynb`` at commit fce5f6c, before this refactor began. They are the
only surviving ground truth from that stack: the original dependency set is no
longer installable (torch 1.11+cu113 and the torch-scatter/sparse/cluster
wheels for it are gone from the PyG index), so a live A/B is impossible.

If these drift, the property calculators no longer agree with what the paper
reported, which would invalidate every conditioning target downstream.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem, RDLogger  # noqa: E402

from deepbioisostere import calc_logP, calc_Mw, calc_QED, calc_SAscore  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# smiles -> (logP, QED, Mw, SAscore) as printed to 3 decimal places at fce5f6c.
REFERENCE = {
    "ClC(Cc1c(C(Nc2c(Br)cccc2)=O)cccc1)=O": (4.009, 0.845, 350.966, 1.942),
    "Cc1ccc2cnc(N(C)CCc3ccccn3)nc2c1": (3.012, 0.735, 278.153, 2.309),
}


@pytest.mark.parametrize("smi,expected", REFERENCE.items(), ids=["mol1", "mol2"])
def test_properties_match_pre_refactor_environment(smi, expected):
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    got = (calc_logP(mol), calc_QED(mol), calc_Mw(mol), calc_SAscore(mol))
    for name, g, e in zip(("logP", "QED", "Mw", "SAscore"), got, expected):
        assert f"{g:.3f}" == f"{e:.3f}", (
            f"{name} for {smi} is {g:.3f}, but the pre-refactor environment "
            f"reported {e:.3f}"
        )


def test_property_calculators_are_deterministic():
    """Same input, same answer -- guards against any hidden global state."""
    smi = next(iter(REFERENCE))
    runs = [
        (
            calc_logP(Chem.MolFromSmiles(smi)),
            calc_QED(Chem.MolFromSmiles(smi)),
            calc_Mw(Chem.MolFromSmiles(smi)),
            calc_SAscore(Chem.MolFromSmiles(smi)),
        )
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]
