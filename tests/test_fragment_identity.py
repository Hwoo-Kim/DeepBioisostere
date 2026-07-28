"""The fragment library csv is the authoritative fragment identity.

RDKit's canonical SMILES ranking for stereocentres changed after
fragment_library.csv was generated under rdkit 2022.03. Under 2026.03, 1712 of
the 145854 insertion fragments round-trip to a *different* string, all of them
stereochemical, e.g.

    [5*]N1C[C@H](C)O[C@H](C)C1  ->  [5*]N1C[C@@H](C)O[C@@H](C)C1

If the parsed fragment features are keyed on the re-canonicalised form, those
fragments become unfindable by the csv's own SMILES and loading the library
dies with a KeyError. Training data references fragments by NEW-FRAG-IDX, i.e.
csv row order, so the csv string is the identity that must be preserved.

These tests pin that invariant without needing the 13 MB csv or the ~1 hour
preprocessing pass.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem, RDLogger  # noqa: E402

from deepbioisostere.feature import from_mol  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# Real fragments from fragment_library.csv whose canonical form drifted.
DRIFTING_FRAGMENTS = [
    "[5*]N1C[C@H](C)O[C@H](C)C1",
    "[5*]N1C[C@H](C)N[C@H](C)C1",
    "[5*]N1C[C@H]2C(N)[C@H]2C1",
    "[5*]N1C[C@H](O)[C@@H](SC)C1",
    "[5*]N1[C@H]2CC[C@@H]1C1CNCC12",
]

STABLE_FRAGMENTS = [
    "[16*]c1ccc(F)cc1",
    "[3*]OC",
    "[5*]N1CCOCC1",
]


@pytest.mark.parametrize("smi", DRIFTING_FRAGMENTS)
def test_these_fragments_really_do_drift(smi):
    """Guard the premise: if rdkit ever stabilises, this test should be revisited."""
    assert Chem.MolToSmiles(Chem.MolFromSmiles(smi)) != smi, (
        f"{smi} no longer drifts under rdkit "
        f"{__import__('rdkit').__version__}; the workaround may be revisitable."
    )


@pytest.mark.parametrize("smi", DRIFTING_FRAGMENTS + STABLE_FRAGMENTS)
def test_from_mol_preserves_the_supplied_smiles(smi):
    """This is the invariant the fragment library depends on."""
    feature = from_mol(Chem.MolFromSmiles(smi), type="Frag", original_smiles=smi)
    assert feature.smiles == smi


@pytest.mark.parametrize("smi", DRIFTING_FRAGMENTS)
def test_without_original_smiles_the_key_would_be_wrong(smi):
    """Document the failure mode, so an accidental revert is caught here."""
    feature = from_mol(Chem.MolFromSmiles(smi), type="Frag")
    assert feature.smiles != smi, (
        "from_mol re-canonicalised to the same string; the regression this "
        "test guards would no longer be reproducible."
    )


def test_fragment_library_csv_is_self_consistent_if_present():
    """Full-library check. Skipped when the csv is not in the checkout."""
    import pathlib

    csv = pathlib.Path(__file__).resolve().parents[1] / "fragment_library"
    csv = csv / "fragment_library.csv"
    if not csv.is_file():
        pytest.skip("fragment_library.csv not available")

    import pandas as pd

    df = pd.read_csv(csv, sep="\t", dtype={"DATA-TYPE": str, "BRICS-TYPE": str})
    smis = df[df["NEW-OLD"] == "new"]["FRAG-SMI"].tolist()

    # Every fragment must parse, and keying by the supplied string must be
    # injective -- otherwise the feature dict would silently lose entries.
    assert all(Chem.MolFromSmiles(s) is not None for s in smis)
    assert len(set(smis)) == len(smis), "duplicate FRAG-SMI in the library"
