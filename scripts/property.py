import os
import sys

from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt, qed

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer


def calc_SAscore(mol: Chem.rdchem.Mol):
    return sascorer.calculateScore(mol)


def calc_logP(mol: Chem.rdchem.Mol):
    return MolLogP(mol)


def calc_Mw(mol: Chem.rdchem.Mol):
    return ExactMolWt(mol)


def calc_QED(mol: Chem.rdchem.Mol):
    return qed(mol)


PROPERTIES = ["logp", "mw", "qed", "sa"]
