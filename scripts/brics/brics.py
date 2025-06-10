from itertools import product
from typing import Dict, List, Tuple, Union

import torch
from rdkit import Chem
from rdkit.Chem import BRICS, Atom, BondType, Mol
from rdkit.Chem.rdchem import ChiralType
from torch import Tensor

if __name__ == "__main__":
    from constant import getBricsRule
else:
    from .constant import getBricsRule

SMILES = str


class BRICSTypeMapper:
    r"""
    Method:
      getMapping:
    """
    BRICS_TYPE, BRICS_TYPE_INT, BRICS_SMARTS_FRAG, BRICS_SMARTS_MOL = getBricsRule()

    def __init__(self, inp_type):
        self.inp_type = inp_type
        if inp_type == int:
            self.mapper = self.BRICS_TYPE_INT
        elif inp_type == str:
            self.mapper = self.BRICS_TYPE
        else:
            raise ValueError("BricsMapper type must be int or str.")

    def getMapping(self, brics_type: Tuple[Union[int, str]]) -> Tuple[Tuple]:
        types_list = []
        for i in brics_type:
            types_list.append(self.mapper[i])

        result = []
        for tup in product(*types_list):
            result.append(tuple(sorted(tup)))
        return tuple(set(result))

    def checkCombinations(
        self, brics_types1: List[int], brics_types2: List[int]
    ) -> List[Tuple[int]]:
        # Find allowed mappings (at atom-pair level)
        allowed_mapping = dict()
        for idx1, type1 in enumerate(brics_types1):
            allowed_mapping[idx1] = []
            for idx2, type2 in enumerate(brics_types2):
                if self.inp_type == str:
                    type1, type2 = str(type1), str(type2)
                if type1 in self.mapper[type2]:
                    allowed_mapping[idx1].append(idx2)

        # Examine product of the mappings
        allowed_attachments = []
        val_lists = tuple(allowed_mapping.values())
        attachment_candidates = product(*val_lists)
        for candidate in attachment_candidates:
            if len(candidate) == len(set(candidate)):  # 1-to-1 matching
                allowed_attachments.append(tuple(candidate))

        # example for allowed_attachments: [(1,0,2), (1,2,0)]
        return allowed_attachments


class BRICSModule:
    BRICS_type_mapper = BRICSTypeMapper(inp_type=int)

    @classmethod
    def create_bond(
        cls, rwmol: Chem.RWMol, idx1: int, idx2: int, bondtype: Chem.BondType
    ):

        rwmol.AddBond(idx1, idx2, bondtype)
        for idx in [idx1, idx2]:
            atom = rwmol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "N" and atom.GetIsAromatic() is True:
                atom.SetNumExplicitHs(0)

    @classmethod
    def remove_bond(cls, rwmol, idx1, idx2):
        rwmol.RemoveBond(idx1, idx2)
        for idx in [idx1, idx2]:
            atom = rwmol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "N" and atom.GetIsAromatic() is True:
                atom.SetNumExplicitHs(1)
            else:
                pass

    @classmethod
    def add_dummy_atom(cls, rwmol, index, bondtype=BondType.SINGLE, label=0):
        dummy_atom = Atom("*")
        dummy_atom.SetIsotope(label)  # default: 0
        new_idx = rwmol.AddAtom(dummy_atom)
        cls.create_bond(rwmol, index, new_idx, bondtype)

    @classmethod
    def remove_dummy_atom(cls, rwmol, dummy_index):
        dummy_atom = rwmol.GetAtomWithIdx(dummy_index)
        adj_atom = dummy_atom.GetNeighbors()[0]
        brics_type = dummy_atom.GetIsotope()

        # Consider Chiral Tag
        adj_atom_chiral_tag = adj_atom.GetChiralTag()
        if (
            adj_atom_chiral_tag != ChiralType.CHI_UNSPECIFIED
            and dummy_index > adj_atom.GetIdx()
        ):
            adj_atom.InvertChirality()

        dummy_bonds = dummy_atom.GetBonds()
        for bond in dummy_bonds:
            atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            cls.remove_bond(rwmol, atom1, atom2)
        rwmol.RemoveAtom(dummy_index)

        adj_atom_index = adj_atom.GetIdx()
        return brics_type, adj_atom_index

    @classmethod
    def get_adjacent_fragments(cls, original_mol, change_indice):
        """
        Parameters
        ----------
        cls : BRICSModule
        original_mol :
        change_indice :

        Returns
        -------
        List[Mol]
            Resultant molecules of examining all the composing cases.
            If more than one cases pass the BRICS rule, returning list would contain more than one
            Mol objects.
        """
        To_Break = []
        bond_indice_to_break, bond_types_to_break = [], []
        # [((3, 2), ('3', '4')), ((3, 4), ('3', '4'))]
        BRICS_bonds = list(BRICS.FindBRICSBonds(original_mol))
        for bond in BRICS_bonds:
            bond_idx, bond_type = bond
            changes = [atom_idx in change_indice for atom_idx in bond_idx]
            if any(changes) and not all(changes):
                To_Break.append(bond)
                bond = original_mol.GetBondBetweenAtoms(*bond_idx)
                bond_indice_to_break.append(bond.GetIdx())

                begin_atom_idx = bond.GetBeginAtomIdx()
                end_atom_idx = bond.GetEndAtomIdx()
                if (begin_atom_idx, end_atom_idx) == bond_idx:
                    bond_type = (bond_type[1], bond_type[0])
                bond_types_to_break.append(tuple(map(int, bond_type)))

        frags = []
        # broken_mol = BRICS.BreakBRICSBonds(original_mol, To_Break)
        broken_mol = Chem.FragmentOnBonds(
            original_mol,
            bondIndices=bond_indice_to_break,
            dummyLabels=bond_types_to_break,
        )
        broken_mol_list = list(Chem.GetMolFrags(broken_mol, asMols=True, frags=frags))
        frags = frags[: original_mol.GetNumAtoms()]

        frag_to_be_changed = []
        for idx in change_indice:
            frag_to_be_changed.append(frags[idx])

        for f_id in set(frag_to_be_changed):
            assert frag_to_be_changed.count(f_id) == frags.count(f_id), ValueError(
                "Error occured in brics.get_adjacent_fragments method."
            )

        leaving_frag_smi = Chem.MolToSmiles(broken_mol_list[frag_to_be_changed[0]])
        del broken_mol_list[frag_to_be_changed[0]]
        remaining_frags = broken_mol_list  # list

        return remaining_frags, leaving_frag_smi, frags

    @classmethod
    def get_adj_frag_brics_type(cls, original_mol, change_indice: list):
        change_indice = [int(num) for num in change_indice]
        other_frags, _, _ = cls.get_adjacent_fragments(original_mol, change_indice)
        if other_frags is None:
            return None
        brics_types = []
        for frag in other_frags:
            for atom in frag.GetAtoms():
                if atom.GetSymbol() == "*":
                    brics_types.append(int(atom.GetIsotope()))
        brics_types.sort()

        return tuple(brics_types)
        # return ",".join(brics_types)

    @classmethod
    def get_subgraph_idx_to_brics_type(
        cls,
        original_mol,
        BRICS_bonds,
        atom_frag_indice,
        allowed_subgraph,
        allowed_subgraph_idx,
    ):
        change_indice = [int(num) for num in change_indice]
        other_frags, _, _ = cls.get_adjacent_fragments(original_mol, change_indice)
        if other_frags is None:
            return None
        brics_types = []
        for frag in other_frags:
            for atom in frag.GetAtoms():
                if atom.GetSymbol() == "*":
                    brics_types.append(str(atom.GetIsotope()))
        brics_types.sort()

        return ",".join(brics_types)

    @classmethod
    def compose_mols_with_attachment(
        cls,
        original_smi: SMILES,
        change_indice: List[int],
        new_frag_smi: SMILES,
        attachment: List[List[int]],
        get_leaving_frag_smi: bool,
        verbose=False,
    ) -> Mol:
        """
        Compose two molecular fragments to yield a list of new fragments (or possibly complete molecules).
        This method is for 'one' step in the building process.

        Parameters
        ----------
        cls : BRICSModule
        building_mol : Mol
            A molecular fragment in building.
            It is helpful to think this fragment as a center of the overall building process.
            Thus, it possibly contains more than one fragmented parts (*, in other words).
            One important thing is that '*' might appear in anywhere in indexing; different from
              a general indexing convention. After using remove_dummy_atom, the indices of the later
              atoms are decreased by 1.
        frag_to_compose : Mol
            A molecular fragment to compose to 'building_mol'.
            This is actually an adjacent fragment of the original fragment so it is 'terminal' fragment.
            Thus, it contains only one fragmented part and '*' apperas in last in the atom indexing.
        removal_dummy_atom_idx : int
            Index of a dummy atom in the 'building_mol' that would be removed by composition.
            The adjacent atom will be joined with 'frag_to_compose' fragment.
            'frag_to_compose' has only one dummy atom, so it does not need index.

        Returns
        -------
        Mol
            A resultant molecule of the given composition.
            If no one passes the BRICS rule, it returns None.
        """

        # Read input information
        if verbose:
            print(f"original_smi: {original_smi}")
            print(f"new_frag_smi: {new_frag_smi}")
            print(f"change_indice: {change_indice}")
        mol = Chem.MolFromSmiles(original_smi)
        frag_mol = Chem.MolFromSmiles(new_frag_smi)
        (
            remain_frag_mols,
            leaving_frag_smi,
            atom_frag_indice,
        ) = BRICSModule.get_adjacent_fragments(mol, change_indice)
        leaving_frag_idx = list(set([atom_frag_indice[_] for _ in change_indice]))[0]
        remain_frag_mols.insert(leaving_frag_idx, None)

        attachment = torch.tensor(attachment).transpose(0, 1).tolist()
        if verbose:
            print(f"atom_frag_indice: {atom_frag_indice}")
            print(f"attachment: {attachment}")

        # Fragment atom idx to remaining fragment
        for pair in attachment:
            original_atom_idx, new_frag_atom_idx = pair
            remaining_frag_idx = atom_frag_indice[original_atom_idx]
            # if verbose:
            #    print(f"remaining_frag_idx: {remaining_frag_idx}")
            #    print(f"new_frag_atom_idx: {new_frag_atom_idx}")

            new_frag_atom = frag_mol.GetAtomWithIdx(new_frag_atom_idx)
            props_dict = new_frag_atom.GetPropsAsDict()
            if "remaining_frag_idx" in props_dict:
                new_frag_atom.SetProp(
                    "remaining_frag_idx",
                    props_dict["remaining_frag_idx"] + f",{remaining_frag_idx}",
                )
            else:
                new_frag_atom.SetProp("remaining_frag_idx", f",{remaining_frag_idx}")

        # Combine the fragments following the attachment
        building_mol = frag_mol
        while True:
            for atom in building_mol.GetAtoms():
                props_dict = atom.GetPropsAsDict()
                if "remaining_frag_idx" in props_dict:
                    new_frag_atom_idx = atom.GetIdx()
                    remaining_frag_idx_list = props_dict["remaining_frag_idx"].split(
                        ","
                    )[1:]
                    remaining_frag_idx = int(remaining_frag_idx_list.pop())
                    if len(remaining_frag_idx_list) == 0:
                        atom.ClearProp("remaining_frag_idx")
                    else:
                        atom.SetProp(
                            "remaining_frag_idx",
                            "," + ",".join(remaining_frag_idx_list),
                        )

                    remaining_frag_mol = remain_frag_mols[remaining_frag_idx]
                    building_mol = cls.compose_two_mols(
                        building_mol, remaining_frag_mol, new_frag_atom_idx
                    )
                    break
            else:
                break

        if get_leaving_frag_smi:
            return Chem.MolToSmiles(building_mol), leaving_frag_smi
        else:
            return Chem.MolToSmiles(building_mol)

        ## Read input information
        # removal_dummy_atom = building_mol.GetAtomWithIdx(removal_dummy_atom_idx)
        # removal_dummy_atom_brics_type = removal_dummy_atom.GetIsotope()
        # assert (
        #    removal_dummy_atom.GetSymbol() == "*"
        # ), "An atom corresponding to the given index is not a dummy atom."

        # rwmol_compose_frag = Chem.RWMol(frag_to_compose)
        # for atom in rwmol_compose_frag.GetAtoms():
        #    if atom.GetSymbol() == "*":
        #        (
        #            composing_frag_brics_type,
        #            composing_frag_atom_index,
        #        ) = cls.remove_dummy_atom(rwmol_compose_frag, atom.GetIdx())
        #        break  # Remind: frag_to_compose has only one dummy atom

        ## Get possible dummy atoms based on the BRICS rule
        # possible_brics_types = list(
        #    map(
        #        lambda x: x[0],
        #        cls.BRICS_type_mapper.getMapping([composing_frag_brics_type]),
        #    )
        # )
        # assert (
        #    removal_dummy_atom_brics_type in possible_brics_types
        # ), "An atom corresponding to the given index cannot be composed following the BRICS rule."

        ## Get the composition result
        # copied_building_rwmol = Chem.RWMol(copy.deepcopy(building_mol))
        # (
        #    brics_type,
        #    adj_atom_index,
        # ) = cls.remove_dummy_atom(  # All the indices of the remaining atoms decrease by 1.
        #    copied_building_rwmol, removal_dummy_atom_idx
        # )
        # combined_mol = Chem.CombineMols(copied_building_rwmol, rwmol_compose_frag)
        # combined_rwmol = Chem.RWMol(combined_mol)

        # if brics_type == "7":
        #    bond_type = Chem.BondType.DOUBLE
        # else:
        #    bond_type = Chem.BondType.SINGLE

        # cls.create_bond(
        #    combined_rwmol,
        #    adj_atom_index,
        #    composing_frag_atom_index + copied_building_rwmol.GetNumAtoms(),
        #    bond_type,
        # )
        # return combined_rwmol.GetMol()

    @classmethod
    def compose_two_mols(
        cls, building_mol: Mol, frag_to_compose: Mol, composing_atom_idx: int
    ) -> Mol:
        """
        Compose two molecular fragments to yield a list of new fragments (or possibly complete molecules).
        This method is for 'one' step in the building process.

        Parameters
        ----------
        cls : BRICSModule
        building_mol : Mol
            A molecular fragment in building, the new frament.
            It is helpful to think this fragment as a center of the overall building process.
            Thus, it possibly contains more than one fragmented parts (*, in other words).
            One important thing is that '*' might appear in anywhere in indexing; different from
              a general indexing convention. After using remove_dummy_atom, the indices of the later
              atoms are decreased by 1.
        frag_to_compose : Mol
            A molecular fragment to compose to 'building_mol'.
            This is actually an adjacent fragment of the original fragment so it is 'terminal' fragment.
            Thus, it contains only one fragmented part and '*' apperas in last in the atom indexing.
        composing_atom_idx : int
            Index of an atom in the 'building_mol' that would form a bond with the frag_to_compose.
            An adjacent dummy atom will be removed.
            'frag_to_compose' has only one dummy atom, so it does not need index.

        Returns
        -------
        Mol
            A resultant molecule of the given composition.
            If no one passes the BRICS rule, it returns None.
        """

        # Get removal dummy atom
        # print(building_mol.GetAtomWithIdx(composing_atom_idx).GetSymbol())
        # print(composing_atom_idx)
        for atom in building_mol.GetAtomWithIdx(composing_atom_idx).GetNeighbors():
            if atom.GetSymbol() == "*":
                removal_dummy_atom_idx = atom.GetIdx()
                break
        else:
            raise Exception("The specified composing atom has no dummy atom")

        # assert (
        #    removal_dummy_atom.GetSymbol() == "*"
        # ), "An atom corresponding to the given index is not a dummy atom."

        # Remove the dummy atom in frag_to_compose
        frag_to_compose_rwmol = Chem.RWMol(frag_to_compose)
        for atom in frag_to_compose_rwmol.GetAtoms():
            if atom.GetSymbol() == "*":
                (
                    composing_frag_brics_type,
                    composing_frag_atom_index,
                ) = cls.remove_dummy_atom(frag_to_compose_rwmol, atom.GetIdx())
                break  # Remind: frag_to_compose has only one dummy atom

        ## Get possible dummy atoms based on the BRICS rule
        # possible_brics_types = list(
        #    map(
        #        lambda x: x[0],
        #        cls.BRICS_type_mapper.getMapping([composing_frag_brics_type]),
        #    )
        # )
        # assert (
        #    removal_dummy_atom_brics_type in possible_brics_types
        # ), "An atom corresponding to the given index cannot be composed following the BRICS rule."

        # Remove a dummy atom in building_mol
        # copied_building_rwmol = Chem.RWMol(copy.deepcopy(building_mol))
        building_rwmol = Chem.RWMol(building_mol)
        (
            brics_type,
            adj_atom_index,
        ) = cls.remove_dummy_atom(  # All the indices of the later remaining atoms decrease by 1.
            building_rwmol, removal_dummy_atom_idx
        )

        # Combine two mol objects
        combined_mol = Chem.CombineMols(building_rwmol, frag_to_compose_rwmol)
        combined_rwmol = Chem.RWMol(combined_mol)

        if brics_type == 7:
            bond_type = Chem.BondType.DOUBLE
        else:
            bond_type = Chem.BondType.SINGLE

        cls.create_bond(
            combined_rwmol,
            adj_atom_index,
            composing_frag_atom_index + building_rwmol.GetNumAtoms(),
            bond_type,
        )
        return combined_rwmol.GetMol()

    @classmethod
    def enumerate_allowed_combinations(
        cls,
        ref_atom_indices: List[int],
        ref_brics_types: List[str],
        frags_adj_dummy_inform: List[List[Tuple[int, int]]],
        frag_idxs,
        logger=print
    ) -> Dict[int, List[List[List[int]]]]:
        """
        Enumerate all possible combinations between remaining atoms and new fragment atoms.
        First, all the combinations are enumerated, and then they are examined to be allowed or not by BRICS rule.
        We note that the indices for fragment atoms are accumulated for batching.

        Parameters
        ----------
        cls : BRICSModule
        ref_atom_indices : List[int]
            Indices of the remaining atoms with broken edges from the original molecule.
        ref_brics_types : List[int]
            BRICS types of the remaining atoms with broken edges from the original molecule.
        frags_adj_dummy_inform : List[SMILES]

        frags_num_atoms

        Returns
        -------
        allowed_combinations : dict
            frag_idx to list of allowed_mappings
            ex) allowed_combinations = {0: [[(3,2), (13,2), (20,4)], [(3,2), (13,4), (20,2)]], 1: [], ...}
            ex) allowed_combinations = {0: [[[3,13,20], [2,2,4]], [[3,13,20], [2,4,2]], 1: [], ...}
        """

        ref_brics_types = list(map(int, ref_brics_types))
        allowed_combinations = dict()
        # accumulate_num_atoms = 0
        # for frag_idx, brics_inform in enumerate(frags_adj_dummy_inform):    # brics_inform: List[Tuple[int,int]]
        for frag_idx, brics_inform in zip(
            frag_idxs, frags_adj_dummy_inform
        ):  # brics_inform: List[Tuple[int,int]]
            # Read fragment atom indices and brics information
            frag_atom_indice = []
            frag_brics_types = []
            for atom_idx, brics_type in brics_inform:
                frag_atom_indice.append(atom_idx)  # List[int]
                frag_brics_types.append(brics_type)  # List[int]

            # Check combinations between leaving parts and inserting fragments
            if len(ref_brics_types) != len(frag_brics_types):   # The number of dummny atoms is different
                logger("Error occured in BRICSModule.enumerate_allowed_combinations:")
                logger("The number of dummy atoms are different.")
                logger(f"frag_idx: {frag_idx}")
                logger(f"brics_inform: {brics_inform}")
                logger(f"ref_brics_types: {ref_brics_types}")
                logger(f"frag_brics_types: {frag_brics_types}")
                return frag_idx
            allowed_mappings = cls.BRICS_type_mapper.checkCombinations(
                ref_brics_types, frag_brics_types
            )
            if len(allowed_mappings) == 0:
                logger("Error occured in BRICSModule.enumerate_allowed_combinations:")
                logger("No allowed_mappings.")
                logger(f"frag_idx: {frag_idx}")
                logger(f"brics_inform: {brics_inform}")
                logger(f"ref_brics_types: {ref_brics_types}")
                logger(f"frag_brics_types: {frag_brics_types}")
                return frag_idx

            # allowed_mappings: [(1,0,2), (1,2,0)]
            allowed_atom_indice_mappings = []
            for mapping in allowed_mappings:
                indice_mapping = [
                    [ref_atom_indices[ref_idx], frag_atom_indice[frag_atom_idx]]
                    for ref_idx, frag_atom_idx in enumerate(mapping)
                ]
                if not indice_mapping in allowed_atom_indice_mappings:
                    allowed_atom_indice_mappings.append(indice_mapping)
            allowed_atom_indice_mappings = (
                torch.tensor(allowed_atom_indice_mappings)
                .long()
                .transpose(1, 2)
                .tolist()
            )

            # Save into allowed_combinations dictionary
            allowed_combinations[frag_idx] = allowed_atom_indice_mappings

        return allowed_combinations

    # @classmethod
    # def enumerate_composing_mols(
    #    cls, building_mol: Mol, frag_to_compose: Mol
    # ) -> List[Mol]:
    #    """
    #    Deprecated.
    #    Compose two molecular fragments to yield a list of new fragments (or possibly complete molecules).
    #    This method is for 'one' step in the building process.

    #    Parameters
    #    ----------
    #    cls : BRICSModule
    #    building_mol : Mol
    #        A molecular fragment in building.
    #        It is helpful to think this fragment as a center of the overall building process.
    #        Thus, it possibly contains more than one fragmented parts (*, in other words).
    #        One important thing is that '*' might appear in anywhere in indexing; different from
    #          a general indexing convention. After using remove_dummy_atom, the indices of the later
    #          atoms are decreased by 1.
    #    frag_to_compose : Mol
    #        A molecular fragment to compose to 'building_mol'.
    #        This is actually an adjacent fragment of the original fragment so it is 'terminal' fragment.
    #        Thus, it contains only one fragmented part and '*' apperas in last in the atom indexing.

    #    Returns
    #    -------
    #    List[Mol]
    #        Resultant molecules of examining all the composing cases.
    #        If more than one cases pass the BRICS rule, returning list would contain more than one
    #        Mol objects.
    #    """
    #    # Read input information
    #    dummy_atoms_in_building_mol = []
    #    for atom in building_mol.GetAtoms():
    #        if atom.GetSymbol() == "*":
    #            dummy_atoms_in_building_mol.append((atom, atom.GetIsotope()))

    #    rwmol_compose_frag = Chem.RWMol(frag_to_compose)
    #    for atom in rwmol_compose_frag.GetAtoms():
    #        if atom.GetSymbol() == "*":
    #            (
    #                composing_frag_brics_type,
    #                composing_frag_atom_index,
    #            ) = cls.remove_dummy_atom(rwmol_compose_frag, atom.GetIdx())
    #            break  # Remind: frag_to_compose has only one dummy atom

    #    # Get possible dummy atoms based on the BRICS rule
    #    possible_brics_types = list(
    #        map(
    #            lambda x: x[0],
    #            cls.BRICS_type_mapper.getMapping([composing_frag_brics_type]),
    #        )
    #    )
    #    dummy_atoms_in_building_mol = (
    #        list(  # dummy atoms which can be composed with composing_frag only remain
    #            filter(
    #                lambda x: x[1] in possible_brics_types, dummy_atoms_in_building_mol
    #            )  # x: [Atom, BRICS_type]
    #        )
    #    )
    #    if len(dummy_atoms_in_building_mol) == 0:  # cannot compose
    #        return None

    #    # Get possible composing results
    #    compose_results = []
    #    for dummy_atom, _ in dummy_atoms_in_building_mol:  # [Atom, BRICS_TYPE]
    #        copied_building_rwmol = Chem.RWMol(copy.deepcopy(building_mol))
    #        (
    #            brics_type,
    #            adj_atom_index,
    #        ) = cls.remove_dummy_atom(  # All the indices of the remaining atoms decrease by 1.
    #            copied_building_rwmol, dummy_atom.GetIdx()
    #        )
    #        combined_mol = Chem.CombineMols(copied_building_rwmol, rwmol_compose_frag)
    #        combined_rwmol = Chem.RWMol(combined_mol)

    #        if brics_type == "7":
    #            bond_type = Chem.BondType.DOUBLE
    #        else:
    #            bond_type = Chem.BondType.SINGLE

    #        cls.create_bond(
    #            combined_rwmol,
    #            adj_atom_index,
    #            composing_frag_atom_index + copied_building_rwmol.GetNumAtoms(),
    #            bond_type,
    #        )
    #        compose_results.append(combined_rwmol.GetMol())

    #    if len(compose_results) == 0:
    #        return None

    #    return compose_results

    @staticmethod
    def get_allowed_subgraph(
        frag_atom_indice: dict,
        edge_index_f: list,
        num_atoms: int,
        max_num_change_atoms: int,
    ) -> Tuple[Tensor, Tensor]:
        subgraph_list = list(frag_atom_indice.keys())
        subgraph_list = [{subgraph} for subgraph in subgraph_list]
        for subgraph in subgraph_list:
            for edge in edge_index_f:
                if subgraph.intersection(edge):
                    extened_subgraph = subgraph.union(edge)
                    if not extened_subgraph in subgraph_list:
                        subgraph_list.append(extened_subgraph)

        allowed_subgraphs_list = []
        for subgraph in subgraph_list:
            subgraph_num_atoms = sum(len(frag_atom_indice[f_idx]) for f_idx in subgraph)
            if subgraph_num_atoms > max_num_change_atoms:
                continue
            if subgraph_num_atoms > num_atoms - subgraph_num_atoms:
                continue
            allowed_subgraphs_list.append(sorted(subgraph))

        allowed_subgraph, allowed_subgraph_idx = [], []
        for subgraph_idx, subgraph in enumerate(allowed_subgraphs_list):
            allowed_subgraph += subgraph
            allowed_subgraph_idx += [subgraph_idx for _ in range(len(subgraph))]
        allowed_subgraph = torch.tensor(allowed_subgraph).long()
        allowed_subgraph_idx = torch.tensor(allowed_subgraph_idx).long()

        return allowed_subgraph, allowed_subgraph_idx

    @staticmethod
    def get_allowed_neg_subgraph(
        frag_atom_indice: dict,
        edge_index_f: list,
        leaving_frag: list,
        num_atoms: int,
        max_num_change_atoms: int,
    ):
        # frag_atom_indice = dict()
        # for i, f_id in enumerate(atom_frag_indice):
        #    try:
        #        frag_atom_indice[f_id].append(i)
        #    except:
        #        frag_atom_indice[f_id] = [i]

        subgraph_list = list(frag_atom_indice.keys())
        subgraph_list = [{subgraph} for subgraph in subgraph_list]
        for subgraph in subgraph_list:
            for edge in edge_index_f:
                if subgraph.intersection(edge):
                    extened_subgraph = subgraph.union(edge)
                    if not extened_subgraph in subgraph_list:
                        subgraph_list.append(extened_subgraph)

        allowed_subgraphs = []
        for subgraph in subgraph_list:
            subgraph_num_atoms = sum(len(frag_atom_indice[f_idx]) for f_idx in subgraph)
            if subgraph_num_atoms > max_num_change_atoms:
                continue
            if subgraph_num_atoms > num_atoms - subgraph_num_atoms:
                continue
            allowed_subgraphs.append(sorted(subgraph))

        leaving_frag.sort()
        assert (
            leaving_frag in allowed_subgraphs
        ), f"leaving_frag: {leaving_frag}\nallowed_subgraphs: {allowed_subgraphs}\n \
                An error occurred in get_allowed_neg_subgraph in BRICSModule."
        allowed_subgraphs.remove(leaving_frag)

        neg_subgraph, neg_subgraph_idx = [], []
        for subgraph_idx, subgraph in enumerate(allowed_subgraphs):
            neg_subgraph += subgraph
            neg_subgraph_idx += [subgraph_idx for _ in range(len(subgraph))]
        neg_subgraph = torch.tensor(neg_subgraph).long()
        neg_subgraph_idx = torch.tensor(neg_subgraph_idx).long()

        return neg_subgraph, neg_subgraph_idx

    # @classmethod
    # def get_combined_mol(cls, mol_list: List) -> Mol:
    #    if len(mol_list) == 1:
    #        return copy.deepcopy(mol_list[0])
    #    else:
    #        combined_mol = mol_list[0]
    #        for mol in mol_list[1:]:
    #            combined_mol = Chem.CombineMols(combined_mol, mol)
    #        return combined_mol


def print_compose_result(mol: Mol):
    print(Chem.MolToSmiles(mol))


def print_enumerate_result(mol_list: List[Mol]):
    for mol in mol_list:
        print(Chem.MolToSmiles(mol))


def debug_compose_mols_with_attachment():
    print("\nTest1")
    original_smi = "NCCCCCNC[C@H]1O[C@@H](OCCc2c[nH]c3ccccc23)[C@H](OCc2ccccc2)[C@@H](OCc2ccccc2)[C@@H]1OCc1ccccc1"
    answer_smi = "NCCCCCOC[C@@H]1O[C@H](OCCc2c[nH]c3ccccc23)[C@H](OCc2ccccc2)[C@H](OCc2ccccc2)[C@H]1OCc1ccccc1"
    leaving_atoms = [6, 7, 8, 9, 10, 23, 32, 41]
    new_frag_smi = "[3*]OC[C@@H]1O[C@H]([13*])[C@H]([15*])[C@H]([15*])[C@H]1[15*]"
    attachment = [[5, 11, 24, 42, 33], [1, 5, 7, 9, 11]]
    query = (original_smi, leaving_atoms, new_frag_smi, attachment)
    gen_smi = BRICSModule.compose_mols_with_attachment(*query, verbose=True)
    print(f"generated_smiles:\n{gen_smi}")
    print(f"answer for Test1:\n{answer_smi}")

    print("\nTest2")
    original_smi = "CC(=O)OC[C@H]1O[C@@H](O[C@H](CC=C(C)C)C2=CC(=O)c3c(O)ccc(O)c3C2=O)[C@H](OC(C)=O)[C@@H](OC(C)=O)[C@@H]1OC(C)=O"
    answer_smi = "CC(=O)OC[C@@H]1O[C@H](O[C@H](CC=C(C)C)C2=CC(=O)c3c(O)ccc(O)c3C2=O)[C@H](OC(C)=O)[C@H]1OC(C)=O"
    leaving_atoms = [5, 6, 7, 29, 30, 31, 32, 33, 34, 39]
    new_frag_smi = "[13*][C@@H]1O[C@H]([13*])[C@H]([15*])[C@H]1[15*]"
    attachment = [[35, 40, 4, 8], [5, 7, 1, 3]]
    query = (original_smi, leaving_atoms, new_frag_smi, attachment)
    gen_smi = BRICSModule.compose_mols_with_attachment(*query, verbose=True)
    print(f"generated_smiles:\n{gen_smi}")
    print(f"answer for Test2:\n{answer_smi}")


def debug_remove_dummy_atom():
    # smi = "OC[C@@H]1O[C@@H]([C@@H]([C@@H]([C@H]1OC)OCc2ccccc2)OCc3ccccc3)OCCc(c[nH]4)c5c4cccc5"
    # mol = Chem.MolFromSmiles(smi)
    # atom = mol.GetAtomWithIdx(4)
    # print(atom.GetChiralTag())

    smi = "[3*]OC[C@@H]1O[C@H]([13*])[C@H]([15*])[C@H]([15*])[C@H]1[15*]"
    mol = Chem.MolFromSmiles(smi)
    rwmol = Chem.RWMol(mol)
    index = 12

    # before removing dummy atom
    atom = mol.GetAtomWithIdx(index - 1)
    print(atom.GetChiralTag())

    # remove dummy atom
    brics_type, adj_atom_index = BRICSModule.remove_dummy_atom(rwmol, index)
    mol = rwmol.GetMol()
    atom = mol.GetAtomWithIdx(adj_atom_index)
    print(atom.GetChiralTag())

    # compose other fragment
    # combined_mol = Chem.CombineMols(building_rwmol, frag_to_compose_rwmol)
    # combined_rwmol = Chem.RWMol(combined_mol)


def test_atom_properties():
    mol = Chem.MolFromSmiles("CCC")
    for atom in mol.GetAtoms():
        atom.SetProp("1", "2")
    rwmol = Chem.RWMol(mol)
    mol = rwmol.GetMol()
    for atom in mol.GetAtoms():
        print(atom.GetPropsAsDict())
    pass


def debug_stereo_of_dummy():
    new_frag_smi = "[3*]OC[C@@H]1O[C@H]([13*])[C@H]([15*])[C@H]([15*])[C@H]1[15*]"
    print(new_frag_smi)
    mol = Chem.MolFromSmiles(new_frag_smi)
    for atom in mol.GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag != ChiralType.CHI_UNSPECIFIED:
            print(chiral_tag)
    new_frag_smi = "[3*]OC[C@@H]1O[C@H]([C@H]([15*])[C@H]([15*])[C@H]1[15*])[13*]"
    print(new_frag_smi)
    mol = Chem.MolFromSmiles(new_frag_smi)
    for atom in mol.GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag != ChiralType.CHI_UNSPECIFIED:
            print(chiral_tag)
    gen = "NCCCCCOC[C@@H]1O[C@@H](OCCc2c[nH]c3ccccc23)[C@@H](OCc2ccccc2)[C@@H](OCc2ccccc2)[C@@H]1OCc1ccccc1"
    print(gen)
    mol = Chem.MolFromSmiles(gen)
    for atom in mol.GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag != ChiralType.CHI_UNSPECIFIED:
            print(chiral_tag)
    answer = "NCCCCCOC[C@@H]1O[C@H](OCCc2c[nH]c3ccccc23)[C@H](OCc2ccccc2)[C@H](OCc2ccccc2)[C@H]1OCc1ccccc1"
    print(answer)
    mol = Chem.MolFromSmiles(answer)
    for atom in mol.GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag != ChiralType.CHI_UNSPECIFIED:
            print(chiral_tag)
    #    smi = "NCCCCCNC[C@H]1O[C@@H](OCCc2c[nH]c3ccccc23)[C@H](OCc2ccccc2)[C@@H](OCc2ccccc2)[C@@H]1OCc1ccccc1"
    #    print(smi)
    #    mol = Chem.MolFromSmiles(smi)
    #    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    #    #print(brics_bonds)
    #    brics_bonds = [brics_bonds[4], brics_bonds[8], brics_bonds[6], brics_bonds[7], brics_bonds[5]]
    #
    #    bond_indice_to_break, bond_types_to_break = [], []
    #    for brics_bond in brics_bonds:
    #        bond_idx, bond_type = brics_bond
    #        bond_indice_to_break.append(mol.GetBondBetweenAtoms(*bond_idx).GetIdx())
    #        bond_types_to_break.append(tuple(map(int, bond_type)))
    #
    #    for atom in mol.GetAtoms():
    #        chiral_tag = atom.GetChiralTag()
    #        if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
    #            print(chiral_tag)
    #
    # print("\nBreakBRICSBonds")
    # smi = "NCCCCCNC[C@H]1O[C@@H](OCCc2c[nH]c3ccccc23)[C@H](OCc2ccccc2)[C@@H](OCc2ccccc2)[C@@H]1OCc1ccccc1"
    # mol = Chem.MolFromSmiles(smi)
    # broken_mol = BRICS.BreakBRICSBonds(mol, brics_bonds)
    # mol = Chem.GetMolFrags(broken_mol, asMols=True)[1]
    # for atom in mol.GetAtoms():
    #    chiral_tag = atom.GetChiralTag()
    #    if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
    #        print(chiral_tag)
    # print("Re-Generated Mol")
    # for atom in Chem.MolFromSmiles(Chem.MolToSmiles(mol)).GetAtoms():
    #    chiral_tag = atom.GetChiralTag()
    #    if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
    #        print(chiral_tag)
    # print(Chem.MolToSmiles(mol))

    # print("\nFragmentOnBonds")
    # smi = "NCCCCCNC[C@H]1O[C@@H](OCCc2c[nH]c3ccccc23)[C@H](OCc2ccccc2)[C@@H](OCc2ccccc2)[C@@H]1OCc1ccccc1"
    # mol = Chem.MolFromSmiles(smi)
    # broken_mol = Chem.FragmentOnBonds(
    #    mol, bondIndices=bond_indice_to_break, dummyLabels=bond_types_to_break
    # )
    # mol = Chem.GetMolFrags(broken_mol, asMols=True)[1]
    # for atom in mol.GetAtoms():
    #    chiral_tag = atom.GetChiralTag()
    #    if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
    #        print(chiral_tag)
    # print("Re-Generated Mol")
    # for atom in Chem.MolFromSmiles(Chem.MolToSmiles(mol)).GetAtoms():
    #    chiral_tag = atom.GetChiralTag()
    #    if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
    #        print(chiral_tag)
    # print(Chem.MolToSmiles(mol))

    return


def debug_break_brics_bonds():
    # Find BRICS bonds
    smi = "CN[C@@H]1OCCCC1"
    print(f"Test molecule1: {smi}")
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            print(chiral_tag)

    brics_bond = list(BRICS.FindBRICSBonds(mol))[0]
    print(brics_bond)
    bond_idx, bond_type = brics_bond
    bond_indice_to_break = [mol.GetBondBetweenAtoms(*bond_idx).GetIdx()]
    bond_types_to_break = [tuple(map(int, reversed(list(bond_type))))]

    # BreakBRICSBonds
    print("\nBreakBRICSBonds")
    mol = Chem.MolFromSmiles(smi)
    broken_mol = BRICS.BreakBRICSBonds(mol, [brics_bond])
    for atom in broken_mol.GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            print(chiral_tag)

    broken_mol_list = Chem.GetMolFrags(broken_mol, asMols=True)
    print([Chem.MolToSmiles(_) for _ in broken_mol_list])
    # for atom in Chem.MolFromSmiles(Chem.MolToSmiles(broken_mol_list[1])).GetAtoms():
    print("HERE")
    for atom in Chem.MolFromSmiles("[13*][C@@H]1OCCCC1").GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            print(chiral_tag)

    # FragmentOnBonds
    print("\nFragmentOnBonds")
    mol = Chem.MolFromSmiles(smi)
    broken_mol = Chem.FragmentOnBonds(
        mol, bondIndices=bond_indice_to_break, dummyLabels=bond_types_to_break
    )
    for atom in broken_mol.GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            print(chiral_tag)

    broken_mol_list = Chem.GetMolFrags(broken_mol, asMols=True)
    print([Chem.MolToSmiles(_) for _ in broken_mol_list])
    for atom in Chem.MolFromSmiles(Chem.MolToSmiles(broken_mol_list[1])).GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            print(chiral_tag)

    # Find BRICS bonds
    smi = "C1CCCO[C@H]1NC"
    print(f"\n\nTest molecule2: {smi}")
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            print(chiral_tag)

    brics_bond = list(BRICS.FindBRICSBonds(mol))[0]
    bond_idx, bond_type = brics_bond
    brics_bond = [tuple(reversed(list(bond_idx))), tuple(reversed(list(bond_type)))]
    print(brics_bond)
    bond_indice_to_break = [mol.GetBondBetweenAtoms(*bond_idx).GetIdx()]
    bond_types_to_break = [tuple(map(int, bond_type))]

    # BreakBRICSBonds
    print("\nBreakBRICSBonds")
    mol = Chem.MolFromSmiles(smi)
    broken_mol = BRICS.BreakBRICSBonds(mol, [brics_bond])
    for atom in broken_mol.GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            print(chiral_tag)

    broken_mol_list = Chem.GetMolFrags(broken_mol, asMols=True)
    print([Chem.MolToSmiles(_) for _ in broken_mol_list])
    for atom in Chem.MolFromSmiles(Chem.MolToSmiles(broken_mol_list[0])).GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            print(chiral_tag)

    # FragmentOnBonds
    print("\nFragmentOnBonds")
    mol = Chem.MolFromSmiles(smi)
    broken_mol = Chem.FragmentOnBonds(
        mol, bondIndices=bond_indice_to_break, dummyLabels=bond_types_to_break
    )
    for atom in broken_mol.GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            print(chiral_tag)

    broken_mol_list = Chem.GetMolFrags(broken_mol, asMols=True)
    print([Chem.MolToSmiles(_) for _ in broken_mol_list])
    for atom in Chem.MolFromSmiles(Chem.MolToSmiles(broken_mol_list[0])).GetAtoms():
        chiral_tag = atom.GetChiralTag()
        if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            print(chiral_tag)

    return True


if __name__ == "__main__":
    # test_atom_properties()
    # debug_remove_dummy_atom()
    debug_compose_mols_with_attachment()
    # debug_stereo_of_dummy()
    # debug_break_brics_bonds()
