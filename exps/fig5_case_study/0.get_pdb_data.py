from pymol import cmd
from rdkit import Chem
from rdkit.Chem import AllChem

cmd.fetch("8h3g", type="pdb1")
cmd.create("receptor", "chain B and 8h3g and polymer")
cmd.create("ligand", "chain B and 8h3g and resn 7YY")
cmd.remove("not alt ''+A") # remove alternatives
cmd.alter("all", 'alt=""')
cmd.save("8h3g_protein.pdb", "receptor")
cmd.save("_8h3g_ligand.sdf", "ligand")
cmd.save("8h3g_ligand.pdb", "ligand")

ref_mol = Chem.MolFromSmiles("Cn1cnc(Cn2c(=O)nc(Nc3cc4cn(C)nc4cc3Cl)n(Cc3cc(F)c(F)cc3F)c2=O)n1")
mol = Chem.MolFromPDBFile("8h3g_ligand.pdb")
mol = AllChem.AssignBondOrdersFromTemplate(ref_mol, mol)
with Chem.SDWriter("8h3g_ligand.sdf") as w:
    w.write(mol)

