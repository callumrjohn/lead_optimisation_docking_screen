"""
Ligand Preparation Module
Converts SMILES strings to PDBQT files for molecular docking.
Uses RDKit for structure generation and Meeko for AutoDock preparation.
"""

import subprocess
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_mol(smiles: str, output_path: Optional[str] = None) -> str:
    """
    Convert a SMILES string to a 3D .mol file.
    
    Args:
        smiles: SMILES string representation of the molecule
        output_path: Output path for the .mol file. Defaults to 'ligand.mol'.
        
    Returns:
        Path to the generated .mol file
        
    Raises:
        ValueError: If SMILES string is invalid
    """
    if output_path is None:
        output_path = "ligand.mol"
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = Chem.SDWriter(str(output_path))
    writer.write(mol)
    writer.close()
    
    return str(output_path)


def mol_to_pdbqt(mol_path: str, output_path: str) -> str:
    """
    Convert a .mol file to .pdbqt format using Meeko.
    
    Args:
        mol_path: Path to the input .mol file
        output_path: Path where the .pdbqt file will be saved
        
    Returns:
        Path to the generated .pdbqt file
        
    Raises:
        FileNotFoundError: If the .mol file does not exist
        subprocess.CalledProcessError: If Meeko command fails
    """
    mol_path = Path(mol_path)
    output_path = Path(output_path)
    
    if not mol_path.exists():
        raise FileNotFoundError(f"Mol file not found: {mol_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "mk_prepare_ligand",
        "-i", str(mol_path),
        "-o", str(output_path),
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise subprocess.CalledProcessError(
            e.returncode,
            e.cmd,
            output=e.stdout,
            stderr=f"Meeko preparation failed:\n{e.stderr}"
        )
    
    return str(output_path)

def prepare_ligand(smiles: str, output_dir: str = "prepared_ligands") -> str:
    """
    Prepare a ligand for docking by converting a SMILES string to .pdbqt format.
    
    Args:
        smiles: SMILES string of the ligand
        output_dir: Directory where the prepared .pdbqt file will be saved
    Returns:
        Path to the prepared .pdbqt file
    """
    mol_path = smiles_to_mol(smiles, output_path=Path(output_dir) / "ligand.mol")
    pdbqt_path = mol_to_pdbqt(mol_path, output_path=Path(output_dir) / "ligand.pdbqt")

    # Delete the intermediate .mol file
    Path(mol_path).unlink(missing_ok=True)
    
    return pdbqt_path
