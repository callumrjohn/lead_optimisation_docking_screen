"""
Ligand Preparation Module
Converts SMILES strings to PDBQT files for molecular docking.
Uses RDKit for structure generation and Meeko for AutoDock preparation.
"""

import shutil
import subprocess
import sys
import platform
import os
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

# Try to import Meeko Python API for Linux/macOS
if platform.system() != "Windows":
    try:
        from meeko import MoleculePreparation
        MEEKO_API_AVAILABLE = True
    except ImportError:
        MEEKO_API_AVAILABLE = False
else:
    MEEKO_API_AVAILABLE = False


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
    Uses mk_prepare_ligand executable on Windows, Python API on Linux/macOS.
    
    Args:
        mol_path: Path to the input .mol file
        output_path: Path where the .pdbqt file will be saved
        
    Returns:
        Path to the generated .pdbqt file
    """
    mol_path = Path(mol_path)
    output_path = Path(output_path)
    
    if not mol_path.exists():
        raise FileNotFoundError(f"Mol file not found: {mol_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if platform.system() == "Windows":
        # Windows: use mk_prepare_ligand executable
        env_root = Path(sys.executable).parent
        mk_prepare_ligand = env_root / "Scripts" / "mk_prepare_ligand.exe"
        
        if not mk_prepare_ligand.exists():
            mk_prepare_ligand = "mk_prepare_ligand"
        
        cmd = [
            str(mk_prepare_ligand),
            "-i", str(mol_path),
            "-o", str(output_path),
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=os.environ.copy())
            return str(output_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not find 'mk_prepare_ligand'. "
                f"Install Meeko: pip install meeko"
            ) from e
        except subprocess.CalledProcessError as e:
            raise subprocess.CalledProcessError(
                e.returncode,
                e.cmd,
                output=e.stdout,
                stderr=f"Meeko preparation failed:\n{e.stderr}"
            )
    
    else:
        # Linux/macOS: use Meeko Python API
        if not MEEKO_API_AVAILABLE:
            raise ImportError("Meeko Python API not available. Install with: pip install meeko")
        
        try:
            mol = Chem.MolFromMolFile(str(mol_path), removeHs=False)
            if mol is None:
                raise ValueError(f"Could not read molecule from {mol_path}")
            
            preparator = MoleculePreparation()
            preparator.prepare(mol)
            
            with open(output_path, 'w') as f:
                f.write(preparator.write_pdbqt_string())
            
            return str(output_path)
        
        except Exception as e:
            raise Exception(f"Meeko preparation failed: {e}") from e

def prepare_ligand(smiles: str, name: str, output_dir: str = "prepared_ligands") -> str:
    """
    Prepare a ligand for docking by converting a SMILES string to .pdbqt format.
    
    Args:
        smiles: SMILES string of the ligand
        name: Name for the ligand (used in output file naming)
        output_dir: Directory where the prepared .pdbqt file will be saved
    Returns:
        Path to the prepared .pdbqt file
    """
    mol_path = smiles_to_mol(smiles, output_path=Path(output_dir) / f"{name}.mol")
    pdbqt_path = mol_to_pdbqt(mol_path, output_path=Path(output_dir) / f"{name}.pdbqt")

    # Delete the intermediate .mol file
    Path(mol_path).unlink(missing_ok=True)
    
    return pdbqt_path

def main():

    smiles = input("Enter the SMILES string of the ligand: ")
    name = input("Enter the name for the ligand: ")
    output_dir = "data/prepared_ligands"

    try:
        pdbqt_path = prepare_ligand(smiles, name, output_dir=output_dir)
        print(f"Ligand prepared successfully: {pdbqt_path}")
    except Exception as e:
        print(f"Error preparing ligand: {e}")

if __name__ == "__main__":
    main()