"""
Protein Preparation Module
Converts PDB protein files to PDBQT format for molecular docking.
Uses Meeko's mk_prepare_protein command for AutoDock preparation.
"""

import subprocess
import sys
import yaml
from pathlib import Path
from typing import Optional

def prepare_protein(pdb_path: str, output_path: str) -> str:
    """
    Convert a .pdb protein file to .pdbqt format using Meeko's mk_prepare_protein.
    
    Args:
        pdb_path: Path to the input .pdb protein file
        output_path: Path where the .pdbqt file will be saved
        
    Returns:
        Path to the generated .pdbqt file
        
    Raises:
        FileNotFoundError: If the .pdb file does not exist or mk_prepare_protein not found
        subprocess.CalledProcessError: If Meeko command fails
    """
    pdb_path = Path(pdb_path)
    output_path = Path(output_path)

    # remove .pdbqt extension if present to avoid double extensions
    if output_path.suffix == ".pdbqt":
        output_path = output_path.with_suffix("")
    
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Construct the path to mk_prepare_protein relative to the current Python interpreter
    env_root = Path(sys.executable).parent
    mk_prepare_protein = env_root / "Scripts" / "mk_prepare_receptor.exe"
    
    # Fall back to just the command name if the exe isn't found
    if not mk_prepare_protein.exists():
        mk_prepare_protein = Path("mk_prepare_receptor")
    
    # Try different preparation strategies
    strategies = [
        # Strategy 1: Default
        {
            "name": "Default",
            "args": []
        },
        # Strategy 2: A
        {
            "name": "Add altlock",
            "args": [ "--default_altloc", "A"]
        },
        # Strategy 3: Minimal
        {
            "name": "Add -a",
            "args": ["--default_altloc", "A", "-a"]
        },
    ]
    
    last_error = None
    for strategy in strategies:
        try:
            cmd = [str(mk_prepare_protein), "-i", str(pdb_path), "-o", str(output_path)] + strategy["args"]
            print(f"Trying {strategy['name']}...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ Success with {strategy['name']}")
            return str(output_path)
        except subprocess.CalledProcessError as e:
            last_error = e
            print(f"✗ Failed with {strategy['name']}: {e.stderr if e.stderr else 'exit code ' + str(e.returncode)}")
            continue
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not find 'mk_prepare_receptor'. "
                f"Ensure Meeko is installed in the active environment."
            ) from e
    
    # If all strategies failed, raise the last error
    if last_error:
        raise subprocess.CalledProcessError(
            last_error.returncode,
            last_error.cmd,
            output=last_error.stdout,
            stderr=f"All preparation strategies failed:\n{last_error.stderr}"
        )


def write_config_yaml(receptor_path: str, yaml_output_path: str) -> None:
    """
    Write a boilerplate YAML config file for molecular docking.

    Args:
        receptor_path: Path to the prepared receptor (PDBQT) file
        yaml_output_path: Path where the config YAML will be written

    The generated config includes:
    - receptor: Path to the prepared receptor
    - center: Search space center (default [0, 0, 0])
    - size: Search space size (default [20, 20, 20])
    """
    config = {
        'receptor': receptor_path,
        'center': [0, 0, 0],  # Default center - user should adjust based on protein
        'size': [20, 20, 20],  # Default size 20Å box - user may adjust
    }

    with open(yaml_output_path, 'w') as f:
        # Add a comment at the top
        f.write("# Docking configuration\n")
        f.write("# Update 'center' coordinates based on your target binding site\n")
        f.write("# You can use PyMOL, Chimera, or other tools to identify these\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main():
    """Interactive protein preparation"""
    pdb_file = input("Enter the path to the protein PDB file: ")
    output_dir = "data/prepared_proteins"
    pdb_stem = Path(pdb_file).stem

    try:
        pdbqt_path = prepare_protein(pdb_file, output_path=Path(output_dir) / f"{pdb_stem}")
        print(f"Protein prepared successfully: {pdbqt_path}")
        
        # Generate docking config file
        config_path = Path("configs") / f"{pdb_stem}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        write_config_yaml(pdbqt_path, str(config_path))
        print(f"Config file created: {config_path}")
    except Exception as e:
        print(f"Error preparing protein: {e}")


if __name__ == "__main__":
    main()
