"""
AutoDock Binding Calculator
Calculates binding affinity between a ligand (SMILES) and a protein (PDB).
Uses Meeko for ligand preparation and AutoDock Vina for docking.
"""

import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple
from multiprocessing import Pool, cpu_count, freeze_support
import logging

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.ligand_preparation import prepare_ligand

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _dock_ligand_worker(args: Tuple) -> Tuple[str, Dict]:
    """
    Worker function for multiprocessing pool.
    Docks a single ligand and returns (name, results).
    
    Args:
        args: Tuple of (calculator, ligand_name, ligand_path)
        
    Returns:
        Tuple of (ligand_name, result_dict)
    """
    calculator, ligand_name, ligand_path = args
    result = calculator.calculate_binding(ligand_path, name=ligand_name)
    return ligand_name, result


class VinaCalculator:
    """Calculate protein-ligand binding affinity using AutoDock Vina."""
    
    def __init__(self, protein_pdbqt: str, center: Tuple[float, float, float], 
                 size: Tuple[float, float, float] = (20, 20, 20),
                 cpus: Optional[int] = 0,
                 exhaustiveness: Optional[int] = 8, max_evals: Optional[int] = 0,
                 random_seed: Optional[int] = 42, num_modes: Optional[int] = 9,
                 min_rmsd: Optional[float] = 1.0, energy_range: Optional[float] = 3.0,
                 spacing: Optional[float] = 0.375, verbosity: Optional[int] = 1):
        """
        Initialize the AutoDock calculator.
        
        Args:
            protein_pdbqt: Path to protein PDBQT file
            center: Docking center coordinates (x, y, z)
            size: Docking box size (x, y, z) in Angstroms
            exhaustiveness: Exhaustiveness parameter for Vina (higher = more thorough)
            max_evals: Maximum number of evaluations for Vina
            num_modes: Number of binding modes to generate
            energy_range: Range of energies to consider for binding modes
        """
        self.protein_pdbqt = str(protein_pdbqt)
        self.protein_name = Path(protein_pdbqt).stem
        self.center = center
        self.size = size
        self.cpus = cpus
        self.exhaustiveness = exhaustiveness
        self.max_evals = max_evals
        self.random_seed = random_seed
        self.num_modes = num_modes
        self.min_rmsd = min_rmsd
        self.energy_range = energy_range
        self.spacing = spacing
        self.verbosity = verbosity

        if not Path(self.protein_pdbqt).exists():
            raise FileNotFoundError(f"Protein PDBQT not found: {self.protein_pdbqt}")
        
        logger.info(f"Initialized with protein: {self.protein_pdbqt}")
    
    def run_docking(self, ligand_pdbqt: str, output_pdbqt: str) -> Dict:
        """
        Run Vina docking using command-line interface.
        
        Args:
            ligand_pdbqt: Path to ligand PDBQT file
            output_pdbqt: Path where output PDBQT will be saved
            
        Returns:
            Dictionary with docking results including affinity and success status
        """
        # Build vina command
        cmd = [
            self._get_vina_executable(),
            "--receptor", self.protein_pdbqt,
            "--ligand", ligand_pdbqt,
            "--center_x", str(self.center[0]),
            "--center_y", str(self.center[1]),
            "--center_z", str(self.center[2]),
            "--size_x", str(self.size[0]),
            "--size_y", str(self.size[1]),
            "--size_z", str(self.size[2]),
            "--exhaustiveness", str(self.exhaustiveness),
            "--max_evals", str(self.max_evals),
            "--seed", str(self.random_seed),
            "--num_modes", str(self.num_modes),
            "--min_rmsd", str(self.min_rmsd),
            "--energy_range", str(self.energy_range),
            "--spacing", str(self.spacing),
            "--verbosity", str(self.verbosity),
            "--out", output_pdbqt,
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Vina docking completed successfully")
            return {
                "success": True,
                "output_pdbqt": output_pdbqt,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Vina docking failed: {e.stderr}")
            return {
                "success": False,
                "output_pdbqt": None,
                "error": e.stderr,
                "stdout": e.stdout
            }
    
    def calculate_binding(self, 
                          ligand_pdbqt_path: str, 
                          name: str = "ligand",
                          save_pdbqt: bool = False) -> Dict:
        """
        Calculate binding affinity for a ligand PDBQT file.
        
        Args:
            ligand_pdbqt_path: Path to the ligand PDBQT file
            name: Name for the ligand
            save_pdbqt: Whether to save the output PDBQT file to disk
            
        Returns:
            Dictionary containing docking results with keys:
                - success: Boolean indicating success
                - name: Ligand name
                - affinity: Best binding affinity (kcal/mol)
                - affinity_range: Tuple of (worst, best) affinities
                - output_pdbqt: Path to output PDBQT file (only if save_pdbqt=True)
                - message: Status message
        """
        ligand_pdbqt_path = Path(ligand_pdbqt_path)
        
        if not ligand_pdbqt_path.exists():
            return {
                "success": False,
                "name": name,
                "affinity": None,
                "message": f"Ligand PDBQT not found: {ligand_pdbqt_path}"
            }
        
        # Create output path
        output_pdbqt = ligand_pdbqt_path.parent / f"{ligand_pdbqt_path.stem}_{self.protein_name}.pdbqt"
        
        try:
            # Run docking
            dock_result = self.run_docking(str(ligand_pdbqt_path), str(output_pdbqt))
            
            if not dock_result["success"]:
                return {
                    "success": False,
                    "name": name,
                    "affinity": None,
                    "message": f"Docking failed: {dock_result.get('error', 'Unknown error')}"
                }
            
            # Parse results from output PDBQT
            affinity = self._extract_top_affinity(str(output_pdbqt))
            all_affinities = self._extract_all_affinities(str(output_pdbqt))
            
            if affinity is None:
                return {
                    "success": False,
                    "name": name,
                    "affinity": None,
                    "message": "Could not parse affinity from output"
                }
            
            affinity_range = (all_affinities[-1], all_affinities[0]) if len(all_affinities) > 1 else (affinity, affinity)
            
            results = {
                "success": True,
                "name": name,
                "affinity": affinity,
                "affinity_range": affinity_range,
                "modes": all_affinities,
                "output_pdbqt": str(output_pdbqt),
                "message": f"Docking completed. Best affinity: {affinity:.2f} kcal/mol"
            }
            
            logger.info(f"Binding affinity for {name}: {affinity:.2f} kcal/mol")

            if not save_pdbqt:
                # Remove PDBQT from results and delete file from disk if not saving
                results.pop("output_pdbqt", None)
                output_pdbqt_path = Path(output_pdbqt)
                output_pdbqt_path.unlink(missing_ok=True)

            return results
            
        except Exception as e:
            logger.error(f"Error calculating binding for {name}: {e}")
            return {
                "success": False,
                "name": name,
                "affinity": None,
                "message": str(e)
            }
    
    def _get_vina_executable(self) -> str:
        """
        Get the path to the Vina executable.
        On Windows: looks for vina.exe relative to repo root.
        On Unix: uses 'vina' command from PATH.
        """
        if platform.system().lower().startswith("win"):
            # Windows: look for vina.exe in repo root
            repo_root = Path(__file__).resolve().parents[1]
            vina_exe = repo_root / "vina.exe"
            if vina_exe.exists():
                return str(vina_exe)
            raise FileNotFoundError(f"vina.exe not found at {vina_exe}")
        else:
            # Unix: use 'vina' from PATH
            return "vina"
    
    def _extract_top_affinity(self, pdbqt_path: str) -> Optional[float]:
        """Extract the best (top) binding affinity from Vina output PDBQT."""
        try:
            content = Path(pdbqt_path).read_text()
            match = re.search(r'REMARK VINA RESULT:\s+([-\d.]+)', content)
            if match:
                return float(match.group(1))
        except Exception as e:
            logger.warning(f"Could not extract affinity from {pdbqt_path}: {e}")
        return None
    
    def _extract_all_affinities(self, pdbqt_path: str) -> list:
        """Extract all binding affinities from Vina output PDBQT (sorted best-first)."""
        try:
            content = Path(pdbqt_path).read_text()
            affinities = []
            for match in re.finditer(r'REMARK VINA RESULT:\s+([-\d.]+)', content):
                affinities.append(float(match.group(1)))
            return affinities
        except Exception as e:
            logger.warning(f"Could not extract affinities from {pdbqt_path}: {e}")
        return []
    
    def dock_multiple(self, mol_paths: Dict[str, str], use_multiprocessing: bool = True, 
                      num_processes: Optional[int] = None) -> Dict:
        """
        Dock multiple ligands sequentially or in parallel.
        
        Args:
            mol_paths: Dictionary with format {name: path_to_pdbqt_file}
            use_multiprocessing: If True, use multiprocessing pool for parallel docking
            num_processes: Number of processes for pool (default: CPU count)
            
        Returns:
            Dictionary with results for each ligand: {name: result_dict}
        """
        if not use_multiprocessing:
            # Sequential docking
            results = {}
            for name, path in mol_paths.items():
                results[name] = self.calculate_binding(path, name)
            return results
        
        # Parallel docking with multiprocessing
        if num_processes is None:
            num_processes = cpu_count()
        
        # Prepare worker arguments
        worker_args = [
            (self, name, path)
            for name, path in mol_paths.items()
        ]
        
        logger.info(f"Docking {len(mol_paths)} ligands with {num_processes} processes")
        
        results = {}
        with Pool(processes=num_processes) as pool:
            for name, result in pool.imap_unordered(_dock_ligand_worker, worker_args):
                results[name] = result
                if result["success"]:
                    logger.info(f"  {name}: {result['affinity']:.2f} kcal/mol")
                else:
                    logger.warning(f"  {name}: Failed - {result['message']}")
        
        return results


def main():
    """VinaCalculator usage to calculate binding affinity for a single ligand and protein."""

    
    protein_config_path = input("Enter path to protein config YAML: ")
    ligand = input("Enter path to ligand PDBQT or SMILES string of ligand: ")

    vina_config = load_config("configs/vina_binding.yaml")["simulation_parameters"]
    protein_config = load_config(protein_config_path)

    remove_pdbqt = False # Flag to track if we need to clean up a prepared PDBQT file after docking

    if ligand.endswith(".pdbqt"):
        ligand_path = ligand
    
    else:        # Assume it's a SMILES string and prepare ligand
        ligand_path = prepare_ligand(ligand, output_dir="prepared_ligands")
        remove_pdbqt = True  # Mark for cleanup after docking
        if ligand_path is None:
            print("Failed to prepare ligand. Enter a valid PDBQT file or SMILES string.")
            return
    
    calculator = VinaCalculator(
        protein_pdbqt=protein_config["receptor"],
        center=tuple(protein_config["center"]),
        size=tuple(protein_config["size"]),
        cpus=vina_config.get("cpu", 0),
        exhaustiveness=vina_config.get("exhaustiveness", 8),
        max_evals=vina_config.get("max_evals", 0),
        random_seed=vina_config.get("seed", 42),
        num_modes=vina_config.get("num_modes", 9),
        min_rmsd=vina_config.get("min_rmsd", 1.0),
        energy_range=vina_config.get("energy_range", 3.0),
        spacing=vina_config.get("spacing", 0.375),
        verbosity=vina_config.get("verbosity", 1)
    )

    results = calculator.calculate_binding(ligand_path, name=Path(ligand_path).stem)

    print(results['message'])
    print(f"Output saveed to: {results.get('output_pdbqt', 'N/A')}")

    # Remove prepared PDBQT file if it was created from SMILES input
    if remove_pdbqt:
        Path.unlink(ligand_path)  # Clean up prepared ligand file after docking


if __name__ == "__main__":
    freeze_support()
    main()
