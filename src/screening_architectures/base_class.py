"""
Base class and helper functions for screening strategies.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters import MaxMinPicker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ligand_preparation import prepare_ligand
from src.vina_binding import VinaCalculator
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fps_from_smiles(smiles_series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ECFP fingerprints from SMILES strings using RDKit rdFingerprintGenerator batch API.
    
    Args:
        smiles_series: Pandas Series of SMILES strings
        
    Returns:
        Tuple of (smiles_array, fingerprint_array)
    """
    smiles_array = smiles_series.values
    
    # Convert SMILES to molecules
    mols = []
    for smiles in smiles_array:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mols.append(mol)
        else:
            mols.append(None)
    
    # Create Morgan fingerprint generator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    # Generate fingerprints in batch
    fps_list = generator.GetFingerprints(mols, numThreads=1)
    
    # Convert ExplicitBitVect objects to numpy arrays (binary vectors)
    fingerprints = []
    for fp in fps_list:
        if fp is not None:
            # Convert ExplicitBitVect to numpy binary array
            fp_array = np.array(fp, dtype=np.float32)
            fingerprints.append(fp_array)
        else:
            fingerprints.append(np.zeros(2048, dtype=np.float32))
    
    return smiles_array, np.array(fingerprints)


def maxmin_sampler(smiles: List[str], sample_size: int) -> np.ndarray:
    """
    Sample a subset of SMILES strings based on Tanimoto similarity using MaxMinPicker.
    
    Args:
        smiles: List or pd.Series of SMILES strings to sample from
        sample_size: Number of SMILES to sample
        
    Returns:
        Array of sampled SMILES strings
    """
    smiles_array, fp_array = fps_from_smiles(pd.Series(smiles))
    
    n = len(fp_array)
    if n == 0 or sample_size <= 0:
        return np.array([], dtype=object)
    if n == 1:
        return smiles_array[:1]
    
    # If sample size exceeds available compounds, return all
    if sample_size >= n:
        return smiles_array
    
    sample_size = min(sample_size, n)
    
    # Convert to RDKit bit vectors
    rd_fps = [DataStructs.ExplicitBitVect(fp_array.shape[1]) for _ in range(n)]
    for i, fp in enumerate(fp_array):
        on_bits = np.flatnonzero(fp)
        for b in on_bits:
            rd_fps[i].SetBit(int(b))
    
    distances = []
    for i in range(1, n):
        distances.extend(
            DataStructs.BulkTanimotoSimilarity(
                rd_fps[i], rd_fps[:i], returnDistance=True
            )
        )
    
    mmp = MaxMinPicker()
    sample_indices = mmp.Pick(np.array(distances, dtype=np.float64), n, sample_size)
    return smiles_array[np.array(sample_indices, dtype=int)]


def _evaluate_ligand_worker(smiles: str, ligand_name: str, target_calc: VinaCalculator, offtarget_calcs: List[VinaCalculator], target_weight: float, selectivity_weight: float, iteration_index: int = 0, save_pdbqt: bool = False) -> Dict:
    """
    Worker function for parallel ligand evaluation.
    Static function to enable multiprocessing serialization.
    
    Args:
        smiles: SMILES string
        ligand_name: Ligand name
        target_calc: Target VinaCalculator
        offtarget_calcs: List of off-target VinaCalculators
        target_weight: Weight for target in composite score
        selectivity_weight: Weight for selectivity in composite score
        iteration_index: Iteration index (for BO tracking)
        save_pdbqt: Whether to save PDBQT files
        
    Returns:
        Results dictionary
    """
    try:
        logger.info(f"Evaluating {ligand_name}...")
        
        # Prepare ligand
        pdbqt_path = prepare_ligand(smiles, name=ligand_name, output_dir="data/prepared_ligands")
        
        # Target docking
        target_result = target_calc.calculate_binding(pdbqt_path, name=ligand_name, save_pdbqt=save_pdbqt)
        if not target_result["success"]:
            logger.warning(f"Target binding failed for {ligand_name}")
            return {"success": False}
        
        target_affinity = target_result["affinity"]
        
        # Off-target docking
        offtarget_affinities = []
        for i, calc in enumerate(offtarget_calcs):
            result = calc.calculate_binding(pdbqt_path, name=f"{ligand_name}_offtarget_{i}")
            if result["success"] and result["affinity"] is not None:
                offtarget_affinities.append(result["affinity"])
        
        if len(offtarget_affinities) == 0:
            logger.warning(f"No off-target results for {ligand_name}")
            return {"success": False}
        
        # Calculate scores
        mean_offtarget = np.mean(offtarget_affinities)
        selectivity = mean_offtarget - target_affinity
        composite_score = target_weight * (-target_affinity) + selectivity_weight * selectivity
        
        logger.info(
            f"  Target affinity: {target_affinity:.2f} kcal/mol | "
            f"Selectivity: {selectivity:.2f} | "
            f"Composite: {composite_score:.2f}"
        )
        
        return {
            "success": True,
            "smiles": smiles,
            "target_affinity": target_affinity,
            "selectivity": selectivity,
            "composite_score": composite_score,
            "iteration_index": iteration_index,
        }
    
    except Exception as e:
        logger.error(f"Error evaluating {ligand_name}: {e}")
        return {"success": False}


class Screener:
    """Base class for ligand screening strategies"""
    
    def __init__(
        self,
        protein_configs: List[str],
        vina_config: str = "configs/vina_binding.yaml",
        target_weight: float = 0.5,
        selectivity_weight: float = 0.5,
    ):
        """
        Initialize screener base class.
        
        Args:
            protein_configs: List of YAML config paths. First is target, rest are off-targets.
            vina_config: Path to Vina parameters YAML
            target_weight: Weight for target binding in composite objective
            selectivity_weight: Weight for selectivity in composite objective
        """
        if len(protein_configs) < 2:
            raise ValueError("Need at least 2 proteins (1 target + 1 off-target)")
        
        self.protein_configs = protein_configs
        self.target_config = load_config(protein_configs[0])
        self.offtarget_configs = [load_config(cfg) for cfg in protein_configs[1:]]
        self.vina_config = load_config(vina_config)["simulation_parameters"]
        
        self.target_weight = target_weight
        self.selectivity_weight = selectivity_weight
        
        # Initialize calculators
        self.target_calc = self._init_calculator(self.target_config)
        self.offtarget_calcs = [self._init_calculator(cfg) for cfg in self.offtarget_configs]
        
        # Data storage
        self.smiles_list = []
        self.target_affinities = []
        self.selectivity_scores = []
        self.composite_scores = []
        self.iteration_indices = []
        self.fingerprints = None
    
    def _init_calculator(self, config: Dict) -> VinaCalculator:
        """Initialize VinaCalculator from config"""
        return VinaCalculator(
            protein_pdbqt=config["receptor"],
            center=tuple(config["center"]),
            size=tuple(config["size"]),
            cpus=self.vina_config.get("cpu", 4),
            exhaustiveness=self.vina_config.get("exhaustiveness", 8),
        )
    
    def evaluate_batch(self, smiles_list: List[str], iteration_index: int = 0, save_pdbqt: bool = False, n_processes: Optional[int] = None) -> None:
        """
        Evaluate a batch of SMILES in parallel using multiprocessing.
        
        Args:
            smiles_list: List of SMILES to evaluate
            iteration_index: Which iteration this batch is from
            save_pdbqt: Whether to save the output PDBQT files
            n_processes: Number of processes (default: use all available CPUs - 1)
        """
        if not smiles_list:
            return
        
        if n_processes is None:
            n_processes = max(1, cpu_count() - 1)
        
        logger.info(f"Evaluating batch of {len(smiles_list)} ligands using {n_processes} processes")
        
        # Create argument tuples for starmap
        eval_args = [
            (
                smiles,
                f"ligand_{len(self.smiles_list) + i}",
                self.target_calc,
                self.offtarget_calcs,
                self.target_weight,
                self.selectivity_weight,
                iteration_index,
                save_pdbqt
            )
            for i, smiles in enumerate(smiles_list)
        ]
        
        # Evaluate in parallel
        with Pool(processes=n_processes) as pool:
            results = pool.starmap(_evaluate_ligand_worker, eval_args)
        
        # Update state with results
        successful_count = 0
        for result in results:
            if result.get("success"):
                self.smiles_list.append(result["smiles"])
                self.target_affinities.append(result["target_affinity"])
                self.selectivity_scores.append(result["selectivity"])
                self.composite_scores.append(result["composite_score"])
                self.iteration_indices.append(result.get("iteration_index", iteration_index))
                successful_count += 1
        
        logger.info(f"Batch evaluation complete: {successful_count}/{len(smiles_list)} ligands evaluated successfully")
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Return ranked results"""
        df = pd.DataFrame({
            "smiles": self.smiles_list,
            "target_affinity": self.target_affinities,
            "selectivity": self.selectivity_scores,
            "composite_score": self.composite_scores,
        })
        
        return df.sort_values("composite_score", ascending=False).reset_index(drop=True)
