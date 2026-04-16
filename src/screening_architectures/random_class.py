"""
Random screening strategy - baseline screening approach.
"""

import sys
import logging
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.screening_architectures.base_class import Screener

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomScreener(Screener):
    """Random screening benchmark for ligand evaluation"""
    
    def __init__(
        self,
        protein_configs: List[str],
        vina_config: str = "configs/vina_binding.yaml",
        target_weight: float = 0.5,
        selectivity_weight: float = 0.5,
    ):
        """Initialize random screener (inherits from Screener base class)"""
        super().__init__(protein_configs, vina_config, target_weight, selectivity_weight)
        logger.info(f"Initialized random screener with 1 target + {len(self.offtarget_calcs)} off-targets")
    
    def run_random_screening(
        self,
        smiles_file: str,
        total_budget: int = 50,
        batch_size: int = 5,
        save_pdbqt: bool = False,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Run random screening on a subset of SMILES.
        
        Args:
            smiles_file: Path to CSV with SMILES column
            total_budget: Total evaluations to perform
            batch_size: Batch size per evaluation cycle
            save_pdbqt: Whether to save the output PDBQT files
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame of results ranked by composite score
        """
        logger.info(f"Starting random screening (budget={total_budget})")
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Load all SMILES
        df = pd.read_csv(smiles_file)
        all_smiles = df["analogue_smiles"].tolist()
        
        logger.info(f"Loaded {len(all_smiles)} SMILES from {smiles_file}")
        
        # Randomly sample up to budget
        sample_size = min(total_budget, len(all_smiles))
        sampled_indices = np.random.choice(len(all_smiles), size=sample_size, replace=False)
        screened_smiles = [all_smiles[i] for i in sorted(sampled_indices)]
        
        logger.info(f"Randomly selected {sample_size} SMILES for evaluation")
        
        # Evaluate in batches
        n_batches = (len(screened_smiles) + batch_size - 1) // batch_size
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(screened_smiles))
            batch = screened_smiles[start_idx:end_idx]
            
            logger.info(f"\nBatch {batch_idx + 1}/{n_batches} ({len(batch)} ligands)")
            self.evaluate_batch(batch, save_pdbqt=save_pdbqt)
        
        logger.info("\n=== Random Screening Complete ===")
        return self.get_results_dataframe()
