"""
Full screening strategy - exhaustive screening approach.
"""

import sys
import logging
from pathlib import Path
from typing import List
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.screening_architectures.base_class import Screener

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FullScreener(Screener):
    """Full screening for comprehensive ligand evaluation"""
    
    def __init__(
        self,
        protein_configs: List[str],
        vina_config: str = "configs/vina_binding.yaml",
        target_weight: float = 0.5,
        selectivity_weight: float = 0.5,
    ):
        """Initialize full screener (inherits from Screener base class)"""
        super().__init__(protein_configs, vina_config, target_weight, selectivity_weight)
        logger.info(f"Initialized full screener with 1 target + {len(self.offtarget_calcs)} off-targets")
    
    def run_full_screening(
        self,
        smiles_file: str,
        batch_size: int = 5,
        save_pdbqt: bool = False,
    ) -> pd.DataFrame:
        """
        Run complete screening on all SMILES.
        
        Args:
            smiles_file: Path to CSV with SMILES column
            batch_size: Batch size per evaluation cycle
            save_pdbqt: Whether to save the output PDBQT files
            
        Returns:
            DataFrame of results ranked by composite score
        """
        logger.info(f"Starting full screening (batch size={batch_size})")
        
        # Load all SMILES
        df = pd.read_csv(smiles_file)
        all_smiles = df["analogue_smiles"].tolist()
        
        logger.info(f"Loaded {len(all_smiles)} SMILES from {smiles_file}")
        logger.info(f"Screening all {len(all_smiles)} ligands")
        
        # Evaluate all in batches
        n_batches = (len(all_smiles) + batch_size - 1) // batch_size
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_smiles))
            batch = all_smiles[start_idx:end_idx]
            
            logger.info(f"\nBatch {batch_idx + 1}/{n_batches} ({len(batch)} ligands, {start_idx + 1}-{end_idx}/{len(all_smiles)} total)")
            self.evaluate_batch(batch, save_pdbqt=save_pdbqt)
        
        logger.info("\n=== Full Screening Complete ===")
        logger.info(f"Successfully evaluated {len(self.smiles_list)}/{len(all_smiles)} ligands")
        return self.get_results_dataframe()
