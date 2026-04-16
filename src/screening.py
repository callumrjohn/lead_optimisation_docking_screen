"""
Lead Optimization Screening
Unified screening interface supporting multiple strategies via configuration:
- BO: Bayesian Optimization with diversity constraints
- Random: Random sampling baseline
- Full: Complete exhaustive screening

This module serves as the main dispatcher that routes to the appropriate
screening architecture based on the config file.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.screening_architectures import BayesianOptimizer, RandomScreener, FullScreener
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def main(config_file: str = "configs/screening.yaml"):
    """
    Run screening from configuration file.
    Dispatches to BO, Random, or Full screening based on config method.
    
    Args:
        config_file: Path to YAML configuration file with screening parameters
    """
    # Load configuration
    logger.info(f"Loading configuration from {config_file}")
    config = load_config(config_file)
    
    # Get screening method
    method = config.get("method", "bo").lower()
    if method not in ["bo", "random", "full"]:
        raise ValueError(f"Invalid method: {method}. Must be 'bo', 'random', or 'full'")
    
    logger.info(f"Using screening method: {method.upper()}")
    
    # Extract protein configurations
    protein_config = config["proteins"]
    target_protein = protein_config["target"]
    off_target_proteins = protein_config.get("off_targets", [])
    
    protein_configs = [target_protein] + off_target_proteins
    logger.info(f"Loaded {len(protein_configs)} protein configs: {protein_configs}")
    
    # Extract input parameters
    input_config = config["input"]
    smiles_file = input_config["smiles_csv"]
    
    # Extract common parameters
    screening_config = config.get("screening", {})
    target_weight = screening_config.get("target_weight", 0.5)
    selectivity_weight = screening_config.get("selectivity_weight", 0.5)
    batch_size = screening_config.get("batch_size", 5)
    
    # Extract output parameters
    output_config = config["output"]
    output_csv = output_config["results_csv"]
    save_pdbqt = output_config.get("save_pdbqt", False)
    
    logger.info(
        f"Screening Configuration:\n"
        f"  Method: {method.upper()}\n"
        f"  Proteins: {len(protein_configs)} (1 target + {len(off_target_proteins)} off-targets)\n"
        f"  SMILES CSV: {smiles_file}\n"
        f"  Target weight: {target_weight}\n"
        f"  Selectivity weight: {selectivity_weight}\n"
        f"  Batch size: {batch_size}\n"
        f"  Output: {output_csv}"
    )
    
    # Run appropriate screening method
    if method == "bo":
        initial_sample_size = screening_config.get("initial_sample_size", 5)
        total_budget = screening_config.get("total_budget", 50)
        ei_weight = screening_config.get("ei_weight", 0.7)
        diversity_weight = screening_config.get("diversity_weight", 0.3)
        
        logger.info(f"BO Parameters: budget={total_budget}, initial_sample={initial_sample_size}")
        
        optimizer = BayesianOptimizer(
            protein_configs=protein_configs,
            target_weight=target_weight,
            selectivity_weight=selectivity_weight,
            ei_weight=ei_weight,
            diversity_weight=diversity_weight
        )
        results = optimizer.run_optimization_loop(
            initial_smiles_file=smiles_file,
            initial_sample_size=initial_sample_size,
            total_budget=total_budget,
            batch_size=batch_size,
            save_pdbqt=save_pdbqt
        )
    
    elif method == "random":
        total_budget = screening_config.get("total_budget", 50)
        random_seed = screening_config.get("random_seed", 42)
        
        logger.info(f"Random Parameters: budget={total_budget}, seed={random_seed}")
        
        screener = RandomScreener(
            protein_configs=protein_configs,
            target_weight=target_weight,
            selectivity_weight=selectivity_weight
        )
        results = screener.run_random_screening(
            smiles_file=smiles_file,
            total_budget=total_budget,
            batch_size=batch_size,
            save_pdbqt=save_pdbqt,
            random_seed=random_seed
        )
    
    elif method == "full":
        screener = FullScreener(
            protein_configs=protein_configs,
            target_weight=target_weight,
            selectivity_weight=selectivity_weight
        )
        results = screener.run_full_screening(
            smiles_file=smiles_file,
            batch_size=batch_size,
            save_pdbqt=save_pdbqt,
        )
    
    print("\n=== Top Results ===")
    print(results.head(10))
    
    results.to_csv(output_csv, index=False)
    logger.info(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()
