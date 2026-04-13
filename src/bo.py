"""
Bayesian Optimization Loop for Lead Optimization
Multi-objective optimization: binding affinity + selectivity with diversity constraints
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import csv
import tempfile
from scipy.stats import norm
import torch
import gpytorch

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters import MaxMinPicker
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
        iteration_index: BO iteration index
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


class BayesianOptimizer:
    """Multi-objective Bayesian optimization for selective binding"""
    
    def __init__(
        self,
        protein_configs: List[str],
        vina_config: str = "configs/vina_binding.yaml",
        target_weight: float = 0.5,
        selectivity_weight: float = 0.5,
    ):
        """
        Initialize BO optimizer.
        
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
        self.iteration_indices = []  # Track which iteration each ligand was evaluated in
        self.fingerprints = None
        
        logger.info(f"Initialized BO with 1 target + {len(self.offtarget_calcs)} off-targets")
    
    def _init_calculator(self, config: Dict) -> VinaCalculator:
        """Initialize VinaCalculator from config"""
        return VinaCalculator(
            protein_pdbqt=config["receptor"],
            center=tuple(config["center"]),
            size=tuple(config["size"]),
            cpus=self.vina_config.get("cpu", 4),
            exhaustiveness=self.vina_config.get("exhaustiveness", 8),
        )
    
    def calculate_selectivity(self, target_affinity: float, offtarget_affinities: List[float]) -> float:
        """
        Calculate selectivity score.
        
        Args:
            target_affinity: Binding affinity to target (kcal/mol)
            offtarget_affinities: List of affinities to off-targets
            
        Returns:
            Selectivity score (higher is better)
        """
        if len(offtarget_affinities) == 0:
            return 0.0
        
        mean_offtarget = np.mean(offtarget_affinities)
        selectivity = mean_offtarget - target_affinity  # Weak off-target - strong target = high selectivity
        return selectivity
    
    def calculate_composite_score(self, target_affinity: float, selectivity: float) -> float:
        """
        Calculate composite objective score (multi-objective).
        
        Args:
            target_affinity: Target binding affinity
            selectivity: Selectivity score
            
        Returns:
            Composite score = 0.5 * (-affinity) + 0.5 * selectivity
        """
        return self.target_weight * (-target_affinity) + self.selectivity_weight * selectivity
    
    def _dock_offtargets_batch(self, pdbqt_path: str, ligand_name: str) -> List[float]:
        """
        Dock ligand to all off-targets sequentially (for non-parallel evaluation).
        
        Args:
            pdbqt_path: Path to prepared ligand PDBQT
            ligand_name: Ligand name for logging
            
        Returns:
            List of off-target binding affinities
        """
        offtarget_affinities = []
        for i, calc in enumerate(self.offtarget_calcs):
            result = calc.calculate_binding(pdbqt_path, name=f"{ligand_name}_offtarget_{i}")
            if result["success"] and result["affinity"] is not None:
                offtarget_affinities.append(result["affinity"])
        return offtarget_affinities
    
    def evaluate_ligand(self, smiles: str, ligand_name: str = "ligand", iteration_index: int = 0, save_pdbqt: bool = False) -> Dict:
        """
        Evaluate a single ligand sequentially (without parallelization).
        For parallel batch evaluation, use evaluate_batch() instead.
        
        Args:
            smiles: SMILES string
            ligand_name: Name for output files
            iteration_index: Which BO iteration this ligand is from (0 = initial batch)
            save_pdbqt: Whether to save the output PDBQT file
            
        Returns:
            Dictionary with results
        """
        try:
            logger.info(f"Evaluating {ligand_name}...")
            
            # Prepare
            pdbqt_path = prepare_ligand(smiles, name=ligand_name, output_dir="data/prepared_ligands")
            
            # Target docking
            target_result = self.target_calc.calculate_binding(pdbqt_path, name=ligand_name, save_pdbqt=save_pdbqt)
            if not target_result["success"]:
                logger.warning(f"Target binding failed for {ligand_name}")
                return {"success": False}
            
            target_affinity = target_result["affinity"]
            
            # Off-target docking sequentially
            offtarget_affinities = self._dock_offtargets_batch(pdbqt_path, ligand_name)
            
            if len(offtarget_affinities) == 0:
                logger.warning(f"No off-target results for {ligand_name}")
                return {"success": False}
            
            # Calculate selectivity and composite score
            selectivity = self.calculate_selectivity(target_affinity, offtarget_affinities)
            composite_score = self.calculate_composite_score(target_affinity, selectivity)
            
            # Store results
            self.smiles_list.append(smiles)
            self.target_affinities.append(target_affinity)
            self.selectivity_scores.append(selectivity)
            self.composite_scores.append(composite_score)
            self.iteration_indices.append(iteration_index)
            
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
            }
        
        except Exception as e:
            logger.error(f"Error evaluating {ligand_name}: {e}")
            return {"success": False}
    
    def evaluate_batch(self, smiles_list: List[str], iteration_index: int = 0, save_pdbqt: bool = False, n_processes: Optional[int] = None) -> None:
        """
        Evaluate a batch of SMILES in parallel using multiprocessing.
        
        Args:
            smiles_list: List of SMILES to evaluate
            iteration_index: Which BO iteration this batch is from
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
        
        # Update optimizer state with results
        successful_count = 0
        for result in results:
            if result.get("success"):
                self.smiles_list.append(result["smiles"])
                self.target_affinities.append(result["target_affinity"])
                self.selectivity_scores.append(result["selectivity"])
                self.composite_scores.append(result["composite_score"])
                self.iteration_indices.append(result["iteration_index"])
                successful_count += 1
        
        logger.info(f"Batch evaluation complete: {successful_count}/{len(smiles_list)} ligands evaluated successfully")
    
    def fit_gpr_models(self) -> Tuple[gpytorch.models.ExactGP, gpytorch.models.ExactGP]:
        """
        Fit Gaussian Process models for target affinity and selectivity using Gauche Tanimoto kernel.
        
        Returns:
            Tuple of (target_model, selectivity_model)
        """
        if len(self.smiles_list) == 0:
            raise ValueError("No evaluation data")
        
        try:
            logger.info("Generating ECFP fingerprints...")
            _, self.fingerprints = fps_from_smiles(pd.Series(self.smiles_list))
            logger.info(f"Generated fingerprints: shape {self.fingerprints.shape}")
        except Exception as e:
            logger.error(f"Failed to generate fingerprints: {e}", exc_info=True)
            raise
        
        # Convert fingerprints to torch tensors
        X_train = torch.tensor(self.fingerprints, dtype=torch.float32)
        
        # Convert observations to torch tensors
        y_target = torch.tensor([-a for a in self.target_affinities], dtype=torch.float32)  # Negative for maximization
        y_selectivity = torch.tensor([s for s in self.selectivity_scores], dtype=torch.float32)
        
        logger.info(f"Target affinity range: {min(self.target_affinities):.2f} to {max(self.target_affinities):.2f}")
        logger.info(f"Selectivity range: {min(self.selectivity_scores):.2f} to {max(self.selectivity_scores):.2f}")
        
        # Define GP class with Tanimoto kernel
        class TanimotoExactGP(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
            
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
        # Fit models with Tanimoto kernel
        logger.info("Fitting GPR models with Tanimoto kernel...")
        
        # Target affinity model
        try:
            logger.info("Fitting target affinity model...")
            likelihood_target = gpytorch.likelihoods.GaussianLikelihood()
            model_target = TanimotoExactGP(X_train, y_target, likelihood_target)
            self._fit_gp_model(model_target, likelihood_target)
            logger.info("Target affinity model fitted successfully")
        except Exception as e:
            logger.error(f"Failed to fit target model: {e}", exc_info=True)
            raise
        
        # Selectivity model
        try:
            logger.info("Fitting selectivity model...")
            likelihood_selectivity = gpytorch.likelihoods.GaussianLikelihood()
            model_selectivity = TanimotoExactGP(X_train, y_selectivity, likelihood_selectivity)
            self._fit_gp_model(model_selectivity, likelihood_selectivity)
            logger.info("Selectivity model fitted successfully")
        except Exception as e:
            logger.error(f"Failed to fit selectivity model: {e}", exc_info=True)
            raise
        
        return model_target, model_selectivity
    
    
    def select_batch_with_diversity(
        self,
        candidate_smiles: List[str],
        model_target: gpytorch.models.ExactGP,
        model_selectivity: gpytorch.models.ExactGP,
        batch_size: int = 5,
        ei_weight: float = 0.7,
        diversity_weight: float = 0.3,
    ) -> List[str]:
        """
        Select batch of molecules balancing Expected Improvement + Tanimoto diversity.
        
        Args:
            candidate_smiles: Pool of candidates to select from
            model_target: Fitted GP model for target affinity
            model_selectivity: Fitted GP model for selectivity
            batch_size: Number of candidates to select
            ei_weight: Weight for acquisition function (0.7)
            diversity_weight: Weight for diversity penalty (0.3)
            
        Returns:
            Selected SMILES list
        """
        logger.info(f"Selecting {batch_size} candidates balancing EI ({ei_weight}) + diversity ({diversity_weight})")
        
        # Generate fingerprints for candidates
        candidate_array, candidate_fps = fps_from_smiles(pd.Series(candidate_smiles))
        
        # Convert to RDKit format for diversity penalty
        candidate_fps_rdk = [DataStructs.ExplicitBitVect(2048) for _ in range(len(candidate_fps))]
        for i, fp in enumerate(candidate_fps):
            on_bits = np.flatnonzero(fp)
            for b in on_bits:
                candidate_fps_rdk[i].SetBit(int(b))
        
        evaluated_fps_rdk = [DataStructs.ExplicitBitVect(2048) for _ in range(len(self.fingerprints))]
        for i, fp in enumerate(self.fingerprints):
            on_bits = np.flatnonzero(fp)
            for b in on_bits:
                evaluated_fps_rdk[i].SetBit(int(b))
        
        # Compute EI for target affinity using Gauche predictions
        logger.info("Computing Expected Improvement for target affinity...")
        ei_target = self._compute_ei_batch(candidate_fps, model_target, self.target_affinities)
        
        # Compute EI for selectivity using Gauche predictions
        logger.info("Computing Expected Improvement for selectivity...")
        ei_selectivity = self._compute_ei_batch(candidate_fps, model_selectivity, self.selectivity_scores)
        
        # Normalize EI scores to [0, 1]
        ei_target = (ei_target - ei_target.min()) / (ei_target.max() - ei_target.min() + 1e-8)
        ei_selectivity = (ei_selectivity - ei_selectivity.min()) / (ei_selectivity.max() - ei_selectivity.min() + 1e-8)
        
        # Multi-objective acquisition: combine target and selectivity
        # Both are things we want to maximize (target affinity negated, selectivity positive)
        ei_composite = 0.5 * ei_target + 0.5 * ei_selectivity
        
        # Compute diversity penalty (1 - max similarity to evaluated set)
        logger.info("Computing diversity penalties...")
        diversity_scores = []
        for cand_fp in candidate_fps_rdk:
            similarities = DataStructs.BulkTanimotoSimilarity(cand_fp, evaluated_fps_rdk)
            max_similarity = max(similarities) if similarities else 0
            diversity_penalty = 1.0 - max_similarity  # Higher = more distant
            diversity_scores.append(diversity_penalty)
        
        diversity_scores = np.array(diversity_scores)
        diversity_scores = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min() + 1e-8)
        
        # Combine EI and diversity with weights
        logger.info(f"Balancing acquisition ({ei_weight}) and diversity ({diversity_weight})...")
        combined_scores = ei_weight * ei_composite - diversity_weight * (1.0 - diversity_scores)
        
        # Select top-M by combined score
        selected_indices = np.argsort(combined_scores)[-batch_size:]
        selected_smiles = candidate_array[selected_indices]
        
        logger.info(
            f"Selected {len(selected_smiles)} candidates:\n"
            f"  Mean EI: {ei_composite[selected_indices].mean():.3f}\n"
            f"  Mean diversity: {diversity_scores[selected_indices].mean():.3f}"
        )
        
        return selected_smiles.tolist()
    
    def _compute_ei_batch(
        self,
        candidate_fps: np.ndarray,
        model: gpytorch.models.ExactGP,
        observations: List[float],
    ) -> np.ndarray:
        """
        Compute Expected Improvement (EI) for batch of candidates using GP predictions.
        
        Args:
            candidate_fps: Fingerprints of candidate molecules (n_candidates, n_features)
            model: Fitted GP model (GPyTorch ExactGP with Tanimoto kernel)
            observations: Observations used to fit the model
            
        Returns:
            Array of EI scores for each candidate
        """
        # Convert fingerprints to torch tensor
        X_test = torch.tensor(candidate_fps, dtype=torch.float32)
        
        # Use model for predictions
        model.eval()
        with torch.no_grad():
            try:
                # Get predictions from model
                predictions = model.likelihood(model(X_test))
                mean_predictions = predictions.mean.numpy()
                variance_predictions = predictions.variance.numpy()
            except Exception as e:
                logger.warning(f"GP prediction failed: {e}. Using fallback.")
                return np.zeros(len(candidate_fps))
        
        # Get current best observation
        best_observation = np.max(observations)
        
        ei_scores = []
        
        # Compute EI for each candidate
        for i in range(len(candidate_fps)):
            mean = mean_predictions[i]
            variance = max(variance_predictions[i], 1e-8)
            std = np.sqrt(variance)
            
            # Expected Improvement formula
            if std > 1e-8:
                Z = (mean - best_observation) / std
                ei = (mean - best_observation) * norm.cdf(Z) + std * norm.pdf(Z)
            else:
                ei = max(0, mean - best_observation)
            
            ei_scores.append(max(0, ei))  # EI should be non-negative
        
        return np.array(ei_scores)
    
    def _fit_gp_model(
        self,
        model: gpytorch.models.ExactGP,
        likelihood: gpytorch.likelihoods.Likelihood,
        n_iterations: int = 50,
    ) -> None:
        """
        Fit a GP model using Adam optimizer.
        
        Args:
            model: GP model to fit
            likelihood: Likelihood function
            n_iterations: Number of optimization iterations
        """
        model.train()
        likelihood.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        for iter in range(n_iterations):
            optimizer.zero_grad()
            output = model(model.train_inputs[0])
            loss = -mll(output, model.train_targets)
            loss.backward()
            optimizer.step()
        
        model.eval()
        likelihood.eval()
    
    def run_optimization_loop(
        self,
        initial_smiles_file: str,
        initial_sample_size: int = 5,
        total_budget: int = 50,
        batch_size: int = 5,
        save_pdbqt: bool = False,
    ) -> pd.DataFrame:
        """
        Run the full BO optimization loop.
        
        Args:
            initial_smiles_file: Path to CSV with SMILES column
            initial_sample_size: Initial set size from MaxMin sampling
            total_budget: Total evaluations allowed
            batch_size: Batch size per iteration
            save_pdbqt: Whether to save the output PDBQT files
        Returns:
            DataFrame of results ranked by composite score
        """
        logger.info(f"Starting BO optimization loop (budget={total_budget})")
        
        # Load initial SMILES
        df = pd.read_csv(initial_smiles_file)
        
        all_smiles = df["analogue_smiles"].tolist()
        
        # MaxMin sampling for diversity
        logger.info(f"MaxMin sampling: {initial_sample_size} from {len(all_smiles)}")
        initial_subset = maxmin_sampler(all_smiles, initial_sample_size)
        
        # Evaluate initial batch
        logger.info(f"Evaluating initial batch of {len(initial_subset)}")
        self.evaluate_batch(initial_subset.tolist(), iteration_index=0, save_pdbqt=save_pdbqt)
        
        n_iterations = (total_budget - len(self.smiles_list)) // batch_size
        
        # BO loop
        for iteration in range(n_iterations):
            logger.info(f"\n=== BO Iteration {iteration + 1}/{n_iterations} ===")
            logger.info(f"Evaluated so far: {len(self.smiles_list)}/{total_budget}")
            
            # Fit surrogate models
            try:
                logger.info(f"Fitting GP models with {len(self.smiles_list)} observations...")
                model_target, model_selectivity = self.fit_gpr_models()
                logger.info("GP models fitted successfully")
            except Exception as e:
                logger.error(f"Failed to fit models: {e}", exc_info=True)
                break
            
            # Pool of candidates for selection
            candidate_pool = [s for s in all_smiles if s not in self.smiles_list]
            logger.info(f"Candidate pool size: {len(candidate_pool)}")
            
            if len(candidate_pool) == 0:
                logger.warning("No candidates left to evaluate")
                break
            
            # Select next batch with diversity
            try:
                logger.info(f"Selecting batch of {batch_size} from {len(candidate_pool)} candidates...")
                next_batch = self.select_batch_with_diversity(
                    candidate_pool,
                    model_target=model_target,
                    model_selectivity=model_selectivity,
                    batch_size=batch_size,
                    ei_weight=0.7,
                    diversity_weight=0.3,
                )
                logger.info(f"Selected batch: {next_batch}")
            except Exception as e:
                logger.error(f"Failed to select batch: {e}", exc_info=True)
                break
            
            # Evaluate
            logger.info(f"Evaluating batch of {len(next_batch)}")
            self.evaluate_batch(next_batch, iteration_index=iteration+1, save_pdbqt=save_pdbqt)
        
        logger.info("\n=== Optimization Complete ===")
        return self.get_results_dataframe()
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Return ranked results with iteration tracking"""
        df = pd.DataFrame({
            "smiles": self.smiles_list,
            "target_affinity": self.target_affinities,
            "selectivity": self.selectivity_scores,
            "composite_score": self.composite_scores,
            "iteration": self.iteration_indices,
        })
        
        return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


def main(config_file: str = "configs/bo_optimisation.yaml"):
    """
    Run BO optimization from configuration file.
    
    Args:
        config_file: Path to YAML configuration file with BO parameters
    """
    # Load configuration
    logger.info(f"Loading configuration from {config_file}")
    config = load_config(config_file)
    
    # Extract protein configurations
    protein_config = config["proteins"]
    target_protein = protein_config["target"]
    off_target_proteins = protein_config.get("off_targets", [])
    
    protein_configs = [target_protein] + off_target_proteins
    logger.info(f"Loaded {len(protein_configs)} protein configs: {protein_configs}")
    
    # Extract input parameters
    input_config = config["input"]
    initial_smiles = input_config["smiles_csv"]
    
    # Extract optimization parameters
    opt_config = config["optimization"]
    budget = opt_config["total_budget"]
    initial_sample_size = opt_config["initial_sample_size"]
    batch_size = opt_config["batch_size"]
    target_weight = opt_config.get("target_weight", 0.5)
    selectivity_weight = opt_config.get("selectivity_weight", 0.5)
    
    # Extract output parameters
    output_config = config["output"]
    output_csv = output_config["results_csv"]
    save_pdbqt = output_config.get("save_pdbqt", False)
    
    logger.info(
        f"BO Configuration:\n"
        f"  Proteins: {len(protein_configs)} (1 target + {len(off_target_proteins)} off-targets)\n"
        f"  SMILES CSV: {initial_smiles}\n"
        f"  Budget: {budget}\n"
        f"  Initial sample: {initial_sample_size}\n"
        f"  Batch size: {batch_size}\n"
        f"  Target weight: {target_weight}\n"
        f"  Selectivity weight: {selectivity_weight}\n"
        f"  Output: {output_csv}"
    )
    
    # Run optimization
    optimizer = BayesianOptimizer(
        protein_configs=protein_configs,
        target_weight=target_weight,
        selectivity_weight=selectivity_weight
    )
    results = optimizer.run_optimization_loop(
        initial_smiles_file=initial_smiles,
        initial_sample_size=initial_sample_size,
        total_budget=budget,
        batch_size=batch_size,
        save_pdbqt=save_pdbqt
    )
    
    print("\n=== Top Results ===")
    print(results.head(10))
    
    results.to_csv(output_csv, index=False)
    logger.info(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()
