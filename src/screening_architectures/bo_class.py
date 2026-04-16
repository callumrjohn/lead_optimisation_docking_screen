"""
Bayesian Optimization screening strategy.
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
import gpytorch
from rdkit.Chem import DataStructs
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.screening_architectures.base_class import Screener, fps_from_smiles, maxmin_sampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianOptimizer(Screener):
    """Multi-objective Bayesian optimization for selective binding"""
    
    def __init__(
        self,
        protein_configs: List[str],
        vina_config: str = "configs/vina_binding.yaml",
        target_weight: float = 0.5,
        selectivity_weight: float = 0.5,
    ):
        """Initialize BO optimizer (inherits from Screener base class)"""
        super().__init__(protein_configs, vina_config, target_weight, selectivity_weight)
        logger.info(f"Initialized BO with 1 target + {len(self.offtarget_calcs)} off-targets")
    
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
        ei_target_weight: float = 0.5,
        ei_selectivity_weight: float = 0.5,
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
            ei_target_weight: Weight for target affinity EI
            ei_selectivity_weight: Weight for selectivity EI
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
        ei_composite = ei_target_weight * ei_target + ei_selectivity_weight * ei_selectivity
        
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
        target_weight: float = 0.5,
        selectivity_weight: float = 0.5,
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
                    ei_target_weight=target_weight,
                    ei_selectivity_weight=selectivity_weight,
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
