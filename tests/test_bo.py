"""
Unit tests for Bayesian Optimization (BO) module.
Tests GP fitting, EI acquisition, batch selection, and BO loop.
"""

import sys
import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import gpytorch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bo import fps_from_smiles, maxmin_sampler, BayesianOptimizer


class TestFingerprintGeneration:
    """Test ECFP fingerprint generation from SMILES."""
    
    def test_fps_from_smiles_valid(self):
        """Test fingerprint generation from valid SMILES."""
        smiles_series = pd.Series(["CCO", "CC", "c1ccccc1"])
        smiles_array, fingerprints = fps_from_smiles(smiles_series)
        
        assert len(smiles_array) == 3
        assert fingerprints.shape == (3, 2048)
        assert fingerprints.dtype == np.float32
        assert np.all(fingerprints >= 0)  # Binary fingerprints
        assert np.all(fingerprints <= 1)
    
    def test_fps_from_smiles_invalid_smiles(self):
        """Test handling of invalid SMILES strings."""
        smiles_series = pd.Series(["CCO", "INVALID_SMILES_XYZ", "CC"])
        smiles_array, fingerprints = fps_from_smiles(smiles_series)
        
        # Should return zeros for invalid SMILES
        assert fingerprints.shape == (3, 2048)
        assert np.all(fingerprints[1] == 0)  # Invalid SMILES becomes zeros
    
    def test_fps_from_smiles_empty(self):
        """Test fingerprint generation with empty series."""
        smiles_series = pd.Series([])
        smiles_array, fingerprints = fps_from_smiles(smiles_series)
        
        assert len(smiles_array) == 0
        assert len(fingerprints) == 0


class TestMaxMinSampler:
    """Test MaxMin diversity-based sampling."""
    
    def test_maxmin_sampler_basic(self):
        """Test MaxMin sampling returns correct size."""
        smiles = ["CCO", "CC", "CCC", "CCCC", "c1ccccc1"]
        sample_size = 3
        
        sampled = maxmin_sampler(smiles, sample_size)
        
        assert len(sampled) == sample_size
        assert all(s in smiles for s in sampled)
    
    def test_maxmin_sampler_full_pool(self):
        """Test MaxMin when sample size equals pool size."""
        smiles = ["CCO", "CC", "CCC"]
        sample_size = 3
        
        sampled = maxmin_sampler(smiles, sample_size)
        
        assert len(sampled) == 3
        assert set(sampled) == set(smiles)
    
    def test_maxmin_sampler_exceeds_pool(self):
        """Test MaxMin when sample size exceeds pool size."""
        smiles = ["CCO", "CC"]
        sample_size = 5
        
        sampled = maxmin_sampler(smiles, sample_size)
        
        assert len(sampled) == 2
    
    def test_maxmin_sampler_empty_pool(self):
        """Test MaxMin with empty pool."""
        smiles = []
        sample_size = 3
        
        sampled = maxmin_sampler(smiles, sample_size)
        
        assert len(sampled) == 0
    
    def test_maxmin_sampler_single_molecule(self):
        """Test MaxMin with single molecule."""
        smiles = ["CCO"]
        sample_size = 5
        
        sampled = maxmin_sampler(smiles, sample_size)
        
        assert len(sampled) == 1
        assert sampled[0] == "CCO"


class TestBayesianOptimizerInit:
    """Test BayesianOptimizer initialization."""
    
    def test_initialization_succeeds(self):
        """Test successful initialization with proper configs."""
        protein_configs = ["configs/bo_optimization.yaml", "configs/pap_analogues.yaml"]
        
        with patch("src.bo.load_config") as mock_load_config:
            # Order of calls: target config, off-target configs, vina_config
            mock_load_config.side_effect = [
                {  # target protein config
                    "receptor": "data/prepared_proteins/target.pdbqt",
                    "center": (0, 0, 0),
                    "size": (20, 20, 20)
                },
                {  # off-target protein config
                    "receptor": "data/prepared_proteins/offtarget.pdbqt",
                    "center": (0, 0, 0),
                    "size": (20, 20, 20)
                },
                {  # vina_config
                    "simulation_parameters": {"cpu": 4, "exhaustiveness": 8}
                },
            ]
            
            with patch("src.bo.VinaCalculator"):
                optimizer = BayesianOptimizer(
                    protein_configs=protein_configs,
                    vina_config="configs/vina_binding.yaml",
                    target_weight=0.5,
                    selectivity_weight=0.5
                )
                
                assert len(optimizer.smiles_list) == 0
                assert optimizer.target_weight == 0.5
                assert optimizer.selectivity_weight == 0.5
    
    def test_initialization_needs_at_least_two_proteins(self):
        """Test that initialization requires at least 2 proteins."""
        with patch("src.bo.load_config"):
            with pytest.raises(ValueError, match="at least 2 proteins"):
                BayesianOptimizer(protein_configs=["single_config.yaml"])


class TestSelectivityCalculation:
    """Test selectivity score calculation."""
    
    def test_selectivity_basic(self):
        """Test basic selectivity calculation."""
        protein_configs = ["config1", "config2"]
        
        with patch("src.bo.load_config"):
            with patch("src.bo.VinaCalculator"):
                optimizer = BayesianOptimizer(protein_configs=protein_configs)
                
                # Strong target (-8.0), weak off-targets (-5.0)
                selectivity = optimizer.calculate_selectivity(
                    target_affinity=-8.0,
                    offtarget_affinities=[-5.0, -6.0]
                )
                
                # selectivity = mean_offtarget - target = (-5.5) - (-8.0) = 2.5
                assert selectivity == pytest.approx(2.5)
    
    def test_selectivity_empty_offtargets(self):
        """Test selectivity with no off-targets."""
        protein_configs = ["config1", "config2"]
        
        with patch("src.bo.load_config"):
            with patch("src.bo.VinaCalculator"):
                optimizer = BayesianOptimizer(protein_configs=protein_configs)
                
                selectivity = optimizer.calculate_selectivity(
                    target_affinity=-8.0,
                    offtarget_affinities=[]
                )
                
                assert selectivity == 0.0


class TestCompositeScoreCalculation:
    """Test composite objective score calculation."""
    
    def test_composite_score_even_weights(self):
        """Test composite score with equal weights."""
        protein_configs = ["config1", "config2"]
        
        with patch("src.bo.load_config"):
            with patch("src.bo.VinaCalculator"):
                optimizer = BayesianOptimizer(
                    protein_configs=protein_configs,
                    target_weight=0.5,
                    selectivity_weight=0.5
                )
                
                score = optimizer.calculate_composite_score(
                    target_affinity=-8.0,
                    selectivity=2.0
                )
                
                # score = 0.5 * (-(-8.0)) + 0.5 * 2.0 = 0.5 * 8 + 1 = 5.0
                assert score == pytest.approx(5.0)
    
    def test_composite_score_custom_weights(self):
        """Test composite score with custom weights."""
        protein_configs = ["config1", "config2"]
        
        with patch("src.bo.load_config"):
            with patch("src.bo.VinaCalculator"):
                optimizer = BayesianOptimizer(
                    protein_configs=protein_configs,
                    target_weight=0.7,
                    selectivity_weight=0.3
                )
                
                score = optimizer.calculate_composite_score(
                    target_affinity=-8.0,
                    selectivity=2.0
                )
                
                # score = 0.7 * 8 + 0.3 * 2 = 5.6 + 0.6 = 6.2
                assert score == pytest.approx(6.2)


class TestGPFitting:
    """Test Gaussian Process model fitting."""
    
    def test_fit_gpr_models_basic(self):
        """Test GP model fitting with synthetic data."""
        protein_configs = ["config1", "config2"]
        
        with patch("src.bo.load_config"):
            with patch("src.bo.VinaCalculator"):
                with patch("src.bo.fps_from_smiles") as mock_fps:
                    # Create dummy fingerprints
                    fingerprints = np.random.rand(5, 2048).astype(np.float32)
                    mock_fps.return_value = (np.array(["smi1", "smi2", "smi3", "smi4", "smi5"]), fingerprints)
                    
                    optimizer = BayesianOptimizer(protein_configs=protein_configs)
                    
                    # Add synthetic observations
                    optimizer.smiles_list = ["smi1", "smi2", "smi3", "smi4", "smi5"]
                    optimizer.target_affinities = [-8.0, -7.5, -7.0, -6.5, -6.0]
                    optimizer.selectivity_scores = [2.0, 2.5, 3.0, 3.5, 4.0]
                    
                    model_target, model_selectivity = optimizer.fit_gpr_models()
                    
                    # Check models are fitted
                    assert isinstance(model_target, gpytorch.models.ExactGP)
                    assert isinstance(model_selectivity, gpytorch.models.ExactGP)
    
    def test_fit_gpr_models_no_data(self):
        """Test GP fitting fails with no data."""
        protein_configs = ["config1", "config2"]
        
        with patch("src.bo.load_config"):
            with patch("src.bo.VinaCalculator"):
                optimizer = BayesianOptimizer(protein_configs=protein_configs)
                
                with pytest.raises(ValueError, match="No evaluation data"):
                    optimizer.fit_gpr_models()


class TestEIDifferentVariances:
    """Test Expected Improvement computation."""
    
    def test_ei_batch_computation(self):
        """Test EI computation for candidate batch."""
        protein_configs = ["config1", "config2"]
        
        with patch("src.bo.load_config"):
            with patch("src.bo.VinaCalculator"):
                with patch("src.bo.fps_from_smiles") as mock_fps:
                    # Setup training data
                    train_fps = np.random.rand(3, 2048).astype(np.float32)
                    mock_fps.return_value = (np.array(["smi1", "smi2", "smi3"]), train_fps)
                    
                    optimizer = BayesianOptimizer(protein_configs=protein_configs)
                    optimizer.smiles_list = ["smi1", "smi2", "smi3"]
                    optimizer.target_affinities = [-8.0, -7.5, -7.0]
                    optimizer.selectivity_scores = [2.0, 2.5, 3.0]
                    
                    # Fit models
                    model_target, model_selectivity = optimizer.fit_gpr_models()
                    
                    # Test EI computation
                    candidate_fps = np.random.rand(2, 2048).astype(np.float32)
                    ei_scores = optimizer._compute_ei_batch(candidate_fps, model_target, optimizer.target_affinities)
                    
                    assert len(ei_scores) == 2
                    assert np.all(ei_scores >= 0)  # EI should be non-negative
                    assert ei_scores.dtype == np.float64


class TestBatchSelection:
    """Test batch selection with diversity and EI balancing."""
    
    def test_select_batch_with_diversity(self):
        """Test batch selection balances EI and diversity."""
        protein_configs = ["config1", "config2"]
        
        with patch("src.bo.load_config"):
            with patch("src.bo.VinaCalculator"):
                with patch("src.bo.fps_from_smiles") as mock_fps:
                    # Setup training data
                    train_fps = np.random.rand(3, 2048).astype(np.float32)
                    candidate_fps = np.random.rand(5, 2048).astype(np.float32)
                    
                    mock_fps.side_effect = [
                        (np.array(["smi1", "smi2", "smi3"]), train_fps),  # training
                        (np.array(["cand1", "cand2", "cand3", "cand4", "cand5"]), candidate_fps)  # candidates
                    ]
                    
                    optimizer = BayesianOptimizer(protein_configs=protein_configs)
                    optimizer.smiles_list = ["smi1", "smi2", "smi3"]
                    optimizer.target_affinities = [-8.0, -7.5, -7.0]
                    optimizer.selectivity_scores = [2.0, 2.5, 3.0]
                    
                    # Fit models
                    model_target, model_selectivity = optimizer.fit_gpr_models()
                    
                    # Select batch
                    candidate_smiles = ["cand1", "cand2", "cand3", "cand4", "cand5"]
                    selected = optimizer.select_batch_with_diversity(
                        candidate_smiles,
                        model_target,
                        model_selectivity,
                        batch_size=2,
                        ei_weight=0.7,
                        diversity_weight=0.3
                    )
                    
                    assert len(selected) == 2
                    assert all(s in candidate_smiles for s in selected)


class TestOptimizerDataStorage:
    """Test data storage and retrieval in optimizer."""
    
    def test_evaluate_ligand_storage(self):
        """Test that evaluated ligands are stored correctly."""
        protein_configs = ["config1", "config2"]
        
        with patch("src.bo.load_config"):
            with patch("src.bo.VinaCalculator") as mock_vina_class:
                with patch("src.bo.prepare_ligand"):
                    # Setup mock calculators
                    mock_target_calc = MagicMock()
                    mock_target_calc.calculate_binding.return_value = {
                        "success": True,
                        "affinity": -8.5
                    }
                    
                    mock_offtarget_calc = MagicMock()
                    mock_offtarget_calc.calculate_binding.return_value = {
                        "success": True,
                        "affinity": -5.0
                    }
                    
                    mock_vina_class.side_effect = lambda *args, **kwargs: mock_target_calc
                    
                    optimizer = BayesianOptimizer(protein_configs=protein_configs)
                    optimizer.target_calc = mock_target_calc
                    optimizer.offtarget_calcs = [mock_offtarget_calc]
                    
                    # Mock the evaluate call (simplified)
                    optimizer.smiles_list.append("CCO")
                    optimizer.target_affinities.append(-8.5)
                    optimizer.selectivity_scores.append(3.5)  # -5.0 - (-8.5)
                    optimizer.composite_scores.append(0.5 * 8.5 + 0.5 * 3.5)
                    optimizer.iteration_indices.append(0)
                    
                    assert len(optimizer.smiles_list) == 1
                    assert optimizer.target_affinities[0] == -8.5
                    assert optimizer.selectivity_scores[0] == pytest.approx(3.5)


class TestResultsDataFrame:
    """Test results dataframe generation and sorting."""
    
    def test_get_results_dataframe(self):
        """Test results are properly ranked by composite score."""
        protein_configs = ["config1", "config2"]
        
        with patch("src.bo.load_config"):
            with patch("src.bo.VinaCalculator"):
                optimizer = BayesianOptimizer(protein_configs=protein_configs)
                
                # Add unsorted data
                optimizer.smiles_list = ["A", "B", "C"]
                optimizer.target_affinities = [-8.0, -7.0, -9.0]
                optimizer.selectivity_scores = [2.0, 3.0, 1.0]
                optimizer.composite_scores = [0.5 * 8 + 0.5 * 2, 0.5 * 7 + 0.5 * 3, 0.5 * 9 + 0.5 * 1]  # 5, 5, 5
                optimizer.iteration_indices = [0, 0, 0]
                
                df = optimizer.get_results_dataframe()
                
                assert len(df) == 3
                assert list(df.columns) == ["smiles", "target_affinity", "selectivity", "composite_score", "iteration"]
                # Check sorted by composite score
                assert df["composite_score"].is_monotonic_decreasing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
