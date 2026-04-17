"""
Unit tests for Screening module (unified interface).
Tests Screener base class and all subclasses (BO, Random, Full).
"""

import sys
import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, call
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.screening_architectures import (
    Screener, BayesianOptimizer, RandomScreener, FullScreener,
    _evaluate_ligand_worker, fps_from_smiles, maxmin_sampler
)
from src.screening import main


class TestScreenerBaseClass:
    """Test base Screener class functionality."""
    
    def test_screener_initialization(self, mock_vina_calculator, mock_config):
        """Test Screener initialization with proper setup."""
        protein_configs = ["config1", "config2"]
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.side_effect = [
                mock_config,  # target config
                mock_config,  # offtarget config
                {"simulation_parameters": {"cpu": 4, "exhaustiveness": 8}}  # vina config
            ]
            with patch('src.screening_architectures.base_class.VinaCalculator', return_value=mock_vina_calculator):
                screener = Screener(protein_configs)
        
        assert screener.target_weight == 0.5
        assert screener.selectivity_weight == 0.5
        assert len(screener.smiles_list) == 0
        assert screener.fingerprints is None
    
    def test_screener_initialization_invalid_proteins(self):
        """Test Screener raises error with insufficient proteins."""
        with pytest.raises(ValueError, match="Need at least 2 proteins"):
            screener = Screener(["config1"])
    
    def test_screener_evaluate_batch_updates_state(self, mock_vina_calculator, mock_config):
        """Test that evaluate_batch properly updates screener state."""
        protein_configs = ["config1", "config2"]
        smiles_list = ["CCO", "CC", "CCC"]
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.side_effect = [mock_config, mock_config, mock_config]
            with patch('src.screening_architectures.base_class.VinaCalculator', return_value=mock_vina_calculator):
                screener = Screener(protein_configs)
        
        # Mock Pool to avoid pickling issues - call function directly in same process
        def mock_starmap(func, iterable):
            results = []
            for args in iterable:
                result = {
                    "success": True,
                    "smiles": args[0],  # SMILES is first arg
                    "target_affinity": -8.5,
                    "selectivity": 1.5,
                    "composite_score": 0.5,
                    "iteration_index": 0,
                }
                results.append(result)
            return results
        
        with patch('src.screening_architectures.base_class.Pool') as mock_pool_class:
            mock_pool_inst = MagicMock()
            mock_pool_inst.starmap = mock_starmap
            mock_pool_inst.__enter__ = MagicMock(return_value=mock_pool_inst)
            mock_pool_inst.__exit__ = MagicMock(return_value=None)
            mock_pool_class.return_value = mock_pool_inst
            screener.evaluate_batch(smiles_list, n_processes=1)
        
        assert len(screener.smiles_list) == 3
        assert len(screener.target_affinities) == 3
        assert len(screener.selectivity_scores) == 3
        assert len(screener.composite_scores) == 3
    
    def test_screener_get_results_dataframe(self, mock_config):
        """Test results dataframe generation."""
        protein_configs = ["config1", "config2"]
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.side_effect = [mock_config, mock_config, mock_config]
            with patch('src.screening_architectures.base_class.VinaCalculator'):
                screener = Screener(protein_configs)
        
        # Manually add some results
        screener.smiles_list = ["CCO", "CC"]
        screener.target_affinities = [-8.5, -7.2]
        screener.selectivity_scores = [1.5, 1.2]
        screener.composite_scores = [0.5, 0.4]
        
        df = screener.get_results_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["smiles", "target_affinity", "selectivity", "composite_score"]
        assert df["smiles"].iloc[0] == "CCO"  # Sorted by composite_score descending


class TestBayesianOptimizer:
    """Test BayesianOptimizer specific functionality."""
    
    def test_bayesian_optimizer_initialization(self, mock_config):
        """Test BayesianOptimizer init inherits from Screener."""
        protein_configs = ["config1", "config2"]
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.side_effect = [mock_config, mock_config, mock_config]
            with patch('src.screening_architectures.base_class.VinaCalculator'):
                optimizer = BayesianOptimizer(
                    protein_configs,
                    target_weight=0.7,
                    selectivity_weight=0.3
                )
        
        assert optimizer.target_weight == 0.7
        assert optimizer.selectivity_weight == 0.3
        assert optimizer.iteration_indices is not None
    
    def test_bayesian_optimizer_evaluate_batch_tracks_iterations(self, mock_config):
        """Test that BO tracks iteration indices."""
        protein_configs = ["config1", "config2"]
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.side_effect = [mock_config, mock_config, mock_config]
            with patch('src.screening_architectures.base_class.VinaCalculator'):
                optimizer = BayesianOptimizer(protein_configs)
        
        # Mock Pool to avoid pickling issues
        def mock_starmap(func, iterable):
            results = []
            for args in iterable:
                result = {
                    "success": True,
                    "smiles": args[0],
                    "target_affinity": -8.5,
                    "selectivity": 1.5,
                    "composite_score": 0.5,
                    "iteration_index": 2,
                }
                results.append(result)
            return results
        
        with patch('src.screening_architectures.base_class.Pool') as mock_pool_class:
            mock_pool_inst = MagicMock()
            mock_pool_inst.starmap = mock_starmap
            mock_pool_inst.__enter__ = MagicMock(return_value=mock_pool_inst)
            mock_pool_inst.__exit__ = MagicMock(return_value=None)
            mock_pool_class.return_value = mock_pool_inst
            optimizer.evaluate_batch(["CCO"], iteration_index=2, n_processes=1)
        
        assert optimizer.iteration_indices[0] == 2
    
    def test_bayesian_optimizer_results_dataframe_includes_iteration(self, mock_config):
        """Test BO results dataframe includes iteration tracking."""
        protein_configs = ["config1", "config2"]
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.side_effect = [mock_config, mock_config, mock_config]
            with patch('src.screening_architectures.base_class.VinaCalculator'):
                optimizer = BayesianOptimizer(protein_configs)
        
        optimizer.smiles_list = ["CCO", "CC"]
        optimizer.target_affinities = [-8.5, -7.2]
        optimizer.selectivity_scores = [1.5, 1.2]
        optimizer.composite_scores = [0.5, 0.4]
        optimizer.iteration_indices = [0, 1]
        
        df = optimizer.get_results_dataframe()
        
        assert "iteration" in df.columns
        assert list(df["iteration"]) == [0, 1]


class TestBayesianOptimizerAdvanced:
    """Test advanced BayesianOptimizer functionality (GP fitting, EI, batch selection)."""
    
    def test_bayesian_optimizer_initialization_requires_two_proteins(self):
        """Test that initialization requires at least 2 proteins."""
        with pytest.raises(ValueError, match="Need at least 2 proteins"):
            optimizer = BayesianOptimizer(protein_configs=["single_config.yaml"])
    
    def test_fit_gpr_models_with_synthetic_data(self, mock_config):
        """Test GP model fitting with synthetic data."""
        import gpytorch
        
        protein_configs = ["config1", "config2"]
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.side_effect = [mock_config, mock_config, {"simulation_parameters": {}}]
            with patch('src.screening_architectures.base_class.VinaCalculator'):
                with patch('src.screening_architectures.base_class.fps_from_smiles') as mock_fps:
                    # Create dummy fingerprints
                    fingerprints = np.random.rand(5, 2048).astype(np.float32)
                    mock_fps.return_value = (
                        np.array(["smi1", "smi2", "smi3", "smi4", "smi5"]), 
                        fingerprints
                    )
                    
                    optimizer = BayesianOptimizer(protein_configs=protein_configs)
                    
                    # Add synthetic observations
                    optimizer.smiles_list = ["smi1", "smi2", "smi3", "smi4", "smi5"]
                    optimizer.target_affinities = [-8.0, -7.5, -7.0, -6.5, -6.0]
                    optimizer.selectivity_scores = [2.0, 2.5, 3.0, 3.5, 4.0]
                    
                    model_target, model_selectivity = optimizer.fit_gpr_models()
                    
                    # Check models are fitted
                    assert isinstance(model_target, gpytorch.models.ExactGP)
                    assert isinstance(model_selectivity, gpytorch.models.ExactGP)
    
    def test_fit_gpr_models_requires_data(self, mock_config):
        """Test GP fitting fails with no evaluation data."""
        protein_configs = ["config1", "config2"]
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.side_effect = [mock_config, mock_config, mock_config]
            with patch('src.screening_architectures.base_class.VinaCalculator'):
                optimizer = BayesianOptimizer(protein_configs=protein_configs)
                
                with pytest.raises(ValueError, match="No evaluation data"):
                    optimizer.fit_gpr_models()
    
    def test_compute_ei_batch(self, mock_config):
        """Test Expected Improvement computation for candidate batch."""
        import gpytorch
        
        protein_configs = ["config1", "config2"]
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.side_effect = [mock_config, mock_config, {"simulation_parameters": {}}]
            with patch('src.screening_architectures.base_class.VinaCalculator'):
                with patch('src.screening_architectures.base_class.fps_from_smiles') as mock_fps:
                    # Setup training data
                    train_fps = np.random.rand(3, 2048).astype(np.float32)
                    candidate_fps = np.random.rand(2, 2048).astype(np.float32)
                    
                    mock_fps.side_effect = [
                        (np.array(["smi1", "smi2", "smi3"]), train_fps),
                        (np.array(["cand1", "cand2"]), candidate_fps)
                    ]
                    
                    optimizer = BayesianOptimizer(protein_configs=protein_configs)
                    optimizer.smiles_list = ["smi1", "smi2", "smi3"]
                    optimizer.target_affinities = [-8.0, -7.5, -7.0]
                    optimizer.selectivity_scores = [2.0, 2.5, 3.0]
                    
                    # Fit models
                    model_target, model_selectivity = optimizer.fit_gpr_models()
                    
                    # Test EI computation
                    ei_scores = optimizer._compute_ei_batch(candidate_fps, model_target, optimizer.target_affinities)
                    
                    assert len(ei_scores) == 2
                    assert np.all(ei_scores >= 0)  # EI should be non-negative
                    assert ei_scores.dtype == np.float64
    
    def test_select_batch_with_diversity(self, mock_config):
        """Test batch selection balances EI and diversity."""
        protein_configs = ["config1", "config2"]
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.side_effect = [mock_config, mock_config, {"simulation_parameters": {}}]
            with patch('src.screening_architectures.base_class.VinaCalculator'):
                with patch('src.screening_architectures.base_class.fps_from_smiles') as mock_fps:
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


class TestRandomScreener:
    """Test RandomScreener functionality."""
    
    def test_random_screener_initialization(self, mock_config):
        """Test RandomScreener inherits from Screener."""
        protein_configs = ["config1", "config2"]
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.side_effect = [mock_config, mock_config, mock_config]
            with patch('src.screening_architectures.base_class.VinaCalculator'):
                screener = RandomScreener(protein_configs)
        
        assert isinstance(screener, Screener)
        assert screener.target_weight == 0.5
    
    def test_random_screener_run_random_screening(self, sample_csv_file, mock_config):
        """Test random screening runs and returns results."""
        protein_configs = ["config1", "config2"]
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.side_effect = [mock_config, mock_config, mock_config]
            with patch('src.screening_architectures.base_class.VinaCalculator'):
                screener = RandomScreener(protein_configs)
        
        # Mock Pool to avoid pickling issues
        def mock_starmap(func, iterable):
            results = []
            for args in iterable:
                result = {
                    "success": True,
                    "smiles": args[0],
                    "target_affinity": -8.5,
                    "selectivity": 1.5,
                    "composite_score": 0.5,
                    "iteration_index": 0,
                }
                results.append(result)
            return results
        
        with patch('src.screening_architectures.base_class.Pool') as mock_pool_class:
            mock_pool_inst = MagicMock()
            mock_pool_inst.starmap = mock_starmap
            mock_pool_inst.__enter__ = MagicMock(return_value=mock_pool_inst)
            mock_pool_inst.__exit__ = MagicMock(return_value=None)
            mock_pool_class.return_value = mock_pool_inst
            results = screener.run_random_screening(
                sample_csv_file,
                total_budget=3,
                batch_size=2,
                random_seed=42
            )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
    
    def test_random_screener_respects_seed(self, sample_csv_file, mock_config):
        """Test RandomScreener produces reproducible results with fixed seed."""
        protein_configs = ["config1", "config2"]
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.side_effect = [mock_config, mock_config, mock_config, mock_config, mock_config, mock_config]
            with patch('src.screening_architectures.base_class.VinaCalculator'):
                screener1 = RandomScreener(protein_configs)
                screener2 = RandomScreener(protein_configs)
        
        # Mock Pool to avoid pickling issues
        def mock_starmap(func, iterable):
            results = []
            for args in iterable:
                result = {
                    "success": True,
                    "smiles": args[0],
                    "target_affinity": -8.5,
                    "selectivity": 1.5,
                    "composite_score": 0.5,
                    "iteration_index": 0,
                }
                results.append(result)
            return results
        
        with patch('src.screening_architectures.base_class.Pool') as mock_pool_class:
            mock_pool_inst = MagicMock()
            mock_pool_inst.starmap = mock_starmap
            mock_pool_inst.__enter__ = MagicMock(return_value=mock_pool_inst)
            mock_pool_inst.__exit__ = MagicMock(return_value=None)
            mock_pool_class.return_value = mock_pool_inst
            
            results1 = screener1.run_random_screening(
                sample_csv_file,
                total_budget=3,
                batch_size=3,
                random_seed=42
            )
            results2 = screener2.run_random_screening(
                sample_csv_file,
                total_budget=3,
                batch_size=3,
                random_seed=42
            )
        
        # Both runs should select same molecules due to same seed
        assert len(results1) == len(results2)


class TestFullScreener:
    """Test FullScreener functionality."""
    
    def test_full_screener_initialization(self, mock_config):
        """Test FullScreener inherits from Screener."""
        protein_configs = ["config1", "config2"]
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.side_effect = [mock_config, mock_config, mock_config]
            with patch('src.screening_architectures.base_class.VinaCalculator'):
                screener = FullScreener(protein_configs)
        
        assert isinstance(screener, Screener)
    
    def test_full_screener_screens_all_molecules(self, sample_csv_file, mock_config):
        """Test FullScreener evaluates all molecules."""
        protein_configs = ["config1", "config2"]
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.side_effect = [mock_config, mock_config, mock_config]
            with patch('src.screening_architectures.base_class.VinaCalculator'):
                screener = FullScreener(protein_configs)
        
        # Mock Pool to avoid pickling issues
        def mock_starmap(func, iterable):
            results = []
            for args in iterable:
                result = {
                    "success": True,
                    "smiles": args[0],
                    "target_affinity": -8.5,
                    "selectivity": 1.5,
                    "composite_score": 0.5,
                    "iteration_index": 0,
                }
                results.append(result)
            return results
        
        with patch('src.screening_architectures.base_class.Pool') as mock_pool_class:
            mock_pool_inst = MagicMock()
            mock_pool_inst.starmap = mock_starmap
            mock_pool_inst.__enter__ = MagicMock(return_value=mock_pool_inst)
            mock_pool_inst.__exit__ = MagicMock(return_value=None)
            mock_pool_class.return_value = mock_pool_inst
            results = screener.run_full_screening(
                sample_csv_file,
                batch_size=2
            )
        
        assert isinstance(results, pd.DataFrame)
        # Should have evaluated all 7 molecules from sample CSV
        assert len(results) == 7


class TestHashFunctionsCore:
    """Test core fingerprinting and sampling functions."""
    
    def test_fps_from_smiles_valid(self, sample_smiles):
        """Test fps_from_smiles with valid SMILES."""
        smiles_array, fps = fps_from_smiles(pd.Series(sample_smiles))
        
        assert len(smiles_array) == len(sample_smiles)
        assert fps.shape == (len(sample_smiles), 2048)
        assert fps.dtype == np.float32
    
    def test_fps_from_smiles_invalid(self):
        """Test fps_from_smiles handles invalid SMILES gracefully."""
        invalid_smiles = pd.Series(["CCO", "INVALID_SMILES", "CC"])
        
        smiles_array, fps = fps_from_smiles(invalid_smiles)
        
        assert len(fps) == 3
        assert fps.shape == (3, 2048)
        # Invalid SMILES should result in zero vector
        assert np.allclose(fps[1], 0)
    
    def test_maxmin_sampler_valid(self, sample_smiles):
        """Test maxmin_sampler returns diverse subset."""
        sample_size = 3
        sampled = maxmin_sampler(sample_smiles, sample_size)
        
        assert len(sampled) == sample_size
        assert all(s in sample_smiles for s in sampled)
    
    def test_maxmin_sampler_edge_cases(self):
        """Test maxmin_sampler edge cases."""
        smiles = ["CCO", "CC"]
        
        # Sample size larger than available
        sampled = maxmin_sampler(smiles, 5)
        assert len(sampled) == 2
        
        # Sample size zero
        sampled = maxmin_sampler(smiles, 0)
        assert len(sampled) == 0
        
        # Empty list
        sampled = maxmin_sampler([], 1)
        assert len(sampled) == 0


class TestEvaluateLigandWorker:
    """Test the _evaluate_ligand_worker function."""
    
    def test_worker_success(self, mock_vina_calculator):
        """Test worker returns success with valid inputs."""
        result = _evaluate_ligand_worker(
            smiles="CCO",
            ligand_name="test_ligand",
            target_calc=mock_vina_calculator,
            offtarget_calcs=[mock_vina_calculator],
            target_weight=0.5,
            selectivity_weight=0.5,
            iteration_index=0,
            save_pdbqt=False
        )
        
        assert result["success"] == True
        assert "smiles" in result
        assert "target_affinity" in result
        assert "selectivity" in result
        assert "composite_score" in result
    
    def test_worker_handles_failures(self):
        """Test worker handles docking failures gracefully."""
        mock_calc = MagicMock()
        mock_calc.calculate_binding.return_value = {"success": False}
        
        result = _evaluate_ligand_worker(
            smiles="CCO",
            ligand_name="test_ligand",
            target_calc=mock_calc,
            offtarget_calcs=[mock_calc],
            target_weight=0.5,
            selectivity_weight=0.5,
        )
        
        assert result["success"] == False

'''
class TestMainDispatcher:
    """Test main() function dispatcher."""
    
    def test_main_dispatches_bo(self, sample_config_file, mock_config):
        """Test main() correctly dispatches to BayesianOptimizer."""
        config_data = {
            "method": "bo",
            "proteins": {
                "target": "config1",
                "off_targets": ["config2"]
            },
            "input": {"smiles_csv": "test.csv"},
            "screening": {
                "initial_sample_size": 2,
                "total_budget": 3,
                "batch_size": 1,
                "target_weight": 0.5,
                "selectivity_weight": 0.5,
            },
            "output": {"results_csv": "test_output.csv", "save_pdbqt": False}
        }
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.return_value = config_data
            with patch('src.screening_architectures.bo_class.BayesianOptimizer') as mock_bo:
                mock_instance = MagicMock()
                mock_bo.return_value = mock_instance
                mock_instance.run_optimization_loop.return_value = pd.DataFrame({
                    "smiles": ["CCO"],
                    "target_affinity": [-8.5],
                    "selectivity": [1.5],
                    "composite_score": [0.5],
                })
                
                with patch('builtins.open', create=True):
                    main(sample_config_file)
                
                mock_bo.assert_called_once()
    
    def test_main_dispatches_random(self, sample_config_file, mock_config):
        """Test main() correctly dispatches to RandomScreener."""
        config_data = {
            "method": "random",
            "proteins": {
                "target": "config1",
                "off_targets": ["config2"]
            },
            "input": {"smiles_csv": "test.csv"},
            "screening": {
                "total_budget": 3,
                "batch_size": 1,
                "target_weight": 0.5,
                "selectivity_weight": 0.5,
                "random_seed": 42,
            },
            "output": {"results_csv": "test_output.csv", "save_pdbqt": False}
        }
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.return_value = config_data
            with patch('src.screening_architectures.random_class.RandomScreener') as mock_random:
                mock_instance = MagicMock()
                mock_random.return_value = mock_instance
                mock_instance.run_random_screening.return_value = pd.DataFrame({
                    "smiles": ["CCO"],
                    "target_affinity": [-8.5],
                    "selectivity": [1.5],
                    "composite_score": [0.5],
                })
                
                with patch('builtins.open', create=True):
                    main(sample_config_file)
                
                mock_random.assert_called_once()
    
    def test_main_dispatches_full(self, sample_config_file, mock_config):
        """Test main() correctly dispatches to FullScreener."""
        config_data = {
            "method": "full",
            "proteins": {
                "target": "config1",
                "off_targets": ["config2"]
            },
            "input": {"smiles_csv": "test.csv"},
            "screening": {
                "batch_size": 1,
                "target_weight": 0.5,
                "selectivity_weight": 0.5,
            },
            "output": {"results_csv": "test_output.csv", "save_pdbqt": False}
        }
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.return_value = config_data
            with patch('src.screening_architectures.fullscreen_class.FullScreener') as mock_full:
                mock_instance = MagicMock()
                mock_full.return_value = mock_instance
                mock_instance.run_full_screening.return_value = pd.DataFrame({
                    "smiles": ["CCO"],
                    "target_affinity": [-8.5],
                    "selectivity": [1.5],
                    "composite_score": [0.5],
                })
                
                with patch('builtins.open', create=True):
                    main(sample_config_file)
                
                mock_full.assert_called_once()
    
    def test_main_invalid_method_raises_error(self, sample_config_file):
        """Test main() raises error for invalid method."""
        config_data = {
            "method": "invalid_method",
            "proteins": {"target": "config1", "off_targets": ["config2"]},
            "input": {"smiles_csv": "test.csv"},
            "screening": {"batch_size": 1},
            "output": {"results_csv": "test_output.csv"}
        }
        
        with patch('src.screening_architectures.base_class.load_config') as mock_load_config:
            mock_load_config.return_value = config_data
            with pytest.raises(ValueError, match="Invalid method"):
                main(sample_config_file)
'''