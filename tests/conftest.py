"""
Pytest configuration and shared fixtures.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add root directory to Python path for imports
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))


# ============================================================================
# Fixtures for Mocking External Dependencies
# ============================================================================

@pytest.fixture
def mock_vina_calculator():
    """Mock VinaCalculator for faster tests without real docking."""
    mock_calc = MagicMock()
    
    # Mock successful docking result
    def side_effect_success(pdbqt_path, name=None, save_pdbqt=False):
        return {
            "success": True,
            "affinity": np.random.uniform(-10, -5),  # Realistic affinity range
            "name": name
        }
    
    mock_calc.calculate_binding.side_effect = side_effect_success
    return mock_calc


@pytest.fixture
def mock_prepare_ligand():
    """Mock prepare_ligand function that returns dummy PDBQT paths."""
    def mock_func(smiles, name="ligand", output_dir="data/prepared_ligands"):
        return f"{output_dir}/{name}.pdbqt"
    return mock_func


@pytest.fixture
def mock_config():
    """Mock config dictionary for testing."""
    return {
        "receptor": "data/prepared_proteins/test_protein.pdbqt",
        "center": (0.0, 0.0, 0.0),
        "size": (20, 20, 20),
        "simulation_parameters": {
            "cpu": 4,
            "exhaustiveness": 8
        }
    }


@pytest.fixture
def sample_smiles():
    """Sample SMILES list for testing."""
    return [
        "CCO",                           # ethanol
        "CC(C)O",                        # isopropanol
        "c1ccccc1",                      # benzene
        "c1ccc(cc1)O",                   # phenol
        "CCN(CC)CC",                     # triethylamine
        "C1=CC=C(C=C1)C(=O)O",           # benzoic acid
        "CC(=O)OC1=CC=CC=C1C(=O)O",      # aspirin
    ]


@pytest.fixture
def sample_affinities():
    """Sample binding affinities for testing."""
    return np.array([-8.5, -7.2, -6.9, -8.1, -7.5, -6.8, -8.3])


@pytest.fixture
def sample_csv_file(tmp_path, sample_smiles):
    """Create a temporary CSV file with SMILES for testing."""
    df = pd.DataFrame({
        "analogue_smiles": sample_smiles,
        "compound_id": [f"compound_{i}" for i in range(len(sample_smiles))]
    })
    csv_file = tmp_path / "test_smiles.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def sample_config_file(tmp_path):
    """Create a temporary config file for testing."""
    import yaml
    
    config = {
        "method": "bo",
        "proteins": {
            "target": "configs/protein_configs/test_target.yaml",
            "off_targets": [
                "configs/protein_configs/test_offtarget1.yaml",
                "configs/protein_configs/test_offtarget2.yaml",
            ]
        },
        "input": {
            "smiles_csv": "data/test_smiles.csv"
        },
        "screening": {
            "batch_size": 5,
            "target_weight": 0.5,
            "selectivity_weight": 0.5,
            "initial_sample_size": 2,
            "total_budget": 10,
            "random_seed": 42,
        },
        "output": {
            "results_csv": "test_results.csv",
            "save_pdbqt": False,
        }
    }
    
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_file)


@pytest.fixture
def patched_vina_and_ligand(mock_vina_calculator, mock_prepare_ligand):
    """Patch both VinaCalculator and prepare_ligand for integration tests."""
    with patch('src.screening.VinaCalculator', return_value=mock_vina_calculator), \
         patch('src.screening.prepare_ligand', side_effect=mock_prepare_ligand):
        yield mock_vina_calculator, mock_prepare_ligand
