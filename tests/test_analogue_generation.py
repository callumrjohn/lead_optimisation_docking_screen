"""
Unit tests for analogue_generation module.
Tests scaffold-based molecular analogue generation.
"""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import from root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdkit import Chem

from src.analogue_generation import (
    find_dummy_by_mapnum,
    get_number_of_dummy_atoms,
    pool_substituents,
    combine_and_connect,
    generate_combination_df
)


class TestFindDummyByMapnum:
    """Test finding dummy atoms by map number."""
    
    def test_find_existing_dummy(self):
        """Test finding a dummy atom with specific map number."""
        mol = Chem.MolFromSmiles("c1ccc([*:1])cc1")
        dummy = find_dummy_by_mapnum(mol, 1)
        
        assert dummy is not None
        assert dummy.GetAtomicNum() == 0
        assert dummy.GetAtomMapNum() == 1
    
    def test_find_multiple_dummies(self):
        """Test finding specific dummy atoms in molecule with multiple dummies."""
        mol = Chem.MolFromSmiles("[*:1]c1ccc([*:2])cc1[*:3]")
        
        dummy1 = find_dummy_by_mapnum(mol, 1)
        dummy2 = find_dummy_by_mapnum(mol, 2)
        dummy3 = find_dummy_by_mapnum(mol, 3)
        
        assert dummy1.GetAtomMapNum() == 1
        assert dummy2.GetAtomMapNum() == 2
        assert dummy3.GetAtomMapNum() == 3
    
    def test_find_nonexistent_dummy_raises_error(self):
        """Test that ValueError is raised for non-existent dummy number."""
        mol = Chem.MolFromSmiles("[*:1]c1ccccc1")
        
        with pytest.raises(ValueError, match="Could not find dummy atom"):
            find_dummy_by_mapnum(mol, 999)


class TestGetNumberOfDummyAtoms:
    """Test counting dummy atoms."""
    
    def test_count_single_dummy(self):
        """Test counting a single dummy atom."""
        mol = Chem.MolFromSmiles("[*]c1ccccc1")
        assert get_number_of_dummy_atoms(mol) == 1
    
    def test_count_multiple_dummies(self):
        """Test counting multiple dummy atoms."""
        mol = Chem.MolFromSmiles("[*]c1ccc([*])cc1[*]")
        assert get_number_of_dummy_atoms(mol) == 3
    
    def test_count_zero_dummies(self):
        """Test that benzene has no dummies."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert get_number_of_dummy_atoms(mol) == 0
    
    def test_count_dummies_with_mapnums(self):
        """Test counting dummies with atom map numbers."""
        mol = Chem.MolFromSmiles("[*:1]c1ccc([*:2])cc1")
        assert get_number_of_dummy_atoms(mol) == 2


class TestPoolSubstituents:
    """Test pooling scaffold and substituents."""
    
    def test_pool_single_substituent(self):
        """Test pooling scaffold with single substituent."""
        scaffold = Chem.MolFromSmiles("c1ccc([*:1])cc1")
        substituent = Chem.MolFromSmiles("[*]C")
        
        pooled = pool_substituents(scaffold, substituent)
        
        assert pooled is not None
        assert get_number_of_dummy_atoms(pooled) == 2
    
    def test_pool_multiple_substituents(self):
        """Test pooling scaffold with multiple substituents."""
        scaffold = Chem.MolFromSmiles("[*:1]c1ccc([*:2])cc1")
        sub1 = Chem.MolFromSmiles("[*]C")
        sub2 = Chem.MolFromSmiles("[*]CC")
        
        pooled = pool_substituents(scaffold, [sub1, sub2])
        
        assert get_number_of_dummy_atoms(pooled) == 4
    
    def test_mismatch_dummies_raises_error(self):
        """Test that mismatch in dummy count raises error."""
        scaffold = Chem.MolFromSmiles("[*:1]c1ccc([*:2])cc1")  # 2 dummies
        sub1 = Chem.MolFromSmiles("[*]C")  # 1 dummy
        
        with pytest.raises(ValueError, match="Scaffold must have exactly"):
            pool_substituents(scaffold, [sub1])
    
    def test_substituent_multiple_dummies_raises_error(self):
        """Test that substituent with multiple dummies raises error."""
        scaffold = Chem.MolFromSmiles("[*:1]c1ccccc1")
        bad_sub = Chem.MolFromSmiles("[*]C[*]")  # 2 dummies
        
        with pytest.raises(ValueError, match="Each substituent must have exactly one dummy"):
            pool_substituents(scaffold, bad_sub)


class TestCombineAndConnect:
    """Test combining and connecting molecules."""
    
    def test_simple_connection(self):
        """Test connecting scaffold with single substituent."""
        scaffold = Chem.MolFromSmiles("[*:1]c1ccccc1")
        substituent = Chem.MolFromSmiles("[*:2]C")
        
        pooled = pool_substituents(scaffold, substituent)
        
        # Connect dummy 1 (substituent) to dummy 2 (scaffold)
        result = combine_and_connect(pooled, dummy_index_pairs=[(1, 2)])
        
        assert result is not None
        assert get_number_of_dummy_atoms(result) == 0
        assert result.GetNumAtoms() > 0
    
    def test_multiple_connections(self):
        """Test connecting scaffold with multiple substituents."""
        # Create scaffold with 2 dummies
        scaffold = Chem.MolFromSmiles("[*:1]c1ccc([*:2])cc1")
        sub1 = Chem.MolFromSmiles("[*:3]C")
        sub2 = Chem.MolFromSmiles("[*:4]CC")
        
        pooled = pool_substituents(scaffold, [sub1, sub2])
        
        # Connect dummies: (1->3), (2->4) where 1,2 are from substituents, 3,4 from scaffold
        result = combine_and_connect(pooled, dummy_index_pairs=[(1, 3), (2, 4)])
        
        assert get_number_of_dummy_atoms(result) == 0
    
    def test_result_is_valid_molecule(self):
        """Test that result is a valid sanitized molecule."""
        scaffold = Chem.MolFromSmiles("[*:1]c1ccccc1")
        substituent = Chem.MolFromSmiles("[*:2]C")
        
        pooled = pool_substituents(scaffold, substituent)
        result = combine_and_connect(pooled, dummy_index_pairs=[(1, 2)])
        
        # Should be able to convert to SMILES
        smiles = Chem.MolToSmiles(result)
        assert smiles is not None
        assert len(smiles) > 0


class TestGenerateCombinationDf:
    """Test generating combination DataFrame."""
    
    def test_simple_combination(self):
        """Test generating a simple set of combinations."""
        template = "[*:1]c1ccccc1"
        substituents = {
            1: {
                "methyl": "[*:2]C",
                "ethyl": "[*:2]CC"
            }
        }
        dummy_pairs = [(1, 2)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "analogues.csv"
            df = generate_combination_df(template, substituents, dummy_pairs, str(out_path))
            
            assert len(df) == 2
            assert "scaffold" in df.columns
            assert "R1" in df.columns
            assert "analogue_smiles" in df.columns
            assert out_path.exists()
    
    def test_multiple_positions_withH(self):
        """Test generating combinations with multiple substituent positions."""
        template = "[*:1]c1ccc([*:2])cc1"
        substituents = {
            1: {"methyl": "[*:3]C", "ethyl": "[*:3]CC"},
            2: {"H": "[*:4]", "F": "[*:4]F"}
        }
        dummy_pairs = [(1, 3), (2, 4)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "analogues.csv"
            df = generate_combination_df(template, substituents, dummy_pairs, str(out_path))
            
            # Should have 2 x 2 = 4 combinations
            assert len(df) == 4
            assert "R1" in df.columns
            assert "R2" in df.columns

    def test_multiple_positions(self):
        """Test generating combinations with multiple substituent positions."""
        template = "[*:1]c1ccc([*:2])cc1"
        substituents = {
            1: {"methyl": "[*:3]C", "ethyl": "[*:3]CC"},
            2: {"H": "[*:4]CC", "F": "[*:4]F"}
        }
        dummy_pairs = [(1, 3), (2, 4)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "analogues.csv"
            df = generate_combination_df(template, substituents, dummy_pairs, str(out_path))
            
            # Should have 2 x 2 = 4 combinations
            assert len(df) == 4
            assert "R1" in df.columns
            assert "R2" in df.columns
    
    def test_output_csv_validity(self):
        """Test that output CSV is valid and readable."""
        template = "[*:1]c1ccccc1"
        substituents = {
            1: {"methyl": "[*:2]C", "ethyl": "[*:2]CC"}
        }
        dummy_pairs = [(1, 2)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "analogues.csv"
            df = generate_combination_df(template, substituents, dummy_pairs, str(out_path))
            
            # Read back the CSV
            df_read = pd.read_csv(out_path)
            
            assert len(df_read) == len(df)
            assert list(df_read.columns) == list(df.columns)
    
    def test_valid_smiles_generated(self):
        """Test that generated SMILES are valid."""
        template = "[*:1]c1ccccc1"
        substituents = {
            1: {"methyl": "[*:2]C", "ethyl": "[*:2]CC"}
        }
        dummy_pairs = [(1, 2)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "analogues.csv"
            df = generate_combination_df(template, substituents, dummy_pairs, str(out_path))
            
            # All SMILES should be valid
            for smiles in df["analogue_smiles"]:
                mol = Chem.MolFromSmiles(smiles)
                assert mol is not None


class TestIntegration:
    """Integration tests for full workflow."""
    
    def test_full_scaffold_generation_workflow(self):
        """Test complete workflow from template to generated analogues."""
        # Define scaffold and substituents like in real use case
        template = "[*:1]c1cc([*:2])ccc1"  # para-disubstituted benzene
        substituents = {
            1: {
                "phenyl": "[*:3]c1ccccc1",
                "methyl": "[*:3]C"
            },
            2: {
                "cyano": "[*:4]C#N",
                "methoxy": "[*:4]OC"
            }
        }
        dummy_pairs = [(1, 3), (2, 4)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "analogues.csv"
            df = generate_combination_df(
                template, substituents, dummy_pairs, str(out_path)
            )
            
            # Should generate 2 x 2 = 4 analogues
            assert len(df) == 4
            
            # Verify all required columns
            required_cols = ["scaffold", "R1", "R2", "analogue_smiles"]
            for col in required_cols:
                assert col in df.columns
            
            # Verify all SMILES are valid and contain no dummy atoms
            for smiles in df["analogue_smiles"]:
                mol = Chem.MolFromSmiles(smiles)
                assert mol is not None
                assert get_number_of_dummy_atoms(mol) == 0
