"""
Unit tests for ligand_preparation module.
Tests SMILES to MOL conversion and MOL to PDBQT conversion.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

from src.ligand_preparation import smiles_to_mol, mol_to_pdbqt


class TestSmilesToMol:
    """Test SMILES to MOL conversion."""
    
    def test_valid_smiles(self):
        """Test conversion of valid SMILES string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mol"
            result = smiles_to_mol("CCO", str(output_path))
            
            assert Path(result).exists()
            assert str(output_path) == result
            assert result.endswith(".mol")
    
    def test_default_output_path(self):
        """Test that default output path is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = smiles_to_mol("CCO")
                
                assert Path(result).exists()
                assert result == "ligand.mol"
            finally:
                os.chdir(old_cwd)
    
    def test_invalid_smiles_raises_error(self):
        """Test that invalid SMILES raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SMILES"):
            smiles_to_mol("INVALID_SMILES_XYZ")
    
    def test_empty_smiles_raises_error(self):
        """Test that empty SMILES raises ValueError."""
        with pytest.raises(ValueError):
            smiles_to_mol("")
    
    def test_output_directory_creation(self):
        """Test that output directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "subdir1" / "subdir2" / "ligand.mol"
            result = smiles_to_mol("CCO", str(nested_path))
            
            assert Path(result).exists()
            assert nested_path.exists()
    
    def test_various_smiles(self):
        """Test conversion of various valid SMILES strings."""
        test_smiles = [
            "CC",  # ethane
            "CCO",  # ethanol
            "c1ccccc1",  # benzene
            "CC(=O)O",  # acetic acid
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, smiles in enumerate(test_smiles):
                output_path = Path(tmpdir) / f"mol_{i}.mol"
                result = smiles_to_mol(smiles, str(output_path))
                assert Path(result).exists()


class TestMolToPdbqt:
    """Test MOL to PDBQT conversion."""
    
    @patch('subprocess.run')
    def test_valid_mol_file(self, mock_run):
        """Test conversion of valid MOL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy mol file
            mol_path = Path(tmpdir) / "test.mol"
            mol_path.write_text("dummy mol content")
            
            output_path = Path(tmpdir) / "test.pdbqt"
            result = mol_to_pdbqt(str(mol_path), str(output_path))
            
            assert result == str(output_path)
            mock_run.assert_called_once()
            
            # Check the command
            call_args = mock_run.call_args
            assert call_args[0][0][0].endswith("mk_prepare_ligand.exe") or call_args[0][0][0].endswith("mk_prepare_ligand")
            assert "-i" in call_args[0][0]
            assert "-o" in call_args[0][0]
    
    def test_nonexistent_mol_file(self):
        """Test that FileNotFoundError is raised for missing MOL file."""
        with pytest.raises(FileNotFoundError, match="Mol file not found"):
            mol_to_pdbqt("/nonexistent/path/file.mol", "output.pdbqt")
    
    @patch('subprocess.run')
    def test_output_directory_creation(self, mock_run):
        """Test that output directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mol_path = Path(tmpdir) / "test.mol"
            mol_path.write_text("dummy mol content")
            
            output_path = Path(tmpdir) / "nested" / "dir" / "output.pdbqt"
            result = mol_to_pdbqt(str(mol_path), str(output_path))
            
            assert output_path.parent.exists()
    
    @patch('subprocess.run')
    def test_meeko_command_failure(self, mock_run):
        """Test that subprocess error is handled properly."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "mk_prepare_ligand", stderr="Command failed"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mol_path = Path(tmpdir) / "test.mol"
            mol_path.write_text("dummy mol content")
            
            with pytest.raises(subprocess.CalledProcessError):
                mol_to_pdbqt(str(mol_path), "output.pdbqt")
    
    @patch('subprocess.run')
    def test_command_construction(self, mock_run):
        """Test that the mk_prepare_ligand command is constructed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mol_path = Path(tmpdir) / "ligand.mol"
            mol_path.write_text("dummy content")
            
            output_path = Path(tmpdir) / "ligand.pdbqt"
            mol_to_pdbqt(str(mol_path), str(output_path))
            
            # Verify command structure
            call_args = mock_run.call_args[0][0]
            assert call_args[0].endswith("mk_prepare_ligand.exe") or call_args[0].endswith("mk_prepare_ligand")
            assert call_args[1] == "-i"
            assert str(mol_path) in call_args
            assert call_args[-2] == "-o"
            assert str(output_path) in call_args


class TestIntegration:
    """Integration tests for the full workflow."""
    
    @patch('subprocess.run')
    def test_smiles_to_pdbqt_workflow(self, mock_run):
        """Test the complete workflow from SMILES to PDBQT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Convert SMILES to MOL
            mol_path = Path(tmpdir) / "ligand.mol"
            mol_result = smiles_to_mol("CCO", str(mol_path))
            assert Path(mol_result).exists()
            
            # Convert MOL to PDBQT
            pdbqt_path = Path(tmpdir) / "ligand.pdbqt"
            pdbqt_result = mol_to_pdbqt(mol_result, str(pdbqt_path))
            
            assert pdbqt_result == str(pdbqt_path)
            mock_run.assert_called_once()
