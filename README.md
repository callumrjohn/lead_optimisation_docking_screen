# Lead-optimisation Docking Screen

This is a personal mini-project to recreate (in a simplified way) a lead optimisation campaign that led to the discovery of **Imatinib** using a few _in silico_ methods. It is inspired by a 1996 SAR study on PAP derivatives (DOI: 10.1002/ardp.19963290707).

The repo includes tools for: **analogue generation**, **ligand preparation**, and **binding affinity calculations** using AutoDock Vina with multiprocessing support.

**Current Tech Stack:** RDKit • Meeko • AutoDock Vina • Python Multiprocessing • YAML Configuration


## Overview

This repository provides a complete workflow for computational drug discovery:
- **Analogue generation**: Systematic creation of compound variants through scaffold-based substitution (YAML-configured)
- **Ligand preparation**: Prepare ligands from SMILES using Meeko for docking studies
- **Binding affinity calculation**: Dock ligands to proteins using AutoDock Vina with parallel processing
- **Results extraction**: Parse binding affinities from Vina output

## Project Structure

```
├── src/
│   ├── analogue_generation.py       # Scaffold-based compound generation
│   ├── ligand_preparation.py        # Molecular structure preparation via Meeko
│   ├── vina_binding.py              # Binding affinity calculations with AutoDock Vina
│   └── data_extraction.py           # Parse results from Vina output
├── data/
│   ├── protein_structures/          # Input protein PDB files
│   ├── prepared_proteins/           # Meeko-prepared protein PDBQT structures
│   ├── prepared_ligands/            # Meeko-prepared ligand PDBQT files
│   └── analogue_sets/               # Generated analogue CSV files
├── configs/
│   ├── analogue_generator.yaml      # Configuration for analogue generation
│   └── pap_analogues.yaml           # PAP derivatives configuration
├── tests/
│   ├── test_analogue_generation.py
│   ├── test_ligand_preparation.py
│   └── conftest.py
├── utils/
│   └── config.py                    # YAML configuration utilities
├── environment.yml                  # Conda environment specification
└── README.md
```

## Installation

### Requirements
- Python 3.11+
- Conda (Miniconda or Anaconda)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/callumrjohn/lead_optimisation_docking_screen.git
cd lead_optimisation_docking_screen
```

2. Create the conda environment:
```bash
conda env create -f environment.yaml
conda activate lead_optimisation_docking_screen
```

3. Verify installation by running tests:
```bash
pytest
```

4. Download AutoDock Vina executable, add to the repo folder, and name "vina.exe"

## Current functionalities

### 1. Generate Analogues from YAML Config

Configure analogues in a YAML file (e.g., `configs/pap_analogues.yaml`):

```yaml
set_name: "pap_derivatives"
scaffold_smiles: "O=C([*:6])Nc1cc(Nc2nccc([*:4])n2)c([*:5])cc1"

substituents:
  1:
    "phenyl": "[*:1]c1ccccc1"
    "2-pyridyl": "[*:1]c1ccccn1"
  2:
    "ethyl": "[*:2]CC"
    "fluoro": "[*:2]F"
  3:
    "phenyl": "[*:3]c1ccccc1"

dummy_index_pairs:
  - [1, 4]
  - [2, 5]
  - [3, 6]

csv_dir: "data/analogue_sets"
```

Then run analogue generation:

```bash
python src/analogue_generation.py
# Enter config path: (eg. configs/pap_analogues.yaml)
```

This generates a CSV with all combinations.

### 2. Preparing Ligands for Docking

Convert SMILES to PDBQT for docking:

```python
from src.ligand_preparation import prepare_ligand

smiles = "CN1CCN(CC1)CC2=CC=C(C(NC3=CC(NC4=NC=CC(C5=CN=CC=C5)=N4)=C(C=C3)C)=O)C=C2"
prepared_pdbqt = prepare_ligand(
    smiles=smiles,
    name="Imatinib",
    output_dir="data/prepared_ligands"
)
print(f"Prepared ligand: {prepared_pdbqt}")
```

### 3. Calculate Binding Affinities

Use `VinaCalculator` docking:

```python
from src.vina_binding import VinaCalculator

calculator = VinaCalculator(
    protein_pdbqt="data/prepared_proteins/protein.pdbqt",
    center=(31.2, 33.5, 28.8),  # Binding pocket center
    size=(20, 20, 20),
    exhaustiveness=8
)

# Dock single ligand
result = calculator.calculate_binding("data/prepared_ligands/ligand.pdbqt")
print(f"Binding affinity: {result['affinity']} kcal/mol")

# Dock multiple ligands in parallel
ligand_dict = {
    "ligand1": "data/prepared_ligands/lig1.pdbqt",
    "ligand2": "data/prepared_ligands/lig2.pdbqt",
    "ligand3": "data/prepared_ligands/lig3.pdbqt"
}
results = calculator.dock_multiple(ligand_dict, use_multiprocessing=True)
for name, result in results.items():
    print(f"{name}: {result['affinity']} kcal/mol")
```

### 4. Extract Results

Parse binding affinities from Vina PDBQT output:

```python
from src.data_extraction import extract_top_binding

affinity = extract_top_binding("ligand_out.pdbqt")
print(f"Top binding mode: {affinity}")
```

## Testing

Run all tests:

```bash
pytest -v
```

Run specific test module:

```bash
pytest tests/test_analogue_generation.py -v
```

## Citation

> Inspiration: _Phenylamino-Pyrimidine (PAP) Derivatives: A New Class of Potent and Selective Inhibitors of Protein Kinase C (PKC)_, J. Zimmermann _et al_., 1996, DOI: 10.1002/ardp.19963290707

> Protein crystal structures: _*8UAK*_: _Crystal structure of the catalytic domain of human PKC alpha (D463N, V568I, S657E) in complex with Darovasertib (NVP-LXS196) at 2.82-A resolution_, *PDB DOI*: 10.2210/pdb8UAK/pdb, *Publication DOI*: 10.1021/acs.jmedchem.3c02002

## Status

✅ **Completed:**
- Scaffold-based analogue generation with YAML configuration
- Ligand preparation using Meeko (SMILES → PDBQT)
- AutoDock Vina integration (subprocess-based for cross-platform support)
- Binding affinity extraction from Vina PDBQT output
- Multiprocessing support for parallel docking (with Windows safety guards)
- Comprehensive test suite (32+ tests)

🔮 **Future:**
- Integration pipeline automation
- Performance benchmarking
- Bayesian optimisation loop integration

