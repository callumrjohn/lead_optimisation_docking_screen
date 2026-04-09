# Lead-optimisation Docking Screen

Computational workflow for AutoDock Vina binding simulations and multi-objective Bayesian optimization for target binding and off-target selectivity.

This project provides tools for **analogue generation**, **ligand & protein preparation**, **binding affinity calculations**, and **multi-objective Bayesian optimization** with parallel processing support.


> 💡 Built as a fun personal project to explore using some open-source tools in the context of lead-optimisation.


## Overview

This repository provides a complete workflow for computational drug discovery:
- **Analogue generation**: Systematic creation of compound variants through scaffold-based substitution (YAML-configured)
- **Ligand preparation**: Prepare ligands from SMILES using Meeko for docking studies
- **Binding affinity calculation**: Dock ligands to proteins using AutoDock Vina with parallel processing
- **Multi-objective Bayesian optimization**: Optimize for simultaneous maximization of target binding affinity and selectivity over off-targets
- **Results extraction**: Parse binding affinities from Vina output

## Project Structure

```
├── src/
│   ├── analogue_generation.py       # Scaffold-based compound generation
│   ├── ligand_preparation.py        # Molecular structure preparation via Meeko
│   ├── protein_preparation.py       # Protein structure preparation via Meeko
│   ├── vina_binding.py              # Binding affinity calculations with AutoDock Vina
│   ├── bo.py                        # Multi-objective Bayesian optimization loop
│   ├── data_extraction.py           # Parse results from Vina output
│   └── utils/
│       └── config.py                # YAML configuration utilities
├── data/
│   ├── protein_structures/          # Input protein PDB files
│   ├── prepared_proteins/           # Meeko-prepared protein PDBQT structures
│   ├── prepared_ligands/            # Meeko-prepared ligand PDBQT files
│   └── analogue_sets/               # Generated analogue CSV files
├── configs/
│   ├── analogue_generator.yaml      # Configuration for analogue generation
│   ├── pap_analogues.yaml           # PAP derivatives configuration
│   ├── protein_configs/             # Target and off-target protein configs
│   └── vina_binding.yaml            # AutoDock Vina simulation parameters
├── tests/
│   ├── test_analogue_generation.py
│   ├── test_ligand_preparation.py
│   ├── test_bo.py
│   └── conftest.py
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
    "methyl": "[*:2]C"
    "chloro": "[*:2]Cl"
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
```

**Required inputs (provided interactively):**
  - Path to analogue generation YAML file

This generates a CSV with all combinations.

### 2. Preparing Proteins for Docking

Convert protein PDB structures to PDBQT format for docking:

```bash
python src/protein_preparation.py
```

**Required inputs (provided interactively):**
- Path to protein PDB file
- Name for the protein

### 3. Preparing Ligands for Docking (Optional)

Convert SMILES to PDBQT for docking:

```bash
python src/ligand_preparation.py
```

**Required inputs (provided interactively):**
- SMILES string of the ligand
- Name for the ligand

### 4. Calculate Binding Affinities

Configure protein docking parameters in a YAML file (e.g., `configs/protein.yaml`):

```yaml
receptor: "data/prepared_proteins/proteins.pdbqt"
center: [31.2, 33.5, 28.8]  # Example binding pocket center (X, Y, Z)
size: [20, 20, 20]           # Search box dimensions (X, Y, Z)
```

Configure Vina simulation parameters in `configs/vina_binding.yaml`:

```yaml
simulation_parameters:
  cpu: 4
  exhaustiveness: 8
  max_evals: 0
  seed: 42
  num_modes: 9
  min_rmsd: 1.0
  energy_range: 3.0
  spacing: 0.375
  verbosity: 1
```

Then run docking:

```bash
python src/vina_binding.py
```

**Required inputs (provided interactively):**
- Path to protein config YAML file
- Path to ligand PDBQT file OR SMILES string of ligand

### 5. Multi-Objective Bayesian Optimization

Optimize compounds for simultaneous **target binding affinity** and **selectivity** against off-targets using Bayesian optimization with Gaussian Process surrogates and Expected Improvement acquisition functions.

**How it works:**
1. **Initial sampling**: MaxMin algorithm selects diverse initial compounds from a pool based on Tanimoto similarity
2. **Objective functions**:
   - Target binding score = -affinity (minimize affinity for tighter binding)
   - Selectivity score = mean(off-target affinities) - target affinity (higher selectivity = weaker off-target binding)
   - Composite objective = 0.5 × target + 0.5 × selectivity
3. **Surrogate models**: Fits separate GP models with Tanimoto kernels for target and selectivity
4. **Acquisition function**: Balances Expected Improvement with Tanimoto similarity penalty to encorage analogue exploration.
5. **Iterative refinement**: Selects new batches of candidates until evaluation budget is exhausted

**Configuration files required:**

Target protein config (`configs/target_protein.yaml`):
```yaml
receptor: "data/prepared_proteins/target_clean.pdbqt"
center: [31.2, 33.5, 28.8]  # Binding pocket center
size: [20, 20, 20]           # Search box dimensions
```

Off-target protein configs (`configs/offtarget_1.yaml`, etc.):
```yaml
receptor: "data/prepared_proteins/offtarget_clean.pdbqt"
center: [30.0, 32.0, 29.0]
size: [20, 20, 20]
```

Vina parameters (`configs/vina_binding.yaml`):
```yaml
simulation_parameters:
  cpu: 4
  exhaustiveness: 8
  ...
```

**Usage:**

```python
from src.bo import BayesianOptimizer

optimizer = BayesianOptimizer(
    protein_configs=[
        "configs/target_protein.yaml",
        "configs/offtarget_1.yaml",
        "configs/offtarget_2.yaml"
    ],
    vina_config="configs/vina_binding.yaml"
)

results = optimizer.run_optimization_loop(
    initial_smiles_file="data/analogue_sets/compounds.csv",
    initial_sample_size=5,
    total_budget=50,
    batch_size=5
)

print(results.head())
results.to_csv("bo_results.csv", index=False)
```

**Output:**
- CSV with ranked compounds sorted by composite score
- Columns: `smiles`, `batch`, `target_affinity`, `selectivity`, `composite_score`

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

> Inspiration for analogue generation: _Potent and selective inhibitors of the Abl-kinase: phenylamino-pyrimidine (PAP) derivatives_, J. Zimmermann _et al_., 1997, DOI: 10.1016/S0960-894X(96)00601-4

> Protein crystal structure: _*8UAK*_: _Crystal structure of the catalytic domain of human PKC alpha (D463N, V568I, S657E) in complex with Darovasertib (NVP-LXS196) at 2.82-A resolution_, *PDB DOI*: 10.2210/pdb8UAK/pdb, *Publication DOI*: 10.1021/acs.jmedchem.3c02002

## Status

✅ **Implemented:**
- Analogue generation, ligand & protein preparation
- Parallel docking with AutoDock Vina
- Multi-objective Bayesian optimization
- Comprehensive test suite

🔮 **Future:**
- End-to-end pipeline automation

