# Lead Optimisation Docking Screen

This is a personal mini-project to recreate (in a simplified way) a lead optimisation campaign that led to the discovery of **Imatinib** using a few _in silico_ methods. It is inspired by a 1996 SAR study on PAP derivatives (DOI: 10.1002/ardp.19963290707).

Right now, this repo includes tools for: **analogue generation** and **ligand preparation**.


## Overview

This repository currently provides tools and workflows for:
- **Analogue generation**: Systematic creation of compound variants through scaffold-based substitution
- **Ligand preparation**: Prepare ligands from SMILES using Meeko for binding studies

## Project Structure

```
├── src/
│   ├── analogue_generation.py      # Scaffold-based compound generation
│   ├── ligand_preparation.py        # Molecular structure preparation
│   └── __pycache__/
├── data/
│   ├── protein_structures/          # Input protein PDB files
│   ├── prepared_proteins/           # Prepared protein structures
│   └── prepared_ligands/            # Output ligand files
├── tests/
│   ├── test_analogue_generation.py
│   ├── test_ligand_preparation.py
│   ├── conftest.py
│   └── __pycache__/
├── configs/
│   └── analogue_generator.yaml      # Configuration for analogue generation
├── utils/
│   └── config.py                    # Configuration utilities
├── environment.yml                   # Conda environment specification
├── pytest.ini                        # Pytest configuration
└── README.md
```

## Installation

### Requirements
- Python 3.11+
- Conda (Miniconda or Anaconda)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd lead_optimisation_docking_screening
```

2. Create the conda environment:
```bash
conda env create -f environment.yaml
conda activate <env-name>
```

3. Verify installation by running tests:
```bash
pytest
```

## Usage

### Generate Analogues

Create scaffold-based compound variants for screening:

```python
from src.analogue_generation import generate_combination_df

template = "[*:1]c1ccccc1"  # Scaffold SMILES with dummy atoms
substituents = {
    1: {"methyl": "[*:2]C", "ethyl": "[*:2]CC"}
}
dummy_pairs = [(1, 2)]

df = generate_combination_df(
    template=template,
    substituents=substituents,
    dummy_index_pairs=dummy_pairs,
    combination_csv_path="data/generated_analogues.csv"
)
```

### Prepare Ligands

Process molecular structures for computational workflows:

```python
from src.ligand_preparation import prepare_ligand

prepare_ligand(
    smiles="CCc1ccccc1",
    output_dir="data/prepared_ligands"
)
```

## Citation

This work is based on the SAR campaign described in:

> Original reference: _Phenylamino-Pyrimidine(PAP) Derivatives: A New Class of Potent andSelective Inhibitors of Protein Kinase C (PKC)_, J. Zimmermann _et al_., 1996, DOI: 10.1002/ardp.19963290707

## Status

Initial commit - Analogue generation and ligand preparation modules with testing.

