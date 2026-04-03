"""Test the updated VinaCalculator class."""

from src.vina_binding import VinaCalculator

# Test parameters
protein = "data/prepared_proteins/prkca_exp_8uak_clean.pdbqt"
ligand = "data/prepared_ligands/imatinib.pdbqt"
center = (31.2, -7.3, 12.7)
size = (20.0, 20.0, 20.0)

# Initialize calculator
calculator = VinaCalculator(
    protein_pdbqt=protein,
    center=center,
    size=size,
    exhaustiveness=8,
    num_modes=9
)

print(f"Calculator initialized with {protein}")
print(f"Center: {center}, Size: {size}\n")

# Run single docking
print("Running docking...")
result = calculator.calculate_binding(ligand, name="imatinib")

print(f"Success: {result['success']}")
print(f"Affinity: {result['affinity']:.3f} kcal/mol")
print(f"All modes: {result.get('modes', [])}")
print(f"Message: {result['message']}")
print(f"Output: {result.get('output_pdbqt', 'N/A')}")
