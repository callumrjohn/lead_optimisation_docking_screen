"""Test multiprocessing with VinaCalculator."""

import sys
import time
from pathlib import Path
from multiprocessing import freeze_support

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.vina_binding import VinaCalculator


def main():
    # Test parameters
    protein = "data/prepared_proteins/prkca_exp_8uak_clean.pdbqt"
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

    # Prepare multiple ligands (same ligand for testing)
    ligands = {
        "imatinib_1": "data/prepared_ligands/imatinib.pdbqt",
        "imatinib_2": "data/prepared_ligands/imatinib.pdbqt",
        "imatinib_3": "data/prepared_ligands/imatinib.pdbqt",
    }

    print("=" * 60)
    print("SEQUENTIAL DOCKING (use_multiprocessing=False)")
    print("=" * 60)
    start = time.time()
    results_seq = calculator.dock_multiple(ligands, use_multiprocessing=False)
    seq_time = time.time() - start

    print(f"\nSequential results:")
    for name, result in results_seq.items():
        print(f"  {name}: {result['affinity']:.3f} kcal/mol" if result['success'] else f"  {name}: FAILED")
    print(f"Time: {seq_time:.2f}s\n")

    print("=" * 60)
    print("PARALLEL DOCKING (use_multiprocessing=True, num_processes=2)")
    print("=" * 60)
    start = time.time()
    results_par = calculator.dock_multiple(ligands, use_multiprocessing=True, num_processes=2)
    par_time = time.time() - start

    print(f"\nParallel results:")
    for name, result in results_par.items():
        print(f"  {name}: {result['affinity']:.3f} kcal/mol" if result['success'] else f"  {name}: FAILED")
    print(f"Time: {par_time:.2f}s\n")

    print("=" * 60)
    print(f"Speedup: {seq_time/par_time:.2f}x (sequential: {seq_time:.2f}s, parallel: {par_time:.2f}s)")
    print("=" * 60)


if __name__ == "__main__":
    freeze_support()
    main()
