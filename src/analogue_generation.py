import pandas as pd
from itertools import product
from typing import Iterable, Tuple
from rdkit import Chem
from src.utils.config import load_config

def find_dummy_by_mapnum(mol: Chem.Mol, mapnum: int):
    """
    Find the dummy atom in a molecule with a specific atom map number.

    Args:
        mol: RDKit molecule object
        mapnum: The atom map number to search for (e.g., 1, 2, 3, etc.)

    Returns:
        RDKit atom object: The dummy atom with the specified map number
    """

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == mapnum:
            return atom
    raise ValueError(f"Could not find dummy atom with map number {mapnum}")


def get_number_of_dummy_atoms(mol: Chem.Mol) -> int:

    """
    Count the number of dummy atoms (atomic number 0) in a molecule.
    
    Args:
        mol: RDKit molecule object
    
    Returns:
        int: The number of dummy atoms in the molecule
    """
    
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            count += 1
    return count

def pool_substituents(scaffold: Chem.Mol, substituents) -> Chem.Mol:

    """
    Combine scaffold + substituents without connecting or removing dummy atoms for further processing.
    The number of substituents must match the number of dummy atoms in the scaffold, and each substituent must have exactly one dummy atom.

    Args:
        scaffold: RDKit molecule object representing the scaffold with dummy atoms
        substituents: RDKit molecule object of list of RDKit molecule objects representing the substituent(s) with dummy atoms

    Returns:
        RDKit molecule object representing an unbonded combination of the scaffold and substituents
    """
    
    substituents = [substituents] if isinstance(substituents, Chem.Mol) else substituents

    # Ensure all substituents have exactly one dummy atom
    for sub in substituents:
        if get_number_of_dummy_atoms(sub) != 1:
            raise ValueError("Each substituent must have exactly one dummy atom")
    
    # Ensure scaffold has the correct number ofdummy atoms
    if get_number_of_dummy_atoms(scaffold) != len(substituents):
        raise ValueError(f"Scaffold must have exactly {len(substituents)} dummy atoms")

    mol_pool = scaffold
    for sub in substituents:
        mol_pool = Chem.CombineMols(mol_pool, sub)
    return mol_pool


def combine_and_connect(mol_pool, dummy_index_pairs: Iterable[Tuple[int, int]]):
    """
    From a combined molecule pool of scaffold and substituents, connect the substituents to the scaffold.
    Forms bonds between the dummy atoms according to the provided pairs, then removes all dummy atoms.
    If a substituent dummy has no neighbors (bare dummy), it represents no substitution (hydrogen).

    Args:
        mol_pool: RDKit molecule object representing the combined scaffold and substituents with dummy atoms
        dummy_index_pairs: Pairs of dummy atom indices to connect

    Returns:
        RDKit molecule object of the final combined molecule 

    """

    # Use EditableMol to modify the molecule
    em = Chem.EditableMol(mol_pool)
    
    dummy_pairs = [(find_dummy_by_mapnum(mol_pool, i), find_dummy_by_mapnum(mol_pool, j)) for i, j in dummy_index_pairs]

    atoms_to_remove = []

    for scaffold_dummy, sub_dummy in dummy_pairs:
        sub_neighbors = sub_dummy.GetNeighbors()
        scaffold_neighbors = scaffold_dummy.GetNeighbors()
        
        if len(scaffold_neighbors) != 1:
            raise ValueError(
                f"Dummy atom *:{scaffold_dummy.GetAtomMapNum()} does not have exactly one neighbour"
            )
        
        # Check if substituent dummy has neighbors
        if len(sub_neighbors) == 0:
            # Bare dummy represents no substitution (hydrogen) - just remove both dummies
            pass
        elif len(sub_neighbors) == 1:
            # Normal case: connect the two neighbors of the dummies
            to_bond1 = sub_neighbors[0].GetIdx()
            to_bond2 = scaffold_neighbors[0].GetIdx()
            em.AddBond(to_bond1, to_bond2, Chem.BondType.SINGLE)
        else:
            raise ValueError(
                f"Dummy atom *:{sub_dummy.GetAtomMapNum()} has {len(sub_neighbors)} neighbours, expected 0 or 1"
            )

        atoms_to_remove.extend([sub_dummy.GetIdx(), scaffold_dummy.GetIdx()])

    # Remove dummy atoms in descending index order
    for idx in sorted(set(atoms_to_remove), reverse=True):
        em.RemoveAtom(idx)

    mol = em.GetMol()
    
    Chem.SanitizeMol(mol)
    
    return mol

def generate_combination_df(template: str, 
                            substituents: dict, 
                            dummy_index_pairs: Iterable[Tuple[int, int]],
                            combination_csv_path: str = "data/generated_analogues.csv"
                            ) -> pd.DataFrame:

    """
    Generate all combinations of substituents on a given scaffold template.
    
    Args:
        template: SMILES string of the scaffold with dummy atoms
        substituents: Dictionary where keys are dummy atom map numbers and values are dictionaries of substituent name to SMILES
        dummy_index_pairs: Pairs of dummy atom indices to connect
        combination_csv_path: Path to save the generated combinations as a CSV file
    
    Returns:
        DataFrame with columns for each substituent position and a column for the final combined SMILES
    """

    df_rows = []
    errors = 0

    for substituent_combination in product(*[substituents[i].items() for i in sorted(substituents.keys())]):
        try:
            base = Chem.MolFromSmiles(template)
            substituent_mols = [Chem.MolFromSmiles(smi) for _, smi in substituent_combination]

            # Pool the scaffold and substituents first
            pooled = pool_substituents(base, substituent_mols)
            
            # Then connect them
            final_mol = combine_and_connect(pooled, dummy_index_pairs=dummy_index_pairs)
            
            canonical_smiles = Chem.MolToSmiles(final_mol)

            df_rows.append([template] + [name for name, _ in substituent_combination] + [canonical_smiles])

        except Exception as e:
            errors += 1
            # Print error message for debugging
            print(f"Error processing combination {[name for name, _ in substituent_combination]}: {e}")

    df = pd.DataFrame(df_rows, columns=["scaffold"] + [f"R{i}" for i in range(1, len(substituents) + 1)] + ["analogue_smiles"])
    df.to_csv(combination_csv_path, index=False)

    print(f"Generated {len(df)} molecules with {errors} errors.")

    return df

def main():

    config_path = input("Enter the path to the configuration file: ")
    
    config = load_config([config_path])

    set_name = config.get("set_name", "analogue_set")
    scaffold_smiles = config.get("scaffold_smiles")
    substituent_dict = config.get("substituents", {})
    
    dummy_index_pairs = config.get("dummy_index_pairs", [])
    combination_csv_path = config.get("csv_dir", "data/analogue_sets") + f"/{set_name}_combinations.csv"

    # Calculate total combinations: multiply the number of options at each position
    from functools import reduce
    from operator import mul
    num_combinations_per_position = [len(substituent_dict[i]) for i in sorted(substituent_dict.keys())]
    number_of_combinations = reduce(mul, num_combinations_per_position, 1)

    print("=" * 60)
    print(f"Generating {number_of_combinations} analogue combinations for set: {set_name}")
    print("=" * 60)

    generate_combination_df(scaffold_smiles, 
                            substituent_dict, 
                            dummy_index_pairs, 
                            combination_csv_path)
    
if __name__ == "__main__":
    main()