from pathlib import Path
import re


def extract_top_binding(pdbqt_path: str) -> float:
    """
    Extract the top (best) binding affinity from Vina output PDBQT file.
    
    Args:
        pdbqt_path: Path to the Vina output PDBQT file
        
    Returns:
        Binding affinity (kcal/mol) of the top pose
        
    Raises:
        ValueError: If no VINA RESULT line is found
    """
    content = Path(pdbqt_path).read_text()
    
    # Find the first REMARK VINA RESULT line
    match = re.search(r'REMARK VINA RESULT:\s+([-\d.]+)', content)
    
    if not match:
        raise ValueError(f"No 'REMARK VINA RESULT:' found in {pdbqt_path}")
    
    return float(match.group(1))
