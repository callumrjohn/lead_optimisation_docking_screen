"""
Pytest configuration and shared fixtures.
"""

import sys
from pathlib import Path

# Add root directory to Python path for imports
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))
