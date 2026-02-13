"""
Utility to maintain a persistent ledger of processed and trained data files.
This ensures that data is not double-counted or re-processed even after cleanup.
"""

from pathlib import Path
from typing import Set, Union, List

def get_ledger_path() -> Path:
    """Get the path to the data ledger file."""
    # Place it in data/ directory as it's part of the data state
    ledger_path = Path("data/data_ledger.txt")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    return ledger_path

def load_ledger() -> Set[str]:
    """Load all file stems from the ledger."""
    path = get_ledger_path()
    if not path.exists():
        return set()
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}

def append_to_ledger(stems: Union[str, List[str]]):
    """Append new stems to the ledger."""
    if not stems:
        return
        
    if isinstance(stems, str):
        stems = [stems]
    
    # Get currently loaded to avoid duplicates if possible
    current = load_ledger()
    
    path = get_ledger_path()
    with open(path, "a") as f:
        for stem in stems:
            if stem not in current:
                f.write(f"{stem}\n")
                current.add(stem) # Local update to avoid double writing in same call
