from pathlib import Path
import torch
from typing import Optional

def validate_data_directory(data_dir: Path, min_files: int = 1) -> bool:
    """Validate that data directory has required files."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
    npz_files = list(data_dir.glob("**/*.npz"))
    if len(npz_files) < min_files:
        raise ValueError(
            f"Data directory {data_dir} has only {len(npz_files)} files, "
            f"minimum {min_files} required"
        )
    return True

def validate_model_checkpoint(checkpoint_path: Path) -> bool:
    """Validate model checkpoint exists and is loadable."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    try:
        # Load on CPU for validation
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        required_keys = ['model_state_dict', 'epoch']
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Checkpoint missing required key: {key}")
        return True
    except Exception as e:
        raise ValueError(f"Invalid checkpoint file: {e}")
