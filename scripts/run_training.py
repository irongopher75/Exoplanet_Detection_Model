#!/usr/bin/env python3
"""
Run PINN training on processed data.

This script loads processed light curves and trains the CombinedExoplanetModel.
"""

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging
from src.utils.seeding import set_all_seeds
from src.utils.config import load_config
from src.models.combined_model import CombinedExoplanetModel
from src.models.pinn import LightCurveDataset
from src.ingestion.standardize import StandardizedLightCurve
from src.training.trainer import PINNTrainer

def load_processed_data(data_dir: Path):
    """Load all .npz files from directory."""
    logger = logging.getLogger(__name__)
    files = list(data_dir.glob("**/*.npz"))
    logger.info(f"Found {len(files)} processed files in {data_dir}")
    
    light_curves = []
    skipped = 0
    
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            # Create a mock object with necessary attributes for Dataset
            class MockLC:
                pass
            
            lc = MockLC()
            lc.time = data['time']
            lc.flux = data['flux']
            lc.flux_err = data.get('flux_err')
            lc.metadata = {} # Metadata separate
            
            # Simple check
            if len(lc.time) > 10:
                light_curves.append(lc)
            else:
                skipped += 1
        except Exception:
            skipped += 1
            
    logger.info(f"Loaded {len(light_curves)} valid light curves (skipped {skipped})")
    return light_curves

def main():
    parser = argparse.ArgumentParser(description="Run PINN training")
    parser.add_argument('--config', type=str, default='configs/experiment.yaml', help='Config file')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--archive-dir', type=str, default=None, help='Directory to move processed files to after training')
    parser.add_argument('--test-dir', type=str, default='data/test', help='Directory to reserve test files')
    parser.add_argument('--n-test-files', type=int, default=2, help='Number of files to reserve for testing')
    
    args = parser.parse_args()
    
    setup_logging(log_dir=Path("outputs/logs"), log_file="training.log")
    logger = logging.getLogger(__name__)
    
    # Load data
    import numpy as np # Needed for loader
    
    # We use the dataset class from source
    # But since we have .npz files, we need to adapt them
    # For now, let's look at how run_full_training did it or use the Trainer directly
    
    # Actually, let's use the `src.main.train_pinn` if available, it's safer
    from src.main import train_pinn
    
    train_pinn(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        archive_dir=args.archive_dir,
        test_dir=args.test_dir,
        n_test_files=args.n_test_files,
        device="auto"
    )

if __name__ == "__main__":
    main()
