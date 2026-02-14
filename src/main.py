"""
Main entry point for exoplanet detection pipeline.

This module provides command-line interfaces for:
- Data download
- Model training
- Evaluation
- Inference
"""

import argparse
import sys
from pathlib import Path
import logging
import yaml
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models.pinn import PINN, LightCurveDataset
from src.training.trainer import PINNTrainer
from src.ingestion.standardize import load_and_standardize
from src.utils.ledger import append_to_ledger


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def train_pinn(
    config_path: str,
    data_dir: str = "data/processed",
    output_dir: str = "outputs",
    archive_dir: str = None,
    test_dir: str = "data/test",
    n_test_files: int = 2,
    device: str = "auto",
    epochs: int = None,
    resume: bool = False
):
    """
    Train PINN model.
    
    Parameters
    ----------
    config_path : str
        Path to training configuration YAML.
    data_dir : str
        Directory containing processed light curves.
    output_dir : str
        Output directory for checkpoints and logs.
    archive_dir : str, optional
        Directory to move processed files to after training.
    test_dir : str
        Directory to save reserved test files for manual inspection.
    n_test_files : int
        Number of files to reserve for testing from this batch.
    device : str
        Device to use ('auto', 'cuda', 'cpu').
    epochs : int, optional
        Override epochs from config.
    resume : bool
        If True, attempts to resume from the latest checkpoint in output_dir/checkpoints.
    """
    import shutil
    import re
    logger = setup_logging()
    logger.info("Starting PINN training")
    
    # Load training config
    with open(config_path, 'r') as f:
        train_config = yaml.safe_load(f)
    
    # Setup device
    if device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger.info(f"Using device: {device}")

    # Create model first so we can load weights if resuming
    model = PINN(
        input_dim=3,
        encoder_dims=train_config.get('encoder_dims', [64, 128, 256]),
        encoder_kernels=train_config.get('encoder_kernels', [5, 5, 5]),
        param_head_dims=train_config.get('param_head_dims', [256, 128, 64]),
        dropout=train_config.get('dropout', 0.1)
    )
    
    # Resume logic
    checkpoint_to_load = None
    if resume:
        checkpoint_dir = Path(output_dir) / "checkpoints"
        if checkpoint_dir.exists():
            # Look for best_model.pt or the latest checkpoint
            checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if checkpoints:
                # Sort by epoch number
                def get_epoch(p):
                    match = re.search(r'epoch_(\d+)', p.name)
                    return int(match.group(1)) if match else 0
                
                checkpoints.sort(key=get_epoch, reverse=True)
                checkpoint_to_load = checkpoints[0]
            elif (checkpoint_dir / "best_model.pt").exists():
                checkpoint_to_load = checkpoint_dir / "best_model.pt"
            
    if checkpoint_to_load:
        logger.info(f"💾 Resuming from checkpoint: {checkpoint_to_load}")
    
    # Load data
    logger.info(f"Loading data from {data_dir}")
    data_path = Path(data_dir)
    
    # Find all light curve files
    lc_files = list(data_path.glob("**/*.npz"))
    
    # Filter by ledger to ensure we only use "new" data
    from src.utils.ledger import load_ledger
    ledger = load_ledger()
    initial_count = len(lc_files)
    lc_files = [f for f in lc_files if f.stem not in ledger]
    
    if initial_count > len(lc_files):
        logger.info(f"Filtered out {initial_count - len(lc_files)} files already present in the ledger.")
        
    logger.info(f"Found {len(lc_files)} new light curve files")
    
    if len(lc_files) == 0:
        logger.error("No new light curve files found. Run data download first.")
        return
    
    # Load and standardize light curves
    light_curves = []
    loaded_files = []
    for lc_file in lc_files:  # Process all available files
        try:
            lc = load_and_standardize(lc_file)
            light_curves.append(lc)
            loaded_files.append(lc_file)
        except Exception as e:
            logger.warning(f"Failed to load {lc_file}: {e}")
    
    logger.info(f"Loaded {len(light_curves)} light curves")
    
    # Record these files in the ledger to ensure they are marked as 'consumed'
    if loaded_files:
        append_to_ledger([f.stem for f in loaded_files])
    
    if len(light_curves) == 0:
        logger.error("No valid light curves loaded.")
        return
        
    # Reserve test data
    test_lcs = []
    test_files = []
    
    if n_test_files > 0:
        import random
        # Select indices to reserve for testing
        test_indices = random.sample(range(len(light_curves)), min(n_test_files, len(light_curves)))
        test_indices.sort(reverse=True) # Sort reverse to pop correctly
        
        for idx in test_indices:
            test_lcs.append(light_curves.pop(idx))
            test_files.append(loaded_files.pop(idx))
            
        logger.info(f"Reserved {len(test_files)} light curves for manual testing in {test_dir}")
    
    # Create datasets
    n_total = len(light_curves)
    train_size = max(1, int(0.8 * n_total)) if n_total > 0 else 0
    val_size = n_total - train_size
    
    train_lcs = light_curves[:train_size]
    val_lcs = light_curves[train_size:]
    
    train_dataset = LightCurveDataset(
        train_lcs,
        max_length=train_config.get('max_length', 1000),
        normalize_time=True
    )
    val_dataset = LightCurveDataset(
        val_lcs,
        max_length=train_config.get('max_length', 1000),
        normalize_time=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get('batch_size', 32),
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.get('batch_size', 32),
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create trainer
    # Merge overrides into config
    if epochs:
        train_config['epochs'] = epochs
        
    trainer = PINNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=Path(output_dir) / "checkpoints",
        logger=logger,
        config=train_config
    )
    
    # Load checkpoint if requested
    if checkpoint_to_load:
        trainer.load_checkpoint(checkpoint_to_load.name)
    
    # Train
    trainer.train(
        epochs=epochs if epochs is not None else train_config.get('epochs', 200),
        save_every=train_config.get('save_every', 10)
    )
    
    logger.info("Training complete")
    
    # Handle Test Data Reservation
    if test_files and test_dir:
        test_path = Path(test_dir)
        test_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Storing {len(test_files)} test datasets in {test_path}")
        
        for file_path in test_files:
            try:
                dest_path = test_path / file_path.name
                shutil.move(str(file_path), str(dest_path))
                # Also move metadata if it exists
                meta_file = file_path.with_suffix('.json')
                if meta_file.exists():
                    shutil.move(str(meta_file), str(test_path / meta_file.name))
            except Exception as e:
                logger.error(f"Failed to store test file {file_path}: {e}")

    # Archive remaining files if requested
    if archive_dir:
        archive_path = Path(archive_dir)
        archive_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Archiving remaining {len(loaded_files)} files to {archive_path}")
        
        for file_path in loaded_files:
            try:
                # Maintain relative structure if possible, or just flat move
                # For simplicity, let's flat move but handle name collisions
                dest_path = archive_path / file_path.name
                if dest_path.exists():
                     # timestamp if exists
                    import time
                    timestamp = int(time.time())
                    dest_path = archive_path / f"{file_path.stem}_{timestamp}{file_path.suffix}"
                
                shutil.move(str(file_path), str(dest_path))
                # Also move metadata if it exists
                meta_file = file_path.with_suffix('.json')
                if meta_file.exists():
                    # If the npz file was renamed, we should rename the json too
                    if dest_path.name != file_path.name:
                        new_meta_name = f"{Path(dest_path).stem}.json"
                        shutil.move(str(meta_file), str(archive_path / new_meta_name))
                    else:
                        shutil.move(str(meta_file), str(archive_path / meta_file.name))
            except Exception as e:
                logger.error(f"Failed to archive {file_path}: {e}")
        
        logger.info("Archival complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Exoplanet Detection Pipeline"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train PINN model')
    train_parser.add_argument(
        '--archive-dir',
        type=str,
        default=None,
        help='Directory to move processed files to after training'
    )
    train_parser.add_argument(
        '--test-dir',
        type=str,
        default='data/test',
        help='Directory to save reserved test files for manual inspection'
    )
    train_parser.add_argument(
        '--n-test-files',
        type=int,
        default=2,
        help='Number of files to reserve for testing from each batch'
    )
    train_parser.add_argument(
        '--config',
        type=str,
        default='configs/training.yaml',
        help='Training configuration file'
    )
    train_parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing processed light curves'
    )
    train_parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for checkpoints'
    )
    train_parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs to train (overrides config)'
    )
    train_parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from the latest checkpoint'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_pinn(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            archive_dir=args.archive_dir,
            test_dir=args.test_dir,
            n_test_files=args.n_test_files,
            device=args.device,
            epochs=args.epochs if hasattr(args, 'epochs') else None,
            resume=args.resume
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
