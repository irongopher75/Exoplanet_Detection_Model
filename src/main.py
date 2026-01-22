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
    device: str = "auto"
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
    device : str
        Device to use ('auto', 'cuda', 'cpu').
    """
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
    
    # Load data
    logger.info(f"Loading data from {data_dir}")
    data_path = Path(data_dir)
    
    # Find all light curve files
    lc_files = list(data_path.glob("**/*.npz"))
    logger.info(f"Found {len(lc_files)} light curve files")
    
    if len(lc_files) == 0:
        logger.error("No light curve files found. Run data download first.")
        return
    
    # Load and standardize light curves
    light_curves = []
    for lc_file in lc_files[:100]:  # Limit for now
        try:
            lc = load_and_standardize(lc_file)
            light_curves.append(lc)
        except Exception as e:
            logger.warning(f"Failed to load {lc_file}: {e}")
    
    logger.info(f"Loaded {len(light_curves)} light curves")
    
    if len(light_curves) == 0:
        logger.error("No valid light curves loaded.")
        return
    
    # Create datasets
    train_size = int(0.8 * len(light_curves))
    val_size = len(light_curves) - train_size
    
    train_lcs = light_curves[:train_size]
    val_lcs = light_curves[train_size:]
    
    train_dataset = LightCurveDataset(
        train_lcs,
        max_length=train_config.get('max_length', 1000),
        normalize=True
    )
    val_dataset = LightCurveDataset(
        val_lcs,
        max_length=train_config.get('max_length', 1000),
        normalize=True
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
    
    # Create model
    model = PINN(
        input_dim=3,
        encoder_dims=train_config.get('encoder_dims', [64, 128, 256]),
        encoder_kernels=train_config.get('encoder_kernels', [5, 5, 5]),
        param_head_dims=train_config.get('param_head_dims', [256, 128, 64]),
        dropout=train_config.get('dropout', 0.1)
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create trainer
    trainer = PINNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=Path(output_dir) / "checkpoints",
        logger=logger,
        config=train_config
    )
    
    # Train
    trainer.train(
        epochs=train_config.get('epochs', 200),
        save_every=train_config.get('save_every', 10)
    )
    
    logger.info("Training complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Exoplanet Detection Pipeline"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train PINN model')
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
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_pinn(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
