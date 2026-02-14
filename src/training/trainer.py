"""
Training infrastructure for PINN models.

This module provides the main training loop, checkpointing, logging,
and integration with physics-informed losses.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import logging
import json
from tqdm import tqdm
import numpy as np

from ..physics.pinn_losses import PhysicsInformedLoss, compute_pinn_loss
from ..training.losses import CombinedLoss


class PINNTrainer:
    """
    Trainer for Physics-Informed Neural Networks.
    
    Handles:
    - Training loop with physics constraints
    - Validation and checkpointing
    - Learning rate scheduling
    - Early stopping
    - Logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        output_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize trainer.
        
        Parameters
        ----------
        model : nn.Module
            PINN model to train.
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader, optional
            Validation data loader.
        loss_fn : nn.Module, optional
            Loss function. If None, creates default physics-informed loss.
        optimizer : optim.Optimizer, optional
            Optimizer. If None, creates Adam optimizer.
        scheduler : optim.lr_scheduler, optional
            Learning rate scheduler.
        device : torch.device, optional
            Device to train on. If None, uses CUDA if available.
        output_dir : Path, optional
            Directory for checkpoints and logs.
        logger : logging.Logger, optional
            Logger instance.
        config : Dict[str, Any], optional
            Training configuration.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir or Path('outputs/checkpoints')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup loss function
        if loss_fn is None:
            physics_loss = PhysicsInformedLoss(
                data_weight=self.config.get('data_weight', 1.0),
                physics_weight=self.config.get('physics_weight', 0.1),
                kepler_weight=self.config.get('kepler_weight', 0.1),
                duration_weight=self.config.get('duration_weight', 0.05),
                reg_weight=self.config.get('reg_weight', 0.01)
            )
            self.loss_fn = CombinedLoss(physics_loss)
        else:
            self.loss_fn = loss_fn
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-3)
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        self.scheduler = scheduler
        if self.scheduler is None:
            sched_type = self.config.get('scheduler', 'plateau')
            if sched_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.get('epochs', 200)
                )
            elif sched_type == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=self.config.get('lr_factor', 0.5),
                    patience=self.config.get('lr_patience', 10)
                )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Setup adaptive weighting if enabled
        self.adaptive_weighting = None
        if self.config.get('adaptive_weighting', True):
            from .adaptive_weighting import AdaptiveLossWeighting
            # We track these keys from PhysicsInformedLoss output
            loss_keys = ['data_loss', 'parameter_bounds', 'keplers_law', 'duration', 'regularization']
            self.adaptive_weighting = AdaptiveLossWeighting(
                loss_keys=loss_keys,
                init_weights={
                    'data_loss': self.config.get('data_weight', 1.0),
                    'parameter_bounds': self.config.get('physics_weight', 0.1),
                    'keplers_law': self.config.get('kepler_weight', 0.1),
                    'duration': self.config.get('duration_weight', 0.05),
                    'regularization': self.config.get('reg_weight', 0.01)
                }
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of average losses.
        """
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in pbar:
            # Move to device
            time = batch['time'].to(self.device)
            flux = batch['flux'].to(self.device)
            flux_err = batch.get('flux_err')
            if flux_err is not None:
                flux_err = flux_err.to(self.device)
            
            # Forward pass
            output = self.model(time, flux, flux_err)
            parameters = output['parameters']
            
            # Generate physics model
            from ..physics.pinn_losses import mandel_agol_transit_torch
            predicted_flux = mandel_agol_transit_torch(
                time=time,
                period=parameters['period'],
                t0=parameters['t0'],
                rp_rs=parameters['rp_rs'],
                a_rs=parameters['a_rs'],
                b=parameters['b'],
                u1=parameters.get('u1', torch.tensor(0.3, device=self.device)),
                u2=parameters.get('u2', torch.tensor(0.3, device=self.device))
            )
            
            # Compute loss components
            loss_dict = self.loss_fn(
                predicted_flux=predicted_flux,
                target_flux=flux,
                parameters=parameters,
                time=time,
                flux_err=flux_err
            )
            
            # Extract flat loss components for weighting
            flat_loss_dict = {
                'data_loss': loss_dict['data_loss'],
                'parameter_bounds': loss_dict['physics_losses']['parameter_bounds'],
                'keplers_law': loss_dict['physics_losses']['keplers_law'],
                'duration': loss_dict['physics_losses']['duration'],
                'regularization': loss_dict['physics_losses']['regularization']
            }
            
            self.optimizer.zero_grad()
            
            if self.adaptive_weighting:
                # Update weights and get total weighted loss
                self.adaptive_weighting.update_weights(flat_loss_dict, self.model)
                loss = self.adaptive_weighting.apply_weights(flat_loss_dict)
            else:
                loss = loss_dict['total_loss']
            
            # Backward and step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value.item()
            
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Average losses
        avg_losses = {key: value / n_batches for key, value in loss_components.items()}
        avg_losses['total'] = total_loss / n_batches
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of validation losses.
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        loss_components = {}
        n_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                time = batch['time'].to(self.device)
                flux = batch['flux'].to(self.device)
                flux_err = batch.get('flux_err')
                if flux_err is not None:
                    flux_err = flux_err.to(self.device)
                
                # Forward pass
                output = self.model(time, flux, flux_err)
                parameters = output['parameters']
                
                # Generate physics model
                from ..physics.pinn_losses import mandel_agol_transit_torch
                predicted_flux = mandel_agol_transit_torch(
                    time=time,
                    period=parameters['period'],
                    t0=parameters['t0'],
                    rp_rs=parameters['rp_rs'],
                    a_rs=parameters['a_rs'],
                    b=parameters['b'],
                    u1=parameters.get('u1', torch.tensor(0.3, device=self.device)),
                    u2=parameters.get('u2', torch.tensor(0.3, device=self.device))
                )
                
                # Compute loss
                loss_dict = self.loss_fn(
                    predicted_flux=predicted_flux,
                    target_flux=flux,
                    parameters=parameters,
                    time=time,
                    flux_err=flux_err
                )
                
                loss = loss_dict['total_loss']
                
                # Accumulate
                total_loss += loss.item()
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        if key not in loss_components:
                            loss_components[key] = 0.0
                        loss_components[key] += value.item()
                
                n_batches += 1
        
        # Average losses
        avg_losses = {key: value / n_batches for key, value in loss_components.items()}
        avg_losses['total'] = total_loss / n_batches
        
        return avg_losses
    
    def train(self, epochs: int, save_every: int = 10) -> None:
        """
        Train model for multiple epochs.
        
        Parameters
        ----------
        epochs : int
            Number of epochs to train.
        save_every : int
            Save checkpoint every N epochs.
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            # Validate
            if self.val_loader is not None:
                val_losses = self.validate()
                self.val_losses.append(val_losses)
                val_loss = val_losses.get('total', float('inf'))
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                    self.logger.info(f"New best validation loss: {val_loss:.6f}")
            else:
                val_losses = {}
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if self.val_loader is not None else train_losses['total'])
                else:
                    self.scheduler.step()
            
            # Logging
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_losses['total']:.6f} - "
                f"Val Loss: {val_losses.get('total', 'N/A')}"
            )
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        self.logger.info("Training complete")
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.
        
        Parameters
        ----------
        filename : str
            Checkpoint filename.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        self.logger.debug(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, filename: str) -> None:
        """
        Load model checkpoint.
        
        Parameters
        ----------
        filename : str
            Checkpoint filename.
        """
        checkpoint_path = self.output_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        try:
            # Try strict loading first
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded checkpoint strictly: {checkpoint_path}")
        except RuntimeError as e:
            self.logger.warning(f"⚠️ Architecture mismatch detected: {e}")
            self.logger.info("Attempting to load compatible weights (non-strict)...")
            
            # Load what we can (e.g. parameter heads)
            msg = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.logger.info(f"Loaded compatible weights. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
            
            # Re-initialize optimizer/scheduler since architecture changed
            self.logger.info("Resetting optimizer and scheduler for new architecture...")
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-3)
            )
            if self.scheduler is not None:
                # Re-create scheduler based on type
                sched_type = self.config.get('scheduler', 'plateau')
                if sched_type == 'plateau':
                    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer,
                        mode='min',
                        factor=self.config.get('lr_factor', 0.5),
                        patience=self.config.get('lr_patience', 10)
                    )
            # We DON'T load the optimizer state as it won't match the new parameters
        
        # Load other state
        if 'optimizer_state_dict' in checkpoint and 'RuntimeError' not in locals():
             try:
                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
             except:
                 self.logger.warning("Could not load optimizer state (likely due to architecture change).")

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint and 'RuntimeError' not in locals():
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                self.logger.warning("Could not load scheduler state.")
        
        self.logger.info(f"Successfully resumed from epoch {self.current_epoch}")
