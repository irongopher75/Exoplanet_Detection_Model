#!/usr/bin/env python3
"""
Orchestrator script to run the full exoplanet detection pipeline:
1. Download and process TESS data
2. Train the PINN model
3. Clean up raw and processed data to save space
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from src.utils.workflow_logging import setup_workflow_logging
from src.utils.workflow_state import WorkflowStateManager, WorkflowState, WorkflowStatus
from src.utils.retry import retry_with_backoff

def run_command(
    command: list, 
    description: str,
    capture_output: bool = False,
    timeout: int = None
) -> tuple[bool, str | None]:
    """Run command with better error handling."""
    logger = logging.getLogger("super_pipeline")
    logger.info(f"--- Starting: {description} ---")
    logger.info(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=capture_output,
            timeout=timeout
        )
        logger.info(f"--- Completed: {description} ---")
        if capture_output and result.stdout:
            logger.debug(f"Output: {result.stdout[:500]}")
        return True, result.stdout if capture_output else None
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout during {description} after {timeout}s")
        return False, None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during {description}: {e}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False, None
    except Exception as e:
        logger.error(f"Unexpected error during {description}: {e}", exc_info=True)
        return False, None

def main():
    parser = argparse.ArgumentParser(description="Full Exoplanet Detection Pipeline Orchestrator")
    
    # Download params
    parser.add_argument('--max-files', type=int, default=20, help='Max files to download and process')
    parser.add_argument('--sectors', type=str, help='Comma-separated list of TESS sectors to process')
    
    # Training params
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    
    # Cleanup params
    parser.add_argument('--no-cleanup', action='store_true', help='Skip the data cleanup step')
    parser.add_argument('--keep-raw', action='store_true', help='Keep raw FITS files (only clean processed NPZ)')
    
    args = parser.parse_args()
    
    # Standardized Logging
    logger = setup_workflow_logging(
        "super_pipeline", 
        log_dir=Path("outputs/logs"),
        log_level="INFO"
    )
    
    # State Management
    state_manager = WorkflowStateManager()
    state = WorkflowState(
        workflow_name="super_pipeline",
        status=WorkflowStatus.RUNNING,
        start_time=datetime.now().isoformat()
    )
    state_manager.save_state(state)

    logger.info("============================================================")
    logger.info("SUPER PIPELINE: STARTING END-TO-END WORKFLOW")
    logger.info("============================================================")

    try:
        # 1. DOWNLOAD AND PROCESS
        state.current_step = "Data Download and Processing"
        state_manager.save_state(state)
        
        download_cmd = [sys.executable, "scripts/process_all_tess.py", "--max-files", str(args.max_files)]
        if args.sectors:
            download_cmd.extend(["--sectors", args.sectors])
        
        success, _ = run_command(download_cmd, state.current_step)
        if not success:
            raise RuntimeError("Pipeline failed during download phase.")

        # 2. TRAINING
        state.current_step = "Model Training"
        state.progress = 0.5
        state_manager.save_state(state)
        
        train_cmd = [
            sys.executable, "scripts/run_training.py", 
            "--epochs", str(args.epochs), 
            "--batch-size", str(args.batch_size),
            "--data-dir", "data/processed/tess"
        ]
        
        success, _ = run_command(train_cmd, state.current_step)
        if not success:
            raise RuntimeError("Pipeline failed during training phase.")

        # 3. CLEANUP
        if not args.no_cleanup:
            state.current_step = "Data Cleanup"
            state_manager.save_state(state)
            
            cleanup_cmd = [sys.executable, "scripts/cleanup_data.py", "--processed", "--force"]
            if not args.keep_raw:
                cleanup_cmd.append("--raw")
                
            success, _ = run_command(cleanup_cmd, state.current_step)
            # Cleanup failure is not fatal
            if not success:
                logger.warning("Cleanup step encountered issues, but training was successful.")
        else:
            logger.info("Skipping cleanup step as requested.")

        state.status = WorkflowStatus.COMPLETED
        state.end_time = datetime.now().isoformat()
        state.progress = 1.0
        state_manager.save_state(state)
        
        logger.info("============================================================")
        logger.info("SUPER PIPELINE: ALL STEPS COMPLETE")
        logger.info("============================================================")

    except Exception as e:
        state.status = WorkflowStatus.FAILED
        state.error = str(e)
        state.end_time = datetime.now().isoformat()
        state_manager.save_state(state)
        logger.error(f"Pipeline Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
