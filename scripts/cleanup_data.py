#!/usr/bin/env python3
"""
Cleanup utility to delete raw and processed data files to save space.
"""

import argparse
import sys
from pathlib import Path
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def cleanup_files(directory: Path, patterns: list, dry_run: bool = True):
    logger = logging.getLogger(__name__)
    if not directory.exists():
        logger.warning(f"Directory {directory} does not exist. Skipping.")
        return 0, 0

    total_deleted = 0
    total_size = 0
    
    files_to_delete = []
    for pattern in patterns:
        files_to_delete.extend(list(directory.glob(pattern)))

    # Filter out .sh files explicitly as a safety measure
    files_to_delete = [f for f in files_to_delete if f.suffix != '.sh']

    if not files_to_delete:
        logger.info(f"No files found matching patterns in {directory}")
        return 0, 0

    for f in files_to_delete:
        if f.is_file():
            size = f.stat().st_size
            if dry_run:
                logger.info(f"[DRY RUN] Would delete: {f} ({size / 1024 / 1024:.2f} MB)")
            else:
                try:
                    f.unlink()
                    logger.info(f"Deleted: {f}")
                except Exception as e:
                    logger.error(f"Failed to delete {f}: {e}")
                    continue
            total_deleted += 1
            total_size += size

    return total_deleted, total_size

def main():
    parser = argparse.ArgumentParser(description="Cleanup data files to save space.")
    parser.add_argument('--raw', action='store_true', help='Clean up raw FITS files in data/raw')
    parser.add_argument('--processed', action='store_true', help='Clean up processed NPZ files in data/processed')
    parser.add_argument('--force', action='store_true', help='Actually delete files (without this, it only performs a dry run)')
    
    args = parser.parse_args()
    logger = setup_logging()

    if not any([args.raw, args.processed]):
        parser.print_help()
        sys.exit(0)

    dry_run = not args.force
    if dry_run:
        logger.info("--- DRY RUN MODE (No files will be deleted) ---")

    grand_total_files = 0
    grand_total_size = 0

    if args.raw:
        logger.info("Cleaning up raw data...")
        count, size = cleanup_files(Path("data/raw"), ["**/*.fits"], dry_run)
        grand_total_files += count
        grand_total_size += size

    if args.processed:
        logger.info("Cleaning up processed data...")
        count, size = cleanup_files(Path("data/processed"), ["**/*.npz"], dry_run)
        grand_total_files += count
        grand_total_size += size

    logger.info("=" * 40)
    status = "Would delete" if dry_run else "Deleted"
    logger.info(f"Cleanup Summary: {status} {grand_total_files} files, Totaling {grand_total_size / 1024 / 1024:.2f} MB")
    
    if dry_run and grand_total_files > 0:
        logger.info("Run again with --force to actually delete these files.")

if __name__ == "__main__":
    main()
