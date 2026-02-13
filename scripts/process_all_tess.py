#!/usr/bin/env python3
"""
Process all TESS .sh files and download/process all light curves.
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import List, Optional
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging
from src.utils.seeding import set_all_seeds
from scripts.download_from_urls import (
    load_urls_from_metadata,
    download_fits_file,
    process_fits_file
)
from src.utils.ledger import load_ledger, append_to_ledger
import shutil


def cleanup_fully_processed_metadata(metadata_dir: Path, ledger_files: set):
    """Move metadata files that only contain already-processed URLs to a subfolder."""
    logger = logging.getLogger(__name__)
    scanned_dir = metadata_dir / "Scanned_URLs"
    
    # Get all top-level metadata files
    from scripts.download_from_urls import extract_filename_from_url
    import json
    
    metadata_files = [f for f in metadata_dir.glob("*.json") if f.is_file()]
    if not metadata_files:
        return
        
    moved_count = 0
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            urls = data.get('urls', [])
            if not urls:
                continue
                
            # Check if all URLs in this file are in the ledger
            all_in_ledger = True
            for url in urls:
                stem = Path(extract_filename_from_url(url)).stem
                if stem not in ledger_files:
                    all_in_ledger = False
                    break
            
            if all_in_ledger:
                scanned_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(metadata_file), str(scanned_dir / metadata_file.name))
                moved_count += 1
                
        except Exception as e:
            logger.warning(f"Failed to check/move {metadata_file}: {e}")
            
    if moved_count > 0:
        logger.info(f"Optimization: Moved {moved_count} fully-processed metadata files to {scanned_dir}")


def process_all_sh_files(
    raw_dir: Path,
    metadata_dir: Path,
    processed_dir: Path,
    max_files: Optional[int] = None,
    sectors: Optional[List[int]] = None,
    resume: bool = True
):
    """Process all TESS .sh files and download/process all URLs."""
    logger = logging.getLogger(__name__)
    
    # Load ledger and clean up metadata files first for speed
    existing_files = set()
    ledger_files = set()
    if resume:
        # Search recursively in processed, archive, and test
        dirs_to_check = [
            processed_dir,
            Path("data/archive"),
            Path("data/test")
        ]
        
        for d in dirs_to_check:
            if d.exists():
                existing_npz = list(d.glob("**/*.npz"))
                existing_files.update({f.stem for f in existing_npz})
        
        # Also check the persistent ledger
        ledger_files = load_ledger()
        existing_files.update(ledger_files)
                
        logger.info(f"Found {len(existing_files)} already processed/archived/tested/ledger files. These will be skipped.")
        
        # Optimization: Move fully-processed metadata files to a subfolder
        cleanup_fully_processed_metadata(metadata_dir, ledger_files)
    
    # Load remaining URLs from active metadata files
    logger.info("Loading URLs from active metadata files...")
    all_urls = load_urls_from_metadata(metadata_dir, sectors=sectors)
    
    if len(all_urls) == 0:
        logger.error("No URLs found in metadata files.")
        return
    
    logger.info(f"Found {len(all_urls):,} total URLs to check against ledger")
    
    # Create timestamped batch directory
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = processed_dir / f"batch_{timestamp}"
    
    # Create directories
    raw_tess_dir = raw_dir / "tess"
    raw_tess_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Update processed_dir to the batch directory for the actual save operation
    target_processed_dir = batch_dir
    
    # Process URLs
    successful_downloads = 0
    successful_processing = 0
    skipped = 0
    failed = 0
    
    logger.info("Starting download and processing...")
    
    for url_info in tqdm(all_urls, desc="Processing TESS files"):
        if max_files is not None and successful_processing >= max_files:
            logger.info(f"Reached limit of {max_files} processed files. Stopping.")
            break
        
        url = url_info['url']
        
        # Extract expected filename
        from scripts.download_from_urls import extract_filename_from_url
        expected_filename = extract_filename_from_url(url)
        expected_stem = Path(expected_filename).stem
        
        # Skip if already processed in THIS directory
        if resume and expected_stem in existing_files:
            skipped += 1
            continue
        
        # Download FITS file
        fits_path = download_fits_file(url, raw_tess_dir)
        if fits_path is None:
            failed += 1
            continue
        
        successful_downloads += 1
        
        # Process FITS file
        processed_path = process_fits_file(fits_path, target_processed_dir)
        if processed_path is None:
            failed += 1
            continue
        
        # Record in ledger immediately upon successful processing
        append_to_ledger(expected_stem)
        
        successful_processing += 1
        
        # Log progress periodically
        if (successful_processing + failed + skipped) % 100 == 0:
            logger.info(
                f"Progress: {successful_processing} processed, "
                f"{skipped} skipped, {failed} failed, "
                f"{len(all_urls) - (successful_processing + failed + skipped)} remaining"
            )
    
    # Final summary
    logger.info("=" * 60)
    logger.info("Processing Complete!")
    logger.info("=" * 60)
    logger.info(f"Total URLs processed: {len(all_urls):,}")
    logger.info(f"  Successful downloads: {successful_downloads:,}")
    logger.info(f"  Successful processing: {successful_processing:,}")
    logger.info(f"  Skipped (already exists): {skipped:,}")
    logger.info(f"  Failed: {failed:,}")
    logger.info(f"")
    logger.info(f"Processed files saved to: {target_processed_dir}")
    logger.info(f"Raw FITS files saved to: {raw_tess_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Process all TESS .sh files and download/process all light curves"
    )
    parser.add_argument(
        '--raw-dir',
        type=str,
        default='data/raw',
        help='Raw data directory'
    )
    parser.add_argument(
        '--metadata-dir',
        type=str,
        default='data/metadata',
        help='Directory containing metadata files'
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed/tess',
        help='Directory to save processed light curves'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to process (None = all)'
    )
    parser.add_argument(
        '--sectors',
        type=str,
        default=None,
        help='Comma-separated list of sectors to process (e.g., "1,2,3")'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all files'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume - reprocess all files'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(
        log_dir=Path("outputs/logs"),
        log_level="INFO",
        log_file="process_all_tess.log"
    )
    
    # Set seed
    set_all_seeds(args.seed)
    
    # Parse sectors
    sectors = None
    if args.sectors:
        sectors = [int(s.strip()) for s in args.sectors.split(',')]
    
    # Process all files
    process_all_sh_files(
        raw_dir=Path(args.raw_dir),
        metadata_dir=Path(args.metadata_dir),
        processed_dir=Path(args.processed_dir),
        max_files=args.max_files,
        sectors=sectors,
        resume=not args.no_resume
    )
    
    logger.info("All processing complete!")


if __name__ == '__main__':
    main()
