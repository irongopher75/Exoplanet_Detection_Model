#!/usr/bin/env python3
"""
Download and process TESS light curves from extracted URLs.

This script:
1. Reads URLs from metadata files
2. Downloads TESS FITS files from URLs
3. Processes FITS files into standardized light curves
4. Saves processed data for training
"""

import argparse
import sys
from pathlib import Path
import json
import logging
import random
import requests
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import time
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging
from src.utils.seeding import set_all_seeds
from src.ingestion.standardize import LightCurveStandardizer, StandardizedLightCurve


def load_urls_from_metadata(metadata_dir: Path, sectors: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Load URLs from metadata files."""
    logger = logging.getLogger(__name__)
    
    all_urls = []
    
    # Find all TESS metadata files
    # Use glob pattern that only matches top-level files to allow moving processed files to a subfolder
    metadata_files = [f for f in metadata_dir.glob("tesscurl_sector_*_metadata.json") if f.is_file()]
    
    if not metadata_files:
        # Fallback: look for ANY top-level json that might contain URLs
        metadata_files = [f for f in metadata_dir.glob("*.json") if f.is_file()]
    
    logger.info(f"Found {len(metadata_files)} metadata files")
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            # Check if this is a URL list format
            sector = data.get('sector')
            
            # Filter by sector if specified
            if sectors is not None and sector is not None and sector not in sectors:
                continue
            
            urls = data.get('urls', [])
            if not urls and isinstance(data, list):
                # Maybe it's a list of metadata objects?
                pass
            
            for url in urls:
                all_urls.append({
                    'url': url,
                    'sector': sector,
                    'source_file': data.get('source_file', ''),
                    'metadata_file': str(metadata_file)
                })
        
        except Exception as e:
            logger.warning(f"Failed to load {metadata_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(all_urls)} URLs")
    return all_urls


def extract_filename_from_url(url: str) -> str:
    """Extract filename from TESS URL."""
    # URL format: .../tess2019253231442-s0016-0000000249722255-0152-s_lc.fits
    if 'uri=' in url:
        # Extract the URI part
        uri_part = url.split('uri=')[-1]
        filename = uri_part.split('/')[-1]
    else:
        filename = url.split('/')[-1]
    
    return filename


def download_fits_file(url: str, output_dir: Path, timeout: int = 300) -> Optional[Path]:
    """Download a single FITS file from URL."""
    logger = logging.getLogger(__name__)
    
    try:
        filename = extract_filename_from_url(url)
        output_path = output_dir / filename
        
        # Skip if already exists
        if output_path.exists():
            return output_path
        
        # Download with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout, stream=True)
                response.raise_for_status()
                
                # Save file
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                return output_path
            
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    # logger.warning(f"Download failed: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to download {url}: {e}")
                    return None
        
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return None


def process_fits_file(fits_path: Path, output_dir: Path) -> Optional[Path]:
    """Process FITS file into standardized light curve."""
    logger = logging.getLogger(__name__)
    
    try:
        # Import lightkurve only when needed
        try:
            import lightkurve as lk
        except ImportError:
            logger.error("lightkurve required")
            return None
        
        # Load light curve
        lc = lk.read(fits_path)
        
        # Extract data
        time = lc.time.value
        
        # Try PDCSAP first, then SAP, then default
        if hasattr(lc, 'pdcsap_flux') and lc.pdcsap_flux is not None:
            flux = lc.pdcsap_flux.value
            flux_err = lc.pdcsap_flux_err.value if hasattr(lc, 'pdcsap_flux_err') else None
        elif hasattr(lc, 'sap_flux') and lc.sap_flux is not None:
            flux = lc.sap_flux.value
            flux_err = lc.sap_flux_err.value if hasattr(lc, 'sap_flux_err') else None
        else:
            flux = lc.flux.value
            flux_err = lc.flux_err.value if hasattr(lc, 'flux_err') else None
        
        # Extract quality flags
        quality = lc.quality.value if hasattr(lc, 'quality') else np.zeros(len(time), dtype=int)
        
        # Extract metadata
        metadata = {
            'source': 'tess',
            'sector': getattr(lc, 'sector', None),
            'camera': getattr(lc, 'camera', None),
            'ccd': getattr(lc, 'ccd', None),
            'cadence': getattr(lc, 'cadence', None),
            'targetid': getattr(lc, 'targetid', None),
            'mission': 'TESS',
            'fits_file': str(fits_path)
        }
        
        # Create standardized light curve
        light_curve = {
            'time': time,
            'flux': flux,
            'flux_err': flux_err,
            'quality': quality,
            'metadata': metadata
        }
        
        # Standardize
        standardizer = LightCurveStandardizer()
        standardized_lc = standardizer.standardize(light_curve, mission='tess')
        
        # Save processed data
        output_filename = fits_path.stem + '.npz'
        output_path = output_dir / output_filename
        np.savez(
            output_path,
            time=standardized_lc.time,
            flux=standardized_lc.flux,
            flux_err=standardized_lc.flux_err,
            quality=standardized_lc.quality if standardized_lc.quality is not None else np.zeros_like(standardized_lc.time, dtype=int)
        )
        
        # Save metadata to data/metadata - Use unique name to avoid conflicts
        metadata_dir = Path("data/metadata")
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / f"{output_path.stem}_processed_meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(standardized_lc.metadata, f, indent=2, default=str)
        
        return output_path
    
    except Exception as e:
        logger.error(f"Failed to process {fits_path}: {e}")
        return None


def download_and_process_urls(
    urls: List[Dict[str, Any]],
    raw_dir: Path,
    processed_dir: Path,
    max_files: Optional[int] = None,
    random_sample: bool = False
):
    """Download and process URLs."""
    logger = logging.getLogger(__name__)
    
    # Limit URLs if specified
    if max_files is not None:
        if random_sample:
            urls = random.sample(urls, min(max_files, len(urls)))
        else:
            urls = urls[:max_files]
    
    logger.info(f"Processing {len(urls)} URLs")
    
    # Create directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and process
    successful_downloads = 0
    successful_processing = 0
    failed = 0
    
    for url_info in tqdm(urls, desc="Downloading and processing"):
        url = url_info['url']
        
        # Download FITS file
        fits_path = download_fits_file(url, raw_dir)
        if fits_path is None:
            failed += 1
            continue
        
        successful_downloads += 1
        
        # Process FITS file
        processed_path = process_fits_file(fits_path, processed_dir)
        if processed_path is None:
            failed += 1
            continue
        
        successful_processing += 1
    
    logger.info(f"Download complete:")
    logger.info(f"  Successful downloads: {successful_downloads}")
    logger.info(f"  Successful processing: {successful_processing}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Processed files saved to: {processed_dir}")


def main():
    pass # Main logic usually in process_all_tess.py now if calling functions directly

if __name__ == '__main__':
    main()
