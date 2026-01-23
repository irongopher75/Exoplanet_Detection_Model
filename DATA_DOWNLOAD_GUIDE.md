# 📥 Data Download Guide

## Current Status: **NO DATA DOWNLOADED YET**

The data directories are currently empty. You need to download data before training.

---

## 🎯 Quick Start: Download Data

### Step 1: Configure Target IDs

Edit `configs/data.yaml` and add target IDs:

**For Kepler data:**
```yaml
photometry:
  kepler:
    enabled: true
    targets:
      kic_ids: [11442793, 11442794, 11442795]  # Add your KIC IDs here
```

**For TESS data:**
```yaml
photometry:
  tess:
    enabled: true
    targets:
      tic_ids: [25155310, 25155311]  # Add your TIC IDs here
```

### Step 2: Run Download Script

```bash
python scripts/download_data.py --config configs/data.yaml
```

### Step 3: Verify Data

Check that files were downloaded:
```bash
ls data/raw/kepler/
ls data/processed/kepler/
```

---

## 📊 Data Sources

### 1. **Kepler Space Telescope** (via MAST)
- **Source:** NASA MAST Archive
- **Access:** Via `lightkurve` package
- **Format:** FITS files
- **What you get:** Light curves for exoplanet-hosting stars
- **Example KIC IDs:** 
  - 11442793 (Kepler-10)
  - 11442794 (Kepler-11)
  - 11442795 (Kepler-12)

### 2. **TESS** (via MAST)
- **Source:** NASA MAST Archive  
- **Access:** Via `lightkurve` package
- **Format:** FITS files
- **What you get:** Light curves from TESS mission
- **Example TIC IDs:**
  - 25155310
  - 25155311

### 3. **NASA Exoplanet Archive**
- **Source:** https://exoplanetarchive.ipac.caltech.edu
- **Access:** REST API
- **Format:** CSV/JSON
- **What you get:** Confirmed exoplanet catalog with parameters
- **No configuration needed** - downloads automatically if enabled

### 4. **Radial Velocity Data** (Optional)
- **Source:** ESO Archives
- **Access:** Via `astroquery`
- **Status:** Disabled by default
- **Requires:** Star names or coordinates

### 5. **Gaia Stellar Parameters** (Optional)
- **Source:** Gaia Archive
- **Access:** Via `astroquery`
- **Status:** Disabled by default
- **Requires:** Coordinates from photometry data

---

## 🔍 Finding Target IDs

### Finding Kepler KIC IDs:
1. Visit: https://exoplanetarchive.ipac.caltech.edu
2. Search for confirmed exoplanets
3. Look for "KIC" column
4. Or use known exoplanet systems:
   - Kepler-10: KIC 11442793
   - Kepler-11: KIC 11442794
   - Kepler-12: KIC 11442795

### Finding TESS TIC IDs:
1. Visit: https://tess.mit.edu/science/tess-input-catalog/
2. Or use TESS exoplanet catalog
3. Look for "TIC" column

---

## 📝 Example Configuration

Here's a complete example `configs/data.yaml` with sample targets:

```yaml
global:
  data_root: "data"
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  metadata_dir: "data/metadata"
  logs_dir: "outputs/logs"
  retry:
    max_attempts: 3
    backoff_factor: 2.0
    timeout_seconds: 300
  show_progress: true
  log_level: "INFO"
  seed: 42  # For reproducibility

photometry:
  kepler:
    enabled: true
    targets:
      kic_ids: [11442793, 11442794, 11442795]  # Example: Kepler-10, 11, 12
    data_products:
      - "SAP"
      - "PDCSAP"
    cadence: "auto"
    quality_filtering: true
  
  tess:
    enabled: true
    targets:
      tic_ids: [25155310, 25155311]  # Example TIC IDs
    data_products:
      - "SAP"
      - "PDCSAP"
    cadence: "auto"
    quality_filtering: true

exoplanet_archive:
  enabled: true
  api_base_url: "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
  tables:
    - name: "exoplanets"
      endpoint: "exoplanets"
    - name: "composite"
      endpoint: "exoplanets"
      format: "composite"
```

---

## 🚀 Download Commands

### Download all sources:
```bash
python scripts/download_data.py --config configs/data.yaml
```

### Download specific sources:
```bash
python scripts/download_data.py --config configs/data.yaml --sources kepler
python scripts/download_data.py --config configs/data.yaml --sources tess exoplanet_archive
```

### Skip cached files (re-download):
```bash
python scripts/download_data.py --config configs/data.yaml --skip-cache
```

---

## 📂 Data Directory Structure

After downloading, your structure will look like:

```
data/
├── raw/
│   ├── kepler/
│   │   ├── kic_11442793_PDCSAP_q01.fits
│   │   ├── kic_11442793_PDCSAP_q02.fits
│   │   └── ...
│   └── tess/
│       └── ...
├── processed/
│   ├── kepler/
│   │   ├── kic_11442793_PDCSAP_q01.npz
│   │   └── ...
│   └── tess/
│       └── ...
└── metadata/
    ├── kepler.json
    ├── tess.json
    ├── exoplanet_archive.json
    └── download_summary.json
```

---

## ⚠️ Important Notes

1. **No data is pre-downloaded** - You must run the download script
2. **Target IDs required** - You must specify KIC or TIC IDs in config
3. **Internet required** - Downloads from NASA MAST archive
4. **Time required** - Large datasets can take hours to download
5. **Storage space** - Each light curve is ~1-5 MB, plan accordingly

---

## 🔧 Troubleshooting

### "No targets specified" error:
- Add KIC IDs or TIC IDs to `configs/data.yaml`
- Or disable unused sources (`enabled: false`)

### Download fails:
- Check internet connection
- Verify target IDs exist
- Check logs in `outputs/logs/`

### Empty directories after download:
- Check download logs for errors
- Verify target IDs are correct
- Some targets may not have data available

---

## 📚 Additional Resources

- **Kepler Data:** https://archive.stsci.edu/kepler/
- **TESS Data:** https://archive.stsci.edu/tess/
- **Exoplanet Archive:** https://exoplanetarchive.ipac.caltech.edu
- **Lightkurve Docs:** https://docs.lightkurve.org/

---

**Next Steps:** After downloading data, you can run training:
```bash
python src/main.py train --config configs/training.yaml
```



