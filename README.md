# Exoplanet Detection System: Instrument-Aware Inference

> **“We keep exoplanet detection models aligned with aging telescopes.”**

## 🧠 One-Paragraph Summary

You are building a system that keeps exoplanet-detection models scientifically valid as telescopes age. Telescopes slowly change how they measure starlight due to sensor degradation, noise drift, and calibration shifts. Detection models trained on old data silently lose accuracy, missing real planets or hallucinating false ones. Your system treats the telescope itself as a time-evolving physical system, continuously adapting the detection model to the telescope’s current behavior while enforcing real orbital physics so the model cannot invent planets. The result is long-term, reliable exoplanet detection that does not decay as hardware ages.

---

## 🎯 The Problem

*   **Telescopes Age & Drift**: Hardware degrades, thermal properties shift, and sensors accumulate damage (e.g., Pixel Sensitivity Dropouts).
*   **Stationarity Fallacy**: Standard ML assumes data distribution $P(X)$ is constant. For space missions, $P(X_{year1}) \neq P(X_{year4})$.
*   **Silent Failure**: Models trained on clean, early-mission data maintain high confidence scores even as they become miscalibrated on noisy late-mission data.
*   **Physics Violation**: "Black box" models often hallucinate planets in noise because they don't understand Keplerian mechanics.

## 💡 The Solution: Instrument-Aware PINNs

We combine three innovations to solve this:

1.  **Physics-Informed Neural Networks (PINNs)**: We don't just predict "Planet/No Planet". We predict orbital parameters ($P, t_0, R_p/R_s$) and use a **differentiable Mandel-Agol transit model** in the loss function. The model *must* find a physically valid orbit to explain the data.
2.  **Calibration Network**: A secondary network explicitly learns the time-varying systematics (drift, aging), effectively "cleaning" the data before the PINN sees it.
3.  **Adaptive Calibration**: The system can be updated on new validation data to learn the *current* state of the instrument without forgetting the *invariant* physics of transits.

---

## 🧪 Scientific Proof (Hackathon Demo)

We have built a demonstration suite to prove this phenomenon and our solution.

### 1. The "Aging" Proof (Simulation)
Running this script simulates a 4-year mission where noise levels drift. It trains a "Static" model (Year 1 only) and our "Adaptive" model.
**Mac/Linux**
```bash
python3 scripts/demonstrate_aging.py
```

**Windows**
```powershell
python scripts\demonstrate_aging.py
```
**Outcome**: You will see a plot showing the Static Model's error exploding over time, while the Adaptive Model remains stable.

### 2. Interactive Dashboard
Explore the effects of telescope aging on detection confidence in real-time.
**Mac/Linux**
```bash
python3 -m streamlit run scripts/dashboard.py
```

**Windows**
```powershell
streamlit run scripts\dashboard.py
```
**Features**:
*   **Drift Slider**: Manually "age" the telescope and watch detection confidence drop.
*   **Physics Overlay**: See exactly what the PINN "thinks" the physical transit looks like compared to the noisy data.

### 3. Full Evaluation
Run the model on the full validation dataset (~1000 light curves) to get quantitative metrics.
**Mac/Linux**
```bash
python3 scripts/run_evaluation.py --config configs/experiment.yaml
```

**Windows**
```powershell
python scripts\run_evaluation.py --config configs\experiment.yaml
```

---

## 🔧 System Architecture

| Component | Responsibility | Technical Implementation |
| :--- | :--- | :--- |
| **`src/models/pinn.py`** | Physics-Informed Core | Encoder -> Attention -> Params -> **Mandel-Agol Loss** |
| **`src/models/calibration_net.py`** | Drift Correction | Dense Network estimating $f_{noise}(t)$ |
| **`src/ingestion/standardize.py`** | Data Normalization | Converts TESS/Kepler FITS to unified `StandardizedLightCurve` |
| **`scripts/process_all_tess.py`** | Data Pipeline | Mass scaling script for TESS data retrieval |

---

## 🚀 Reproduction Steps

**1. Install Dependencies**
```bash
pip install -r requirements.txt
pip install streamlit matplotlib lightkurve
```

**2. Download Data**
(The repository comes with scripts to fetch huge datasets)
**Mac/Linux**
```bash
python3 scripts/process_all_tess.py --max-files 1000
```

**Windows**
```powershell
python scripts\process_all_tess.py --max-files 1000
```

**3. Train Model**
**Mac/Linux**
```bash
python3 scripts/run_training.py --epochs 20 --data-dir data/processed --output-dir outputs
```

**Windows**
```powershell
python scripts\run_training.py --epochs 20 --data-dir data\processed --output-dir outputs
```

**4. Automated Super Pipeline (Recommended)**
Run the entire download, processing, and training workflow in a single command:
**Mac/Linux**
```bash
python3 scripts/super_pipeline.py --max-files 1000 --epochs 200
```

**Windows**
```powershell
python scripts\super_pipeline.py --max-files 1000 --epochs 200
```

**5. Monitor Data Ledger**
The system maintains a persistent ledger of all processed datasets to prevent redundant work. You can check the number of processed files or view the latest entries:

**Mac/Linux**
```bash
# Total count
wc -l data/data_ledger.txt
# Last 10 entries
tail data/data_ledger.txt
```

**Windows**
```powershell
# Total count
(Get-Content data\data_ledger.txt).Count
# Last 10 entries
Get-Content data\data_ledger.txt -Tail 10
```

**6. View Results**
Check `outputs/logs/super_pipeline_*.log` for high-level orchestration or `outputs/logs/process_all_tess.log` for real-time data status. Launch the dashboard to evaluate detected candidates.
