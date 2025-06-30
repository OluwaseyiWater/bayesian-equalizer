# Bayesian Nonparametric Equalizer

This project implements a Bayesian nonparametric channel equalization pipeline using:

* **Dirichlet Process (DP)** priors over delay taps
* **Gaussian Process (GP)** priors over path amplitudes
* **Sequential Monte Carlo (SMC)** inference

It supports simulation, training, and evaluation of delay–Doppler fading channels under variable complexity and pilot configurations.

---

## 📁 Project Structure

```
src/
├── models/             # DP and GP priors
├── inference/          # SMC filter, resampling, prior samplers
├── training/           # Main scripts for training and evaluation
├── utils/              # Simulators, plotting, metrics, logging
experiments/            # Notebooks and sweep experiments
configs/                # YAML config file
```

---

## 🚀 Getting Started

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Run a Training Script

```bash
python src/training/train_smc.py
```

### 3. Run Sweeps

```bash
python src/training/complexity_sweep.py
python src/training/pilot_vs_ber.py
```

### 4. Visualize Delay–Doppler Channel

```python
from src.utils.plotting import plot_delay_doppler_surface
plot_delay_doppler_surface(true_gains, time_points=train_times)
```

---

## ⚙️ Configuration

Use YAML config files to set parameters:

```yaml
# configs/default.yaml
num_particles: 100
snr_db: 20
num_symbols: 100
pilot_fracs: [0.0, 0.05, 0.1, 0.2]
resample_method: systematic
```

Then load with:

```python
from omegaconf import OmegaConf
cfg = OmegaConf.load("configs/default.yaml")
```

---

## 📊 Logging and Results

* ✅ Supports **Weights & Biases** (W\&B)
* ✅ Also logs to CSV in `logs/`

---
