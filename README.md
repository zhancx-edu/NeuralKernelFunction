# ST-NKF: Spatio-Temporal Neural Kernel Function for Earthquake Forecasting

ST-NKF model is a spatiotemporal Neural Point Processes for earthquake forecasting. It generalizes the Epidemic-Type Aftershock Sequence (ETAS) model by replacing parametric kernels with neural kernel functions, enabling flexible and data-driven modeling of seismicity.


---

![ST-NKF](method.jpg)

---

"""
## Setup

### 1. Clone the Repository

git clone https://github.com/zhancx-edu/NeuralKernelFunction.git
cd NeuralKernelFunction

---

### 2. Install Environment

We provide a conda environment configuration file (environment.yml) for reproducibility.

conda env create -f environment.yml
conda activate neuralkernelfunction

---

### 3. Run Experiments

All experiments are conducted using the notebook:

main.ipynb

---

### Step 1: Select Dataset

Choose a dataset from the EarthquakeNPP benchmark by setting:

dataset_name = "ComCat"

Available options:

["ComCat", "SaltonSea", "SanJac", "WHITE", "SCEDC_20", "SCEDC_25", "SCEDC_30"]

---

### Step 2: Configure Model Components

The model consists of three configurable components:

- Temporal kernel (temporal_id)
- Spatial kernel (spatial_id)
- Productivity function (kappa_id)

Each component can be set to:

- "empirical": classical ETAS kernel
- "neural": neural network-based kernel

---

### Example Configurations

#### ETAS Model (Fully Empirical)

temporal_id = "empirical"
spatial_id = "empirical"
kappa_id = "empirical"

---

#### ST-NKF Model (Fully Neural)

temporal_id = "neural"
spatial_id = "neural"
kappa_id = "neural"

---

#### Hybrid Models (Mixed Configuration)

temporal_id = "neural"
spatial_id = "empirical"
kappa_id = "neural"

This flexible design allows controlled comparisons between classical ETAS and neural kernel models.

---

### Step 3: Run the Notebook

After configuring the dataset and model, run all cells in:

main.ipynb

The pipeline will automatically:

- Load and preprocess the dataset
- Construct the point process model
- Train using log-likelihood optimization
- Evaluate performance on validation and test sets
"""


## Datasets

This study is conducted based on the benchmark datasets provided by the [EarthquakeNPP](https://github.com/ss15859/EarthquakeNPP), which is designed for evaluating Neural Point Process (NPP) models on earthquake forecasting tasks.

### EarthquakeNPP Benchmark

EarthquakeNPP is a comprehensive and standardized benchmark for earthquake forecasting, developed to facilitate fair and reproducible comparisons between ETAS and NPPs.


---

### Datasets Used in This Study

We adopt multiple datasets from EarthquakeNPP to evaluate the proposed ST-NKF model:

#### ComCat
- Source: USGS Advanced National Seismic System (ANSS) catalog  
- Region: California  
- Dataset: `ComCat_25`  
- Magnitude threshold: Mw ≥ 2.5  

#### SCEDC
- Source: Southern California Earthquake Data Center (SCEDC)  
- Datasets:
  - `SCEDC_20` (Mw ≥ 2.0)  
  - `SCEDC_25` (Mw ≥ 2.5)  
  - `SCEDC_30` (Mw ≥ 3.0)  

#### QTM (Template Matching Catalog)
- High-resolution catalog constructed via waveform template matching  
- Regions:
  - `SanJac_10` (San Jacinto fault zone)  
  - `SaltonSea_10`  
- Magnitude threshold: Mw ≥ 1.0  

#### WHITE
- High-resolution catalog focusing on the San Jacinto fault region  
- Dataset: `WHITE_06`  
- Magnitude threshold: Mw ≥ 0.6  


---

### Data Usage in This Work

The earthquake catalogs provided by EarthquakeNPP have already undergone standardized preprocessing procedures. These include:

- Projection of geographic coordinates into a planar coordinate system  
- Estimation of the magnitude of completeness for each catalog  
- Filtering of events to retain only earthquakes above the completeness threshold  

These preprocessing steps ensure data consistency and reliability across different datasets.

---

### Additional Processing: Spatial Perturbation

In this work, we further introduce a small spatial perturbation to events that share identical locations in the catalog.

In real earthquake catalogs, multiple events may be recorded with exactly the same coordinates due to limited spatial resolution or rounding during preprocessing. However, such duplicated locations can lead to numerical instability in spatial kernel estimation, especially for neural models.

To address this issue, we add a small uniform random perturbation to the longitude and latitude of duplicated events.

- Noise range: ±0.005 degrees (approximately ±0.5 km)  
- Applied only to events with identical spatial coordinates  
- Repeated until all duplicated locations are removed  

This perturbation is sufficiently small to preserve the physical structure of seismicity while improving numerical stability.

---

### Implementation

The following code illustrates the preprocessing procedure:

```python
import numpy as np

# Range of spatial noise (uniform distribution)
noise_range = 0.005  # degrees (~ ±0.5 km)

# Optional: time perturbation (in days, converted from seconds)
noise_time_range = 0.5 / 86400  

# Iterate until no duplicate (longitude, latitude) pairs remain
while True:
    # Identify duplicated spatial locations
    duplicates = df_modified[
        df_modified.duplicated(subset=['longitude', 'latitude'], keep=False)
    ]
    
    # Stop if no duplicates remain
    if duplicates.empty:
        print("All duplicated locations have been resolved.")
        break
    
    # Add uniform noise to duplicated entries
    for idx in duplicates.index:
        df_modified.loc[idx, 'longitude'] += np.random.uniform(-noise_range, noise_range)
        df_modified.loc[idx, 'latitude'] += np.random.uniform(-noise_range, noise_range)
    
    print(f"{len(duplicates)} duplicated points remain. Applying perturbation...")
```
---

## Model

The conditional intensity function is defined as:



Where:

- μ: background rate  
- κ(m): magnitude-dependent productivity  
- g(t): temporal kernel  
- f(x): spatial kernel  

---

### Model Components

| Component        | Empirical       | Neural           |
|------------------|----------------|------------------|
| Temporal kernel  | Omori law      | Neural network   |
| Spatial kernel   | ETAS spatial   | Neural network   |
| Productivity     | Exponential    | Neural network   |

---

### Neural Kernel Design

Neural kernels are implemented via:



This design provides:

- Flexible function approximation  
- Stable training behavior  
- Compatibility with likelihood-based learning  

---

## Usage

### Initialize Model

```python
from models import KernelPointProcess

model = KernelPointProcess(
    time_step_train=50,
    time_step_val=50,
    time_step_test=50,
    temporal_id="neural",   # or "empirical"
    spatial_id="neural",    # or "empirical"
    kappa_id="neural",      # or "empirical"
    global_m0=2.5,
    area=1.0,
    size_layer=3,
    size_nn=32
)