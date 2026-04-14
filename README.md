# ST-NKF: Spatio-Temporal Neural Kernel Point Process for Earthquake Forecasting

ST-NKF model is a spatiotemporal Neural Point Processes for earthquake forecasting. It generalizes the Epidemic-Type Aftershock Sequence (ETAS) model by replacing parametric kernels with neural kernel functions, enabling flexible and data-driven modeling of seismicity.


---

![ST-NKF](method.jpg)

---

## Setup

1. Clone the repository:

```bash
git clone https://github.com/zhancx-edu/NeuralKernelFunction.git
```

2. Navigate to the project directory:

```bash
cd ST-NKF
```

3. Install dependencies:

```bash
pip install tensorflow numpy pandas
```


## Datasets

This study is conducted based on the benchmark datasets provided by the [EarthquakeNPP](https://github.com/ss15859/EarthquakeNPP) framework, which is specifically designed for evaluating Neural Point Process (NPP) models on earthquake forecasting tasks.

### EarthquakeNPP Benchmark

EarthquakeNPP is a comprehensive and standardized benchmark for earthquake forecasting, developed to facilitate fair and reproducible comparisons between classical statistical models (e.g., ETAS) and modern neural point process models.

Key characteristics of EarthquakeNPP include:

- **Standardized data processing pipeline**  
  All datasets are derived from publicly available earthquake catalogs and undergo consistent preprocessing, including spatial, temporal, and magnitude filtering.

- **Diverse seismic regimes**  
  The datasets cover multiple regions in California, representing realistic operational forecasting scenarios.

- **Wide magnitude range**  
  Several datasets include low-magnitude earthquakes enabled by dense seismic networks and advanced detection techniques.

- **Benchmark compatibility**  
  The framework provides ready-to-use training, validation, and testing splits, enabling direct comparison across different models.

- **ETAS baseline integration**  
  EarthquakeNPP includes a reference implementation of the ETAS model, which is widely used in operational earthquake forecasting by government agencies.

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

#### ETAS (Synthetic Dataset)
- Simulated earthquake catalogs generated using the ETAS model  
- Datasets:
  - `ETAS_25`  
  - `ETAS_incomplete_25` (with missing events to mimic post-large-earthquake incompleteness)  

#### Japan_Deprecated
- Derived from the ANSS ComCat catalog  
- Included for comparison with previous NPP studies  

---

### Data Usage in This Work

In this study, all datasets are directly adopted from EarthquakeNPP without modification, ensuring:

- Fair comparison with existing NPP models  
- Consistency with prior benchmark studies  
- Reproducibility of experimental results  

The same data splits and preprocessing procedures as defined in EarthquakeNPP are used throughout all experiments.

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