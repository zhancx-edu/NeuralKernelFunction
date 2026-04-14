# ST-NKF: Spatio-Temporal Neural Kernel Point Process for Earthquake Forecasting

ST-NKF is a unified framework for earthquake forecasting based on **Neural Point Processes (NPPs)**. It generalizes the classical Epidemic-Type Aftershock Sequence (ETAS) model by replacing parametric kernels with **neural kernel functions**, enabling flexible and data-driven modeling of seismicity.

This repository provides:

- A modular implementation of **spatio-temporal point process models**
- Support for both **empirical (ETAS-style)** and **neural kernels**
- A flexible framework for analyzing **earthquake triggering mechanisms**
- Tools for **likelihood-based evaluation and benchmarking**

---

![ST-NKF](img/model.png)

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

This project supports multiple earthquake catalogs commonly used in seismicity modeling. Dataset configurations are defined in `config.py`.

### Available datasets

#### ComCat
- Source: USGS earthquake catalog  
- Region: California  
- Magnitude threshold: Mw ≥ 2.5  

#### SCEDC
- Southern California Seismic Network  
- Magnitude thresholds:
  - `SCEDC_20` (Mw ≥ 2.0)  
  - `SCEDC_25` (Mw ≥ 2.5)  
  - `SCEDC_30` (Mw ≥ 3.0)  

#### QTM (Template Matching Catalog)
- High-resolution seismic catalog  
- Regions:
  - `SanJac`  
  - `SaltonSea`  
- Magnitude threshold: Mw ≥ 1.0  

#### WHITE
- High-resolution catalog  
- Magnitude threshold: Mw ≥ 0.6  

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