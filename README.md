# ST-NKF: Spatio-Temporal Neural Kernel Function for Earthquake Forecasting

ST-NKF model is a spatiotemporal Neural Point Processes for earthquake forecasting. It generalizes the Epidemic-Type Aftershock Sequence (ETAS) model by replacing parametric kernels with neural kernel functions, enabling flexible and data-driven modeling of seismicity.


---

![ST-NKF](method.jpg)

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/zhancx-edu/NeuralKernelFunction.git
cd NeuralKernelFunction
```

### 2. Install Dependencies

We provide a Conda environment file for reproducibility:

```bash
conda env create -f environment.yml
conda activate neuralkernelfunction
```

---

## 🚀 Running Experiments

All experiments are implemented in:

```bash
main.ipynb
```

---

### 🧩 Step 1: Select Dataset

Set the dataset name in the notebook:

```python
dataset_name = "ComCat"
```

Available datasets:

```python
["ComCat", "SaltonSea", "SanJac", "WHITE", "SCEDC_20", "SCEDC_25", "SCEDC_30"]
```

---

### 🧠 Step 2: Configure Model Components

The model consists of three components:

- **Temporal kernel** (`temporal_id`)
- **Spatial kernel** (`spatial_id`)
- **Productivity function** (`kappa_id`)

Each component supports:

- `"empirical"` — empirical kernel in ETAS model  
- `"neural"` — neural kernel  

---

### 🔧 Example Configurations

#### ETAS (Fully Empirical)

```python
temporal_id = "empirical"
spatial_id = "empirical"
kappa_id   = "empirical"
```

#### ST-NKF (Fully Neural)

```python
temporal_id = "neural"
spatial_id = "neural"
kappa_id   = "neural"
```

#### Hybrid Model (Example)

```python
temporal_id = "neural"
spatial_id = "empirical"
kappa_id   = "neural"
```

This flexible design enables controlled comparisons between classical ETAS and neural kernel models.

---

### ▶️ Step 3: Run the Notebook

Run all cells in:

```bash
main.ipynb
```

The pipeline will automatically:

- Load and preprocess datasets  
- Build the point process model  
- Train via log-likelihood optimization  
- Evaluate performance on validation and test sets  
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

---

## 🧩 Model Configuration

Before initializing the model, we briefly describe the key input parameters of the `KernelPointProcess`.

### 🔑 Input Parameters

- **`time_step_train`, `time_step_val`, `time_step_test`**  
  These parameters define the number of historical events used as input during training, validation, and testing, respectively.  

  In our implementation, both the **ST-NKF** model and the **limited-history ETAS** model take as input a fixed number of the most recent events.  
  The input length is determined by the auxiliary window size of each earthquake catalog:

  ```
  ComCat: 14933  
  SaltonSea: 2144  
  SanJac: 1672  
  WHITE: 2196  
  SCEDC_2.0: 12373  
  SCEDC_2.5: 4242  
  SCEDC_3.0: 1142  
  ```

- **`temporal_id`, `spatial_id`, `kappa_id`**  
  These parameters specify the formulation of the three kernel components:
  - Temporal kernel  
  - Spatial kernel  
  - Productivity function  

  Each can be set to:
  - `"empirical"` — classical ETAS formulation  
  - `"neural"` — neural network-based formulation  

- **`global_m0`**  
  The magnitude threshold (cutoff magnitude) of the earthquake catalog.

- **`area`**  
  The spatial area of the study region.

- **`size_layer`**  
  The number of layers in the neural network.

- **`size_nn`**  
  The number of neurons in each hidden layer.

---

### 🏗️ Initialize Model

```python
model = KernelPointProcess(
    time_step_train=input_dim_train,
    time_step_val=input_dim_val,
    time_step_test=input_dim_test,
    temporal_id=temporal_id,
    spatial_id=spatial_id,
    kappa_id=kappa_id,
    global_m0=config["global_m0"],
    area=obj_area,
    size_layer=5,
    size_nn=32
).set_train_data(
    data_t_train, data_m_train, data_x_train, data_y_train
).set_val_data(
    data_t_val, data_m_val, data_x_val, data_y_val
).set_test_data(
    data_t_test, data_m_test, data_x_test, data_y_test
).set_model().compile().fit_eval(
    epochs=1000, batch_size=128
).eval_train().eval_val().eval_test().save_weights(
    f"weights/{datasets_id}_{temporal_id}_{spatial_id}_{kappa_id}"
)
```

---

## ⏱️ Computational Environment

All experiments were conducted on a workstation equipped with:

- NVIDIA GeForce RTX 3090 GPU  
- AMD EPYC 7302P 16-Core processor (32 threads)  

---

## 📊 Runtime

The table below summarizes the computational time (in minutes) for both ST-NKF and ETAS models.

|    Model     | ComCat | SaltonSea | SanJac | WHITE | SCEDC_20 | SCEDC_25 | SCEDC_30 |
|--------------|--------:|----------:|-------:|------:|---------:|---------:|---------:|
|    ST-NKF    | 86.40   | 14.82     | 3.30   | 10.76 | 141.69   | 24.38    | 2.05     |
|    ETAS      | 12.14   | 2.14      | 0.86   | 2.71  | 18.53    | 4.70     | 1.33     |