# ST-NKF: Spatio-Temporal Neural Kernel Function for Earthquake Forecasting

ST-NKF model is a spatiotemporal Neural Point Processes for earthquake forecasting. It generalizes the Epidemic-Type Aftershock Sequence (ETAS) model by replacing parametric kernels with neural kernel functions, enabling flexible and data-driven modeling of seismicity.


---

![ST-NKF](method.jpg)

---

The conditional intensity function is defined as:

```math
\lambda(t, x, y \mid \mathcal{H}_t) = \mu(x, y) + \sum_{(t_i, x_i, y_i, m_i) \in \mathcal{H}_t} \kappa(m_i) \cdot h(t - t_i) \cdot w(x - x_i, y - y_i)
```

Where:

- **$\mu(x, y)$**: background rate  
- **$\kappa(m)$**: productivity function  
- **$h(t)$**: temporal probability density function  
- **$w(x, y)$**: spatial probability density function  
- **$\mathcal{H}_t$**: historical event set up to time $t$  



These functions can be set as empirical functions or neural functions.


| Component        | Empirical       | Neural           |
|------------------|----------------|------------------|
| Temporal kernel  | Omori law      | Neural network   |
| Spatial kernel   | ETAS spatial   | Neural network   |
| Productivity     | Exponential    | Neural network   |

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
conda activate py311-tf
```

---

### 3. Run Experiments

All experiments can be executed from the command line using the provided Python scripts.

```bash
python run_models.py 1 1 "SanJac" "empirical" "empirical" "empirical"
```

The command-line arguments are:

| Argument | Description |
|----------|-------------|
| **1st** | **Random seed.** Controls both the random initialization of the NKF model and the random sampling of training events. For the ETAS model, the seed only affects the sampling of training events. |
| **2nd** | **GPU ID** used for training (e.g., `0`, `1`). |
| **3rd** | **Dataset name.** Available options are: `"SCEDC_20"`, `"SCEDC_25"`, `"SCEDC_30"`, `"ComCat"`, `"SaltonSea"`, `"SanJac"`, and `"WHITE"`. |
| **4th** | **Productivity function.** Either `"empirical"` or `"neural"`. |
| **5th** | **Temporal function.** Either `"empirical"` or `"neural"`. |
| **6th** | **Spatial function.** Either `"empirical"` or `"neural"`. |


To reproduce all benchmark results (three independent runs of both ETAS and NKF on all earthquake catalogs), simply execute:

```bash
python batch.py
```

This script automatically runs all datasets with three independent random seeds and reproduces the experimental results reported in the paper.



---


We also provide a Jupyter notebook for a quick start:

```bash
main.ipynb
```

---


## 🏗️ Initialize Model

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


We briefly describe the key input parameters of the `KernelPointProcess`.

- **`time_step_train`, `time_step_val`, `time_step_test`**  
  These parameters define the number of historical events used during training, validation, and testing. Both **ST-NKF** and the **limited-history ETAS** model use a fixed number of the most recent events as input. The input length is determined by the auxiliary window size of each earthquake catalog:

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
  These parameters specify whether the temporal kernel, spatial kernel, and productivity function are implemented using empirical ETAS formulations or neural networks.

  | Configuration | Model |
  |--------------|-------|
  | `"empirical"` | Empirical ETAS component |
  | `"neural"` | Neural component |

  Setting all three parameters to `"empirical"` reproduces the ETAS model, while setting all three to `"neural"` reproduces **ST-NKF**. Mixed configurations are also supported, allowing empirical and neural components to be freely combined for ablation studies.

### Examples

#### ETAS

```python
temporal_id = "empirical"
spatial_id  = "empirical"
kappa_id    = "empirical"
```

#### ST-NKF

```python
temporal_id = "neural"
spatial_id  = "neural"
kappa_id    = "neural"
```

#### Hybrid

```python
temporal_id = "neural"
spatial_id  = "empirical"
kappa_id    = "neural"
```


- **`global_m0`**  
  The magnitude threshold (cutoff magnitude) of the earthquake catalog.

- **`area`**  
  The spatial area of the study region.

- **`size_layer`**  
  The number of layers in the neural network.

- **`size_nn`**  
  The number of neurons in each hidden layer.

---




## Datasets

This study is conducted based on the benchmark datasets provided by the [EarthquakeNPP](https://github.com/ss15859/EarthquakeNPP), which is designed for evaluating Neural Point Process (NPP) models on earthquake forecasting tasks.

We adopt multiple datasets from EarthquakeNPP to evaluate the proposed ST-NKF model:

#### ComCat
- Source: USGS Advanced National Seismic System (ANSS) catalog  
- Region: California  
- Dataset: `ComCat_25`  
- Magnitude threshold: M ≥ 2.5  

#### SCEDC
- Source: Southern California Earthquake Data Center (SCEDC)  
- Datasets:
  - `SCEDC_20` (M ≥ 2.0)  
  - `SCEDC_25` (M ≥ 2.5)  
  - `SCEDC_30` (M ≥ 3.0)  

#### QTM (Template Matching Catalog)
- High-resolution catalog constructed via waveform template matching  
- Regions:
  - `SanJac_10` (San Jacinto fault zone)  
  - `SaltonSea_10`  
- Magnitude threshold: M ≥ 1.0  

#### WHITE
- High-resolution catalog focusing on the San Jacinto fault region  
- Dataset: `WHITE_06`  
- Magnitude threshold: M ≥ 0.6  


---

### 📂 Data Usage in This Work

The earthquake catalogs provided by **EarthquakeNPP** have undergone standardized preprocessing procedures, including:

- Selection of earthquakes within specified spatial regions and time periods, followed by temporal splitting into **auxiliary**, **training**, **validation**, and **test** sets  
- Estimation of the **magnitude of completeness** for each catalog  
- Filtering of events to retain only earthquakes above the completeness threshold  

---

### 📊 Dataset Splits

The temporal splits of each earthquake catalog are summarized below:

| Catalog      | Auxiliary Start | Training Start | Validation Start | Testing Start | Testing End |
|-------------|----------------|----------------|------------------|---------------|-------------|
| ComCat      | 1971-01-01     | 1981-01-01     | 1998-01-01       | 2007-01-01    | 2020-01-17  |
| SCEDC_20   | 1981-01-01     | 1985-01-01     | 2005-01-01       | 2014-01-01    | 2020-01-01  |
| SCEDC_25   | 1981-01-01     | 1985-01-01     | 2005-01-01       | 2014-01-01    | 2020-01-01  |
| SCEDC_30   | 1981-01-01     | 1985-01-01     | 2005-01-01       | 2014-01-01    | 2020-01-01  |
| SanJac      | 2008-01-01     | 2009-01-01     | 2014-01-01       | 2016-01-01    | 2018-01-01  |
| SaltonSea   | 2008-01-01     | 2009-01-01     | 2014-01-01       | 2016-01-01    | 2018-01-01  |
| WHITE       | 2008-01-01     | 2009-01-01     | 2014-01-01       | 2017-01-01    | 2021-01-01  |

For detailed preprocessing procedures, please refer to the EarthquakeNPP GitHub repository: https://github.com/ss15859/EarthquakeNPP

---

### Additional Processing: Spatial Perturbation

In this work, we introduce a small spatial perturbation to events that share identical locations in the catalog. Such duplicated locations can lead to numerical instability in spatial kernel estimation for neural models. To address this issue, we add a small uniform random perturbation to the x and y of duplicated events.

- Noise range: ±0.5 km
- Applied only to events with identical spatial coordinates
- Repeated until all duplicated locations are removed
- A fixed random seed is used to ensure reproducibility
- The complete preprocessing implementation is provided in `Data processing.ipynb`


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


### Reproducibility Across Random Seeds

To evaluate the robustness of **ST-NKF**, we independently trained the model three times using different random seeds (`1`, `2`, and `3`). The table below reports the total, temporal, and spatial log-likelihoods on the test sets for each earthquake catalog.

| Dataset | Seed | Total | Temporal | Spatial |
|---------|:----:|------:|---------:|--------:|
| SCEDC_20 | 1 | -5.0757 | 2.5327 | -7.6084 |
|  | 2 | -5.0823 | 2.5435 | -7.6259 |
|  | 3 | -5.0625 | 2.5518 | -7.6142 |
| SCEDC_25 | 1 | -5.5191 | 2.1053 | -7.6245 |
|  | 2 | -5.5391 | 2.1037 | -7.6428 |
|  | 3 | -5.5559 | 2.0873 | -7.6432 |
| SCEDC_30 | 1 | -5.9918 | 1.7407 | -7.7324 |
|  | 2 | -5.9936 | 1.7514 | -7.7450 |
|  | 3 | -5.9278 | 1.8031 | -7.7309 |
| ComCat | 1 | -7.0619 | 1.4242 | -8.4861 |
|  | 2 | -7.0590 | 1.4233 | -8.4823 |
|  | 3 | -7.0589 | 1.4143 | -8.4732 |
| SaltonSea | 1 | -0.1638 | 2.2905 | -2.4543 |
|  | 2 | -0.1642 | 2.3184 | -2.4826 |
|  | 3 | -0.1946 | 2.3003 | -2.4949 |
| SanJac | 1 | -4.4820 | 1.1185 | -5.6005 |
|  | 2 | -4.4922 | 1.1077 | -5.5998 |
|  | 3 | -4.4831 | 1.1122 | -5.5954 |
| WHITE | 1 | -2.3940 | 2.0252 | -4.4192 |
|  | 2 | -2.3886 | 2.0256 | -4.4142 |
|  | 3 | -2.3977 | 2.0289 | -4.4265 |


### Figure Reproduction

We also provide the scripts used to generate the figures presented in the paper.

```text
plot CDF.ipynb
plot comparison.ipynb
plot pattern.ipynb
plot regression.ipynb
plot spatial information gain.ipynb
plot temporal information gain.ipynb
```

These notebooks reproduce the main figures used in the manuscript, including CDF visualization, model comparison, seismicity patterns, regression analysis, and temporal/spatial information gain evaluation.