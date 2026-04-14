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
git clone https://github.com/your_username/ST-NKF.git
```

2. Navigate to the project directory:

```bash
cd ST-NKF
```

3. Install dependencies:

```bash
pip install tensorflow numpy pandas
```
