# Neural Kernel Point Process (ST-NKF)

## 📌 Overview

This repository implements a **Spatio-Temporal Neural Kernel Point Process (ST-NKF)** for earthquake modeling and forecasting.

It generalizes the classical ETAS (Epidemic-Type Aftershock Sequence) model by replacing parametric kernels with **neural network-based kernels**, enabling more flexible and data-driven modeling of seismicity.

---

## ✨ Key Features

- 🔹 Unified framework for **ETAS and Neural Point Processes**
- 🔹 Modular kernel design:
  - Temporal kernel (Omori / Neural)
  - Spatial kernel (ETAS / Neural)
  - Productivity function (Exponential / Neural)
- 🔹 CDF-based neural kernels (stable & interpretable)
- 🔹 EM-like update of background rate (μ)
- 🔹 Likelihood-based training

---

## 🧠 Model Formulation

The conditional intensity function is:
