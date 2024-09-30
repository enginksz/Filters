# Kalman Filters

Kalman filters are powerful mathematical methods used to estimate and track states from noisy measurements. They are commonly applied in control systems, robotics, financial modeling, and many other fields.

## Contents

- [Introduction](#introduction)
- [Fundamentals of Kalman Filters](#fundamentals-of-kalman-filters)
- [Application Areas](#application-areas)
- [Sample Implementation](#sample-implementation)

## Introduction

Kalman filters use an iterative approach to predict system states from time-series data. The filter combines past measurements and current data to provide the best estimates.

## Fundamentals of Kalman Filters

A Kalman filter consists of two main steps:

1. **Prediction Step:** Updates the system state and error covariance.
2. **Update Step:** Adjusts the prediction based on measurement data.

### Formulation

**Prediction:**  
- State equation:  
  \[
  \hat{x}_{k|k-1} = A \hat{x}_{k-1|k-1} + B u_k
  \]

- Error covariance:  
  \[
  P_{k|k-1} = A P_{k-1|k-1} A^T + Q
  \]

**Update:**  
- Gain:  
  \[
  K_k = \frac{P_{k|k-1} H^T}{H P_{k|k-1} H^T + R}
  \]

- State update:  
  \[
  \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - H \hat{x}_{k|k-1})
  \]

- Error covariance update:  
  \[
  P_{k|k} = (I - K_k H) P_{k|k-1}
  \]

## Application Areas

- **Robotics:** Robot position estimation and tracking
- **Finance:** Market forecasting and risk management
- **Autonomous Vehicles:** Path and obstacle detection
- **Weather Forecasting:** Atmospheric data analysis

## Sample Implementation

```bash
python3 gps.py
```

and 

```
python3 kalman.py
```


