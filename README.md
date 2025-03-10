# Gamma ray classifier

## Overview
This repository contains a machine learning project aimed at classifying high-energy gamma-ray events. The goal is to develop a model that can distinguish between gamma-ray signals and hadronic background events, which are caused by cosmic rays, using shower images generated by the CORSIKA Monte Carlo simulation.

## Dataset
The dataset consists of pre-processed shower images with associated statistical parameters. These parameters include Hillas characteristics and energy deposition asymmetry, which serve as features to distinguish between gamma-ray events (signal) and cosmic ray background (noise).

**Source:** [MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)

## Project Goals
- Perform **exploratory data analysis (EDA)** to understand the dataset.
- Train and evaluate different **classification models**.
- Optimize model performance.
- Document findings.

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/Kseos/gamma-ray-classifier.git
cd gamma-ray-classifier
pip install -r requirements.txt
```

## Folder Structure
```
gamma-ray-classifier/
├── data/             # Dataset files
├── notebooks/        # Jupyter notebooks
├── src/              # Code for data processing and modeling
├── results/          # Model outputs
├── README.md         # Documentation
├── requirements.txt  # Dependencies
```

## Features

| Feature   | Type      | Description                                                       |
|-----------|-----------|-------------------------------------------------------------------|
| **fLength** | continuous | Major axis of ellipse [mm]                                       |
| **fWidth**  | continuous | Minor axis of ellipse [mm]                                       |
| **fSize**   | continuous | 10-log of sum of content of all pixels [in #phot]                |
| **fConc**   | continuous | Ratio of sum of two highest pixels over fSize [ratio]            |
| **fConc1**  | continuous | Ratio of highest pixel over fSize [ratio]                        |
| **fAsym**   | continuous | Distance from highest pixel to center, projected onto major axis [mm] |
| **fM3Long** | continuous | 3rd root of third moment along major axis [mm]                   |
| **fM3Trans**| continuous | 3rd root of third moment along minor axis [mm]                   |
| **fAlpha**  | continuous | Angle of major axis with vector to origin [deg]                  |
| **fDist**   | continuous | Distance from origin to center of ellipse [mm]                   |
| **class**   | categorical | g (gamma, signal), h (hadron, background)                        |

### Class Distribution
- **g (gamma, signal)**: 12,332
- **h (hadron, background)**: 6,688

For technical reasons, the number of **h** events is underestimated. In real data, the **h** class represents the majority of events.

### Classification Considerations
Simple accuracy is not meaningful for this dataset. Misclassifying a background event as signal is worse than misclassifying a signal event as background. Model evaluation should be done using an **ROC curve**, with relevant thresholds for background acceptance probabilities: **0.01, 0.02, 0.05, 0.1, 0.2**, depending on experiment quality requirements.

---
📌 **Keywords:** Machine Learning, Gamma-Ray Classification, Cherenkov Telescope, Monte Carlo Simulation.