# Distributed GM-PHD Filters with Self-Referencing Information Diffusion

## Overview

This repository provides a Python framework for distributed multitarget tracking using a network of heterogeneous sensors with limited, partially overlapping, or non-overlapping fields of view (FoVs). 

The primary contribution is the implementation of a novel **Adapt-then-Combine (ATC) Information Diffusion** strategy enhanced with a **Self-Referencing (SR)** mechanism to filter out conflicting estimates. Instead of iterative consensus, this approach relies on a highly efficient one-shot communication topology at each discrete time step:
1. **Adaptation Phase:** Sensors exchange local measurements with adjacent neighbors.
2. **Combination Phase:** Sensors share and fuse Probability Hypothesis Density (PHD) estimates.
3. **Self-Referencing:** Each sensor selectively integrates only the neighboring information that properly aligns with its own local posterior, which naturally suppresses unreliable tracks and handles sensor-specific FoVs, clutter rates, and detection probabilities robustly.

The codebase includes implementations and comparisons of the following key algorithms:
- **SR-GM-PHD** (Diffusion Self-Referencing GM-PHD filter) natively supporting multiple combination strategies for fusion (GCI, MM, UWAA, CWAA).
- **SD-WAA GM-PHD** (State-Dependent Weighted Arithmetic Average GM-PHD) filter, implemented as a comparative baseline.

### Accompanying Paper

Our approaches and the specific simulation scenarios implemented in this repository are thoroughly detailed in the following publication:

> P. Fiedler and K. Dedecius, "Self-Referencing Adapt-Then-Combine Information Diffusion Scheme for Distributed PHD Filtering," *IEEE Signal Processing Letters*, vol. 33, p. 251, 2026.
> **DOI:** [10.1109/LSP.2025.3642058](https://ieeexplore.ieee.org/document/11287995)

## Project Structure

- `example.ipynb`: An interactive Jupyter Notebook containing the full simulation scenario, comparing the ATC-PHD schemes with SD-WAA in a heterogeneous sensor network.
- `DiffPHD/`: Contains the implementation of the Diffusion GM-PHD filter, multi-sensor integration logic, and combinations strategies.
- `SDWAAPHD/`: Contains the implementation of the SD-WAA GM-PHD filter network operations.
- `cached-data.pkl`: Pre-run simulation state and metrics to quickly reproduce plots.

## Installation

The simulation depends on standard scientific Python libraries. You can run the code by cloning the repository and ensuring you have the following installed:

```bash
pip install numpy matplotlib scipy jupyter
```

## Usage

1. Clone the repository locally.
2. Open your terminal and start jupyter:
   ```bash
   jupyter notebook example.ipynb
   ```
3. Run the cells to reproduce the specific findings from the paper.
