# Gaussian Belief Propagation using Markov Random Field

An implementation of _Gaussian Belief Propagation_ (GABP) for 
probabilistic inference on _Markov Random Field_ (MRF) - an 
undirected graphical model.

> This code based on the thoery from [this thesis](https://arxiv.org/abs/0811.2518) by Danny Bickson. Check it out for 
more information.

## Directory Structure

```
├─ notebooks
│  ├─ GaBP MRF.ipynb
│  └─ GaBP MRF Experiments.ipynb
├─ src
│  ├─ gabp_mrf.py
│  ├─ synthetic_data.py
│  ├─ utils.py
│  └─ visualization.py
├─ README.md
└─ requirements.txt
```

## Usage Example

This project features two different notebooks, which provides the usage example:

1. `GaBP MRF.ipynb`:
   * Based on the code from `src.gabp_mrf.py` which provides optimized implementation using [Numba](http://numba.pydata.org/).
   * Provide usage example for most of the implemented features.
   * Provide some useful analysis on the performance of the algorithm.
   * Check it out: 
   <div align="center">
    <a href="https://colab.research.google.com/github/Lin-Sinorodin/Gaussian_Belief_Propagation/blob/main/notebooks/GaBP_MRF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    </div>
2. `GaBP MRF Experiments.ipynb`:
    * This is the script I used for learning the algorithm before writing the optimized implementation.
    * Provide details about the flow of the algorithm in each step, and should be easier to understand than the optimized implementation.
    * Provide summery of the theory, and comparison to the exact solution in comparison to the GaBP solution.
    * Check it out: 
   <div align="center">
    <a href="https://colab.research.google.com/github/Lin-Sinorodin/Gaussian_Belief_Propagation/blob/main/notebooks/GaBP_MRF_Experiments.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    </div>