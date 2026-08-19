# Noise-Dependent Advantages of Expectile over Quantile Learning

A minimal Rescorla-Wagner model of striatal distributional reinforcement learning, built to ask one question: given that expectile and quantile updates are both valid distributional-RL targets at convergence, is there a normative, noise-dependent reason a biological circuit would be favored to implement one rather than the other?

## Overview

Lowet et al. (2025) found that striatal population activity encodes reward variance abstractly, and that this coding is best explained by *expectile* regression implemented via opponent, sign-flipped plasticity in D1- and D2-type medium spiny neurons (REDRL) — not by the *quantile* regression that underlies the original dopamine evidence for distributional RL (Dabney et al., 2020). Since expectile and quantile agents converge to the same correct answers under idealized, noise-free conditions, Bellman-closedness theory alone (Rowland et al., 2019) does not explain why a noisy, non-stationary biological circuit would prefer one over the other.

This project tests that question directly with a minimal population of Rescorla-Wagner units with heterogeneous, asymmetric learning rates, calibrated (not fit) to the real striatal recordings. Three findings, reported in full in [`docs/report/report.pdf`](docs/report/report.pdf):

1. **Baseline validation.** Under clean, noise-free conditions, the model reproduces the qualitative signatures of the real striatal code: an abstraction hierarchy across mean/variance/kurtosis contrasts, opponent geometry between optimistic ("D1-like") and pessimistic ("D2-like") subpopulations with no D1/D2 architecture imposed, and a behavioral readout that discriminates rewarded from unrewarded cues while remaining blind to reward variance.
2. **Noise-dependent divergence.** As learning-signal noise increases, expectile's advantage over quantile in preserving mean-value coding grows monotonically (from a non-significant gap at zero noise to a large, highly significant one at high noise). More categorically: expectile produces true D1/D2-like opponent geometry at every noise level tested, while quantile never does, at any noise level.
3. **Generalization to active decision-making.** The same pattern largely holds in three qualitatively different bandit tasks (tracking abrupt distributional shifts, tracking continuous drift, and identifying a high-value but high-variance option under resource pressure): expectile agents track, adapt, and identify value faster than quantile agents, with the advantage widening under noise in two of the three tasks. The third (risk-sensitive survival) task is the one place this pattern did not hold at the highest noise level tested, and is reported as such.

The report's Discussion situates this result relative to a normative account of D1/D2 opponency grounded in environmental reward statistics rather than learning-signal noise (Jaskir & Frank, 2023), connects the expectile/quantile distinction to the sign-based vs. magnitude-based optimizer literature in machine learning (Bernstein et al., 2018), and reconciles it with statistics literature suggesting the opposite ordering for a different notion of robustness (outlier robustness in static regression, vs. sequential learning-signal noise here).

### References

- Bellemare, M. G., Dabney, W., & Munos, R. (2017). A Distributional Perspective on Reinforcement Learning. *ICML*, PMLR 70:449-458.
- Bernstein, J., Wang, Y.-X., Azizzadenesheli, K., & Anandkumar, A. (2018). signSGD: Compressed Optimisation for Non-Convex Problems. *ICML*, PMLR 80:560-569.
- Costa, V. D., Dal Monte, O., Lucas, D. R., Murray, E. A., & Averbeck, B. B. (2016). Amygdala and Ventral Striatum Make Distinct Contributions to Reinforcement Learning. *Neuron*, 92(2), 505-517.
- Dabney, W., Kurth-Nelson, Z., Uchida, N., Starkweather, C. K., Hassabis, D., Munos, R., & Botvinick, M. (2020). A distributional code for value in dopamine-based reinforcement learning. *Nature*, 577, 671-675.
- Findling, C., Skvortsova, V., Dromnelle, R., Palminteri, S., & Wyart, V. (2019). Computational noise in reward-guided learning drives behavioral variability in volatile environments. *Nature Neuroscience*, 22, 2066-2077.
- Jaskir, A., & Frank, M. J. (2023). On the normative advantages of dopamine and striatal opponency for learning and choice. *eLife*, 12, e85107.
- Lefebvre, G., Lebreton, M., Meyniel, F., Bourgeois-Gironde, S., & Palminteri, S. (2017). Behavioural and neural characterization of optimistic reinforcement learning. *Nature Human Behaviour*, 1, 0067.
- Lowet, A. S., Zheng, Q., Meng, M., Matias, S., Drugowitsch, J., & Uchida, N. (2025). An opponent striatal circuit for distributional reinforcement learning. *Nature*, 639, 717-726.
- Newey, W. K., & Powell, J. L. (1987). Asymmetric Least Squares Estimation and Testing. *Econometrica*, 55(4), 819-847.
- Rowland, M., Dadashi, R., Kumar, S., Munos, R., Bellemare, M. G., & Dabney, W. (2019). Statistics and Samples in Distributional Reinforcement Learning. *ICML*, PMLR 97.
- Shen, Y., Tobia, M. J., Sommer, T., & Obermayer, K. (2014). Risk-sensitive Reinforcement Learning. *Neural Computation*, 26(7), 1298-1328.
- Tamar, A., Glassner, Y., & Mannor, S. (2015). Optimizing the CVaR via Sampling. *AAAI*, 29(1), 2993-2999.
- Waltrup, L. S., Sobotka, F., Kneib, T., & Kauermann, G. (2015). Expectile and quantile regression—David and Goliath? *Statistical Modelling*, 15(5), 433-456.
- Zalocusky, K. A., Ramakrishnan, C., Lerner, T. N., Davidson, T. J., Knutson, B., & Deisseroth, K. (2016). Nucleus accumbens D2R cells signal prior outcomes and control risky decision-making. *Nature*, 531, 642-646.

## Repository structure

```
.
├── 00_prepare_data.ipynb      # Colab only. Downloads real striatal data, computes and
│                               # saves the small calibration artifact. Self-contained.
├── 01_rw_baseline.ipynb       # Local. Baseline validation under clean conditions.
├── 02_noise_robustness.ipynb  # Local. Core result: AI gap and opponent geometry vs. noise.
├── 03_task_generalization.ipynb  # Local. Active decision-making across three bandit tasks.
├── rw_utils.py                 # Shared module: decoding/geometry helpers, RW population,
│                               # generic bandit runner. Imported by 01-03.
├── requirements.txt            # Light dependencies for 01-03 (numpy/pandas/scipy/
│                               # scikit-learn/matplotlib). 00 installs its own, heavier,
│                               # set of dependencies inline (needed to unpickle the
│                               # original paper's data).
├── artifacts/                  # Outputs. empirical_constants.npz (from 00) is the only
│                               # required input to 01-03; the rest are result tables
│                               # (CSV) written by 01-03 for reuse/inspection.
├── figures/                    # PDF figures written by 01-03 (source of truth).
└── docs/
    └── report/
        ├── report.tex          # The write-up: Introduction, Results, Methods, Discussion.
        ├── references.bib      # Bibliography (verified against original sources).
        └── figures/             # Copies of figures/*.pdf, referenced by report.tex.
```

## Run instructions

### 1. Generate the calibration artifact (Google Colab)

Open `00_prepare_data.ipynb` in Colab and run it top to bottom. This is the only notebook that downloads the real striatal recordings (Lowet et al., 2025, via Dryad) or installs the heavier dependency set needed to unpickle them. It ends by saving `artifacts/empirical_constants.npz` and offering it as a download via `files.download(...)`.

Save the downloaded `empirical_constants.npz` into this repo's local `artifacts/` folder before continuing.

### 2. Run the modeling notebooks (local)

Install the requirements at `requirements.txt`. Run `01_rw_baseline.ipynb`, `02_noise_robustness.ipynb`, and `03_task_generalization.ipynb`, in that order, from the repo root (they import `rw_utils.py` and read `artifacts/empirical_constants.npz` via relative paths). Each notebook writes its own figures to `figures/` and result tables to `artifacts/` as it runs.
