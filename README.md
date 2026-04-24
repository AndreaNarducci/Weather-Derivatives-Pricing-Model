# Weather-Derivatives-Pricing-Model
# Weather Derivatives Pricing — HDD Call Options on Amsterdam Temperature

A quantitative framework for pricing Heating Degree Day (HDD) call options using an Ornstein–Uhlenbeck mean-reverting process on daily temperature data. Built as a BSc thesis project in Industrial Engineering, supervised by Prof. Marco Papi (Campus Bio-Medico di Roma).

---

## Motivation

Temperature risk is a major source of revenue uncertainty for energy companies, utilities, and agricultural firms. Weather derivatives — and HDD call options in particular — allow market participants to hedge against colder-than-expected winters driving up heating demand.

This project asks: *how can we price HDD call options analytically under a risk-neutral measure, given a mean-reverting stochastic process for temperature with seasonal drift and heteroskedastic monthly volatility?*

---

## Repository Structure

```
weather-derivatives-pricing/
│
├── weather_derivatives.py      # Main script — parameter estimation, simulation, pricing
├── README.md
└── Amsterdam30years.csv        # Not included — daily mean temperatures for Amsterdam
```

> **Data note:** The temperature series (`Amsterdam30years.csv`) is not included in this repository. The script expects a CSV with a single column of daily average temperatures in °C, covering approximately 30 years (~10,950 observations).

---

## Model

### Temperature Process

The daily temperature follows an Ornstein–Uhlenbeck process with seasonal deterministic drift:

```
T_t = a · T̄(t-1) + (1-a) · T_{t-1} + σ_m · ε_t
```

where the long-run mean is a sinusoidal seasonal function:

```
T̄(t) = A + B·t + C·sin(2πt/365 + D)
```

**Parameters:**
- `a` — discrete mean-reversion coefficient (estimated via Newton-Raphson WLS)
- `A, B, C, D` — seasonal trend parameters (estimated via nonlinear least squares)
- `σ²_m` — monthly heteroskedastic variance (estimated per month, per year)

### Parameter Estimation

| Component | Method |
|---|---|
| Seasonal trend A, B, C, D | Nonlinear least squares (`scipy.optimize.curve_fit`) |
| Mean-reversion `a` (initial) | Unweighted OLS |
| Mean-reversion `a` (final) | Newton-Raphson WLS with monthly variance weights |
| Monthly variance σ²_m | Year-specific MLE on OU residuals |

The yearly variance structure (`residual_vars_yearly`) estimates σ²_m separately for each of the 30 winters — capturing interannual heterogeneity in temperature variability that a single pooled estimate would miss.

---

## Pricing

### HDD Definition

The cumulative Heating Degree Day index over a winter period [0, n] is:

```
H_n = Σₜ max(18 - T_t, 0)
```

with units °C·days. The strike K is expressed in the same units.

### Analytical Pricing Formula

Under the risk-neutral measure Q with market price of risk λ, H_n is approximately Gaussian with:

```
E^Q[H_n] = 18·n - Σ T̄(t) + Σ (λ·σ_t / a)·(1 - exp(-a·t))

Var^Q[H_n] = Σ (σ²_t / 2a)·(1 - exp(-2a·t))
```

The HDD call option price is then:

```
C(K) = e^{-rT} · [(μ - K)·Φ(-α) + σ·φ(α)]
```

where `α = (K - μ)/σ`, `Φ` is the standard normal CDF and `φ` the PDF.

> **Note:** local time indices (t = 0, 1, 2, ...) are used in the exponential terms to prevent numerical collapse on long global time indices.

---

## Outputs

Running `weather_derivatives.py` produces:

- **Deterministic temperature curve** — fitted seasonal trend over 30 years
- **Stochastic simulation** — OU process vs deterministic baseline
- **Yearly variance matrix** — 30×12 table of monthly σ² estimates
- **HDD call price surfaces** — one 3D surface per winter (Strike K × Maturity)
- **Surface animation** — `hdd_call_surface_animation.gif` showing how the price surface evolves across 30 winters

---
## Preview of the Outputs

1. Deterministic temperature function calibrated on 30 years of historical data (area-averaged time series)

<img width="1713" height="561" alt="image" src="https://github.com/user-attachments/assets/2676112d-7c7e-4cec-b47d-032da3afeefd" />

2. Corresponding Stochastic Temperature process

<img width="1713" height="561" alt="image" src="https://github.com/user-attachments/assets/c2653807-794a-4a9b-95fb-4b127e9b0494" />

3. Call-option price surface with respect to strike K (in degree-days) and time-to-maturity T for a generic winter

<img width="844" height="849" alt="image" src="https://github.com/user-attachments/assets/1cde34d3-ea0d-4475-af19-efba95ea8ea6" />



## Requirements

```
numpy
pandas
scipy
matplotlib
scikit-learn
pillow          # for animation export
```

Install with:

```bash
pip install numpy pandas scipy matplotlib scikit-learn pillow
```

---

## References

- Alaton, P., Djehiche, B. & Stillberger, D. (2002). On modelling and pricing weather derivatives. *Applied Mathematical Finance*, 9(1), 1–20.
- Brody, D. C., Syroka, J. & Zervos, M. (2002). Dynamical pricing of weather derivatives. *Quantitative Finance*, 2(3), 189–198.
- Schwartz, E. S. (1997). The stochastic behavior of commodity prices. *Journal of Finance*, 52(3), 923–973.

---

## Author

**Andrea Narducci**, BSc Industrial Engineering, MSc Chemical Engineering
