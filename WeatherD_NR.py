# -*- coding: utf-8 -*-
"""
Weather Derivatives Pricing — HDD Call Options
OU process on Amsterdam temperature with seasonal drift
"""

from scipy.stats import norm
from scipy.stats import shapiro, jarque_bera
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── DATA LOADING ──────────────────────────────────────────────────────────────
df = pd.read_csv('Amsterdam30years.csv', header=None, names=['tavg'])        #specify the path of the CSV with daily temperatures here; advice: use the Giovanni Merra's NASA Database for area-averaged time series

print(df['tavg'])
mean      = df['tavg'].mean()
median    = df['tavg'].median()
std_dev   = df['tavg'].std()
minimum   = df['tavg'].min()
maximum   = df['tavg'].max()
skewness  = df['tavg'].skew()
kurtosis  = df['tavg'].kurtosis()

print("📊 Temperature Statistical Summary:")
print(f"Mean:      {mean:.2f} °C")
print(f"Median:    {median:.2f} °C")
print(f"Std Dev:   {std_dev:.2f}")
print(f"Min:       {minimum:.2f} °C")
print(f"Max:       {maximum:.2f} °C")
print(f"Skewness:  {skewness:.2f}")
print(f"Kurtosis:  {kurtosis:.2f}")

temps = df['tavg'].dropna()
shap_stat, shap_p = shapiro(temps)
print("🔍 Shapiro-Wilk Test:")
print(f"Statistic: {shap_stat:.4f}, p-value: {shap_p:.4f}")
print("❌ Reject normality" if shap_p < 0.05 else "✅ Cannot reject normality")

jb_stat, jb_p = jarque_bera(temps)
print("\n🔍 Jarque-Bera Test:")
print(f"Statistic: {jb_stat:.4f}, p-value: {jb_p:.4f}")
print("❌ Reject normality" if jb_p < 0.05 else "✅ Cannot reject normality")


# ── PARAMETER ESTIMATION ──────────────────────────────────────────────────────
df['t'] = np.arange(len(df))

def seasonal_linear_model(t, a1, a2, a3, a4):
    omega = 2 * np.pi / 365
    return a1 + a2 * t + a3 * np.sin(omega * t) + a4 * np.cos(omega * t)

t_data = df['t'].values
y_data = df['tavg'].values
params, _ = curve_fit(seasonal_linear_model, t_data, y_data)
a1, a2, a3, a4 = params

def temperature_function(t, a1, a2, a3, a4):
    A = a1
    B = a2
    C = np.sqrt(a3**2 + a4**2)
    D = np.arctan2(a4, a3)
    w = 2 * np.pi / 365
    return A + B * t + C * np.sin(w * t + D)

A = a1
B = a2
C = np.sqrt(a3**2 + a4**2)
D = np.arctan2(a4, a3)
w = 2 * np.pi / 365

print(A, B, C, D)

# ── DETERMINISTIC SIMULATION ──────────────────────────────────────────────────
n_days = 10959
t      = np.arange(n_days)

def temperature_function_sim(t, A, B, C, D):
    return A + B * t + C * np.sin(w * t + D)

tempr = temperature_function_sim(t, A, B, C, D)

plt.figure(figsize=(12, 4))
plt.plot(t, tempr, label='Deterministic Temperature', color='crimson')
plt.title('Simulated Deterministic Temperature Curve (30 years)')
plt.xlabel('Day'); plt.ylabel('Temperature (°C)')
plt.grid(True); plt.tight_layout(); plt.legend(); plt.show()

# ── LAGGED TERMS AND DATE ─────────────────────────────────────────────────────
df['T_det']      = temperature_function(df['t'], a1, a2, a3, a4)
df['T_prev']     = df['tavg'].shift(1)
df['T_det_prev'] = df['T_det'].shift(1)
df['date']       = pd.date_range(start='1985-01-01', periods=len(df), freq='D')
df['month']      = df['date'].dt.month

# ── INITIAL ESTIMATE OF a (unweighted) ───────────────────────────────────────
df_clean  = df.dropna().copy()
Y_prev    = df_clean['T_det_prev'] - df_clean['T_prev']
numerator = (Y_prev * (df_clean['tavg'] - df_clean['T_det'])).sum()
denominator = (Y_prev * (df_clean['T_prev'] - df_clean['T_det_prev'])).sum()
a_hat_initial = -np.log(numerator / denominator)
print(f"🔧 Initial â estimate (no variance weighting): {a_hat_initial:.4f}")

# ── GLOBAL MONTHLY VARIANCES (fallback) ───────────────────────────────────────
residual_vars = {}
for month in range(1, 13):
    df_m = df[df['month'] == month].copy()
    df_m['T_prev']     = df_m['tavg'].shift(1)
    df_m['T_det_prev'] = df_m['T_det'].shift(1)
    df_m = df_m.dropna()
    eps  = df_m['tavg'] - a_hat_initial * df_m['T_det_prev'] - (1 - a_hat_initial) * df_m['T_prev']
    residual_vars[month] = (eps**2).sum() / (len(df_m) - 2)

df['sigma2']      = df['month'].map(residual_vars)
df['sigma2_prev'] = df['sigma2'].shift(1)
df_final          = df.dropna().copy()

# ── NEWTON-RAPHSON ESTIMATION ─────────────────────────────────────────────────
def estimate_a_newton_raphson(df_data, residual_vars, a_init=0.01, max_iter=50, tol=1e-10):
    df_clean   = df_data.dropna(subset=['tavg', 'T_prev', 'T_det_prev', 'month']).copy()
    sigma2     = df_clean['month'].map(residual_vars).values
    T_i        = df_clean['tavg'].values
    T_prev     = df_clean['T_prev'].values
    T_det_prev = df_clean['T_det_prev'].values
    b_hat      = T_det_prev - T_prev

    print("\n" + "="*70)
    print("NEWTON-RAPHSON ESTIMATION")
    print("="*70)
    print(f"{'Iter':<6} {'a':<18} {'G(a)':<18} {'|G(a)|':<12}")
    print("-"*70)

    a = a_init
    for iteration in range(max_iter):
        E_T       = a * T_det_prev + (1 - a) * T_prev
        residuals = T_i - E_T
        G         = np.sum((b_hat / sigma2) * residuals)
        dG        = -np.sum(b_hat**2 / sigma2)
        print(f"{iteration:<6} {a:<18.10f} {G:<18.6e} {abs(G):<12.6e}")
        if abs(G) < tol:
            print("-"*70)
            print(f"✅ CONVERGED after {iteration} iterations")
            break
        a = np.clip(a - G / dG, 1e-8, 1 - 1e-8)
    else:
        print("⚠️ Reached max iterations without full convergence")

    print(f"\nFinal estimate: a = {a:.10f}")
    print(f"Final |G(a)| = {abs(G):.6e}")
    print("="*70)
    return a

Y_prev      = df_clean['T_det_prev'] - df_clean['T_prev']
numerator   = (Y_prev * (df_clean['tavg'] - df_clean['T_det'])).sum()
denominator = (Y_prev * (df_clean['T_prev'] - df_clean['T_det_prev'])).sum()
a_initial   = -np.log(numerator / denominator)
print(f"\n📍 Initial guess (unweighted): a = {a_initial:.10f}")

an        = estimate_a_newton_raphson(df, residual_vars, a_init=a_initial)
kappa     = -np.log(1 - an)
half_life = np.log(2) / kappa

print(f"\n📊 Parameter Interpretation:")
print(f"Discrete parameter:   a = {an:.10f}")
print(f"Mean reversion speed: κ = {kappa:.6f} per day")
print(f"Half-life of shocks:  {half_life:.1f} days")

# ── STOCHASTIC SIMULATION ─────────────────────────────────────────────────────
n_sim_days = len(t)
sim_t      = np.arange(n_sim_days)
sim_dates  = pd.date_range(start='1985-01-01', periods=n_sim_days, freq='D')
sim_months = sim_dates.month
T_det_sim  = temperature_function_sim(sim_t, A, B, C, D)
T_stoch    = np.zeros(n_sim_days)
T_stoch[0] = T_det_sim[0]
np.random.seed(42)
epsilons = np.random.normal(0, 1, size=n_sim_days)

for j in range(1, n_sim_days):
    month      = sim_months[j]
    sigma      = np.sqrt(residual_vars[month])
    T_stoch[j] = an * T_det_sim[j-1] + (1 - an) * T_stoch[j-1] + sigma * epsilons[j-1]

plt.figure(figsize=(12, 4))
plt.plot(sim_t, T_det_sim, label='Deterministic', color='crimson')
plt.plot(sim_t, T_stoch,   label='Stochastic (mean-reverting + noise)',
         color='steelblue', alpha=0.7)
plt.title("Deterministic vs Stochastic Temperature Simulation")
plt.xlabel("Day"); plt.ylabel("Temperature (°C)")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ── GLOBAL MONTHLY VARIANCE (refined with final a) ───────────────────────────
df['T_det']      = temperature_function(df['t'], a1, a2, a3, a4)
df['T_prev']     = df['tavg'].shift(1)
df['T_det_prev'] = df['T_det'].shift(1)

residual_vars = {}
for month in range(1, 13):
    df_month = df[df['month'] == month].copy()
    df_month = df_month.dropna(subset=['tavg', 'T_prev', 'T_det_prev'])
    residuals = df_month['tavg'] - (an * df_month['T_det_prev'] + (1 - an) * df_month['T_prev'])
    N = len(residuals)
    residual_vars[month] = (residuals**2).sum() / (N - 2) if N > 2 else 0

print("Estimated monthly variances (σ²):")
for m in range(1, 13):
    print(f"Month {m:2d}: σ² = {residual_vars[m]:.4f}")

# WINTER SLICES 
df["t_global"] = np.arange(len(df))
winter_slices  = []

for i in range(30):
    start_year = 1985 + i
    end_year   = start_year + 1
    start_date = pd.to_datetime(f"{start_year}-10-01")
    end_date   = pd.to_datetime(f"{end_year}-03-31")
    mask       = (df["date"] >= start_date) & (df["date"] <= end_date)
    winter_df  = df.loc[mask].copy()
    winter_df["t"] = df.loc[mask, "t_global"].values
    winter_slices.append(winter_df)

# YEARLY VARIANCES 
residual_vars_yearly = {}

for i, winter_df in enumerate(winter_slices):
    year = 1985 + i
    residual_vars_yearly[year] = {}
    for month in range(1, 13):
        df_m = winter_df[winter_df['date'].dt.month == month].copy()
        df_m['T_prev']     = df_m['tavg'].shift(1)
        df_m['T_det_prev'] = df_m['T_det'].shift(1)
        df_m = df_m.dropna()
        if len(df_m) < 3:
            residual_vars_yearly[year][month] = residual_vars[month]
            continue
        residuals = df_m['tavg'] - (an * df_m['T_det_prev'] + (1 - an) * df_m['T_prev'])
        N = len(residuals)
        residual_vars_yearly[year][month] = (residuals**2).sum() / (N - 2)

# print variance matrix
df_vars = pd.DataFrame(residual_vars_yearly).T
df_vars.index.name = 'Year'
df_vars.columns    = [f'Month_{m}' for m in range(1, 13)]
print("\nYearly variance matrix (first 3 months):")
print(df_vars.iloc[:, :3].to_string())

#PRICING 
def Tm(t, A, B, C, D, w=2*np.pi/365):
    return A + B * t + C * np.sin(w * t + D)

def HDD_call_price(mu, sigma, K, r, T):
    alpha = (K - mu) / sigma
    return np.exp(-r * T) * (
        (mu - K) * norm.cdf(-alpha) +
        sigma / np.sqrt(2 * np.pi) * np.exp(-alpha**2 / 2)
    )

lambda_ = 0.3
r       = 0.02

#EXPECTED VALUE AND VARIANCE (yearly variances + local time index)
expected_Hn_list = []
var_Hn_list      = []

for i, winter_df in enumerate(winter_slices):
    year    = 1985 + i
    rv_year = residual_vars_yearly[year]

    t_vals  = winter_df["t"].values
    t_local = np.arange(len(t_vals))   # ← indici locali 0,1,2,...
    months  = winter_df["date"].dt.month
    Tm_vals = Tm(t_vals, A, B, C, D)

    sigma2_vals = months.map(rv_year).values
    sigma_vals  = np.sqrt(sigma2_vals)

    # Expected value under Q
    correction = (lambda_ * sigma_vals / an) * (1 - np.exp(-an * t_local))
    Hn_Q       = 18 * len(t_vals) - np.sum(Tm_vals) + np.sum(correction)
    expected_Hn_list.append(Hn_Q)

    # Variance (local index prevents exp collapse)
    var_Hn = np.sum((sigma2_vals / (2 * an)) * (1 - np.exp(-2 * an * t_local)))
    var_Hn_list.append(var_Hn)

    print(f"Winter {i+1:2d} ({year}-{year+1}):  "
          f"EV[H_n] = {Hn_Q:8.2f}   "
          f"Var[H_n] = {var_Hn:8.2f}   "
          f"σ[H_n] = {np.sqrt(var_Hn):6.2f}")

# OPTION PRICE SURFACES (yearly variances + local time index) 
K_vals_log   = np.linspace(np.log(50), np.log(2000), 100)
K_vals       = np.exp(K_vals_log)
T_maturities = np.arange(30, 130, 10)

for i, winter_df in enumerate(winter_slices):
    year    = 1985 + i
    rv_year = residual_vars_yearly[year]

    price_surface = np.zeros((len(T_maturities), len(K_vals)))

    for t_idx, T_days in enumerate(T_maturities):
        df_slice = winter_df.iloc[:T_days].copy()
        if len(df_slice) < 10:
            continue

        t_vals  = df_slice["t"].values
        t_local = np.arange(len(t_vals))   # ← indici locali
        months  = df_slice["date"].dt.month
        Tm_vals = Tm(t_vals, A, B, C, D)

        sigma2_vals = months.map(rv_year).values
        sigma_vals  = np.sqrt(sigma2_vals)

        correction = (lambda_ * sigma_vals / an) * (1 - np.exp(-an * t_local))
        mu_T    = float(18 * len(t_vals) - np.sum(Tm_vals) + np.sum(correction))
        var_T   = np.sum((sigma2_vals / (2 * an)) * (1 - np.exp(-2 * an * t_local)))
        sigma_T = float(np.sqrt(var_T))

        for k_idx, K in enumerate(K_vals):
            price_surface[t_idx, k_idx] = HDD_call_price(mu_T, sigma_T, K, r, T_days / 365)

    K_grid, T_grid = np.meshgrid(K_vals, T_maturities)
    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(K_grid, T_grid, price_surface, cmap='viridis')
    ax.set_title(f"Winter {i+1} ({year}–{year+1}) — HDD Call Price Surface")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturity (days)")
    ax.set_zlabel("Call Price")
    plt.tight_layout()
    plt.show()
    
# chiamata

# ── TEMPERATURE RESIDUALS (for external use) ──────────────────────────────────
resid_temp = T_stoch - T_det_sim
df_temp    = pd.DataFrame({
    'date'      : pd.date_range(start='1985-01-01', periods=len(T_stoch), freq='D'),
    'resid_temp': resid_temp
})
df_temp = df_temp[df_temp['date'].dt.dayofweek < 5]
df_temp = df_temp[df_temp['date'].between('2018-01-02', '2025-06-30')]
df_temp = df_temp.reset_index(drop=True)
