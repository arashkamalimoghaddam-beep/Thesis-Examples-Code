import time
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify
from itertools import combinations

# Example 4
# ثبت زمان شروع اجرا
start_time = time.time()

# -------------------------------------------------
# Symbols & Setup
# -------------------------------------------------
eta, tau, t = sp.symbols('eta tau t', real=True)

print("1. Initializing setup and symbolic expressions for Example 1...")

# -------------------------------------------------
# F0 (Initial Approximation)
# -------------------------------------------------
# تعریف تابع g بر اساس فرمول 18 مقاله 
g_expr = eta * sp.exp(eta) + sp.exp(eta) - eta

# محاسبه F0 با حل معادله F0' = g_expr و اعمال شرط مرزی F0(0) = 0
F0_raw = sp.integrate(g_expr, eta)
F0 = F0_raw - F0_raw.subs(eta, 0)

# -------------------------------------------------
# N(F0) Calculation
# -------------------------------------------------
# N[F] = -eta * integral_0^1 [F(tau)] dtau
F0_tau = F0.subs(eta, tau)
I_F0_val = sp.integrate(F0_tau, (tau, 0, 1))
N_F0 = -eta * I_F0_val

# -------------------------------------------------
# Candidate basis (12 bases in total)
# -------------------------------------------------
term = sp.exp(eta) - eta/2
CandidateBasis = [term**i for i in range(1, 13)]

eta_points = np.linspace(0, 1, 25)

# -------------------------------------------------
# Execution: Step 1 (Symbolic Pre-calculations)
# -------------------------------------------------
print(f"2. Calculating symbolic vectors for {len(CandidateBasis)} bases (Please wait, integrating up to power 12)...")

vec_d1_list = []
f1_d1_list = []
vec_d2_list = []
f1_d2_list = []

# بردار سمت راست معادله (b_vec) برای مینیمم‌سازی محلی
# باقیمانده: R = F1' - eta * I_F1 - eta * I_F0 = 0  =>  Ac = eta * I_F0
b_vec_expr = eta * I_F0_val
b_vec_fn = lambdify(eta, b_vec_expr, 'numpy')
b_vec = b_vec_fn(eta_points)
if np.isscalar(b_vec): b_vec = np.full_like(eta_points, b_vec)

for phi in CandidateBasis:
    # ------------------ Delta 1 Contribution ------------------
    # Q1(t) = -phi(t) * N[F0](t)
    Q1_t = -phi.subs(eta, t) * N_F0.subs(eta, t)
    # استفاده از expand برای افزایش چشمگیر سرعت انتگرال‌گیری سیمبولیک
    f1_1 = sp.integrate(sp.expand(Q1_t), (t, 0, eta))

    # محاسبه سهم انتگرالی این پایه در باقیمانده: I_F1 = integral_0^1 F1(tau) dtau
    I_f1_1 = sp.integrate(f1_1.subs(eta, tau), (tau, 0, 1))

    # بردار ارزیابی باقیمانده برای ستون ماتریس: V1 = F1' - eta * I_F1
    V1_expr = Q1_t.subs(t, eta) - eta * I_f1_1

    vec_v1 = lambdify(eta, V1_expr, 'numpy')(eta_points)
    if np.isscalar(vec_v1): vec_v1 = np.full_like(eta_points, vec_v1)
    vec_d1_list.append(vec_v1)
    f1_d1_list.append(f1_1)

    # ------------------ Delta 2 Contribution ------------------
    # Q2(t) = -phi(t)
    Q2_t = -phi.subs(eta, t)
    f1_2 = sp.integrate(sp.expand(Q2_t), (t, 0, eta))

    I_f1_2 = sp.integrate(f1_2.subs(eta, tau), (tau, 0, 1))
    V2_expr = Q2_t.subs(t, eta) - eta * I_f1_2

    vec_v2 = lambdify(eta, V2_expr, 'numpy')(eta_points)
    if np.isscalar(vec_v2): vec_v2 = np.full_like(eta_points, vec_v2)
    vec_d2_list.append(vec_v2)
    f1_d2_list.append(f1_2)

# -------------------------------------------------
# Execution: Step 2 (Optimized Combinatorial Search)
# -------------------------------------------------
print("3. Searching best combination via matrix Least-Squares...")
best_d1_indices = None
best_d2_indices = None
min_error = float('inf')
best_coeffs = None

# تخصیص حافظه برای ماتریس (3 ضریب برای دلتا 1 و 3 ضریب برای دلتا 2)
A_mat = np.empty((len(eta_points), 6))

# ترکیب‌های 3تایی برای دلتا 1 و دلتا 2
for idx_combo_d1 in combinations(range(len(CandidateBasis)), 3):
    for idx_combo_d2 in combinations(range(len(CandidateBasis)), 3):

        # جلوگیری از تداخل پایه‌ها در دلتا 1 و 2 (استفاده از پایه‌های متمایز)
        if set(idx_combo_d1).intersection(idx_combo_d2):
            continue

        A_mat[:, 0] = vec_d1_list[idx_combo_d1[0]]
        A_mat[:, 1] = vec_d1_list[idx_combo_d1[1]]
        A_mat[:, 2] = vec_d1_list[idx_combo_d1[2]]
        A_mat[:, 3] = vec_d2_list[idx_combo_d2[0]]
        A_mat[:, 4] = vec_d2_list[idx_combo_d2[1]]
        A_mat[:, 5] = vec_d2_list[idx_combo_d2[2]]

        coeffs, residuals, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        err = residuals[0] if residuals.size > 0 else np.sum((A_mat @ coeffs - b_vec)**2)

        if err < min_error:
            min_error = err
            best_d1_indices = idx_combo_d1
            best_d2_indices = idx_combo_d2
            best_coeffs = coeffs.copy()

# -------------------------------------------------
# Final Reconstruction & Information Display
# -------------------------------------------------
c1, c2, c3, c4, c5, c6 = best_coeffs

selected_bases_delta1 = [CandidateBasis[i] for i in best_d1_indices]
selected_bases_delta2 = [CandidateBasis[i] for i in best_d2_indices]

delta1_sym = (c1 * selected_bases_delta1[0] +
              c2 * selected_bases_delta1[1] +
              c3 * selected_bases_delta1[2])

delta2_sym = (c4 * selected_bases_delta2[0] +
              c5 * selected_bases_delta2[1] +
              c6 * selected_bases_delta2[2])

# بازسازی F1 و تابع نهایی
F1_sym = (c1 * f1_d1_list[best_d1_indices[0]] +
          c2 * f1_d1_list[best_d1_indices[1]] +
          c3 * f1_d1_list[best_d1_indices[2]] +
          c4 * f1_d2_list[best_d2_indices[0]] +
          c5 * f1_d2_list[best_d2_indices[1]] +
          c6 * f1_d2_list[best_d2_indices[2]])

F_final_sym = F0 + F1_sym

print("\n" + "="*80)
print("SELECTION INFORMATION (Fully Automated & Optimized):")
print(f"Bases chosen for Delta 1 (C1, C2, C3):\n {selected_bases_delta1}")
print(f"Bases chosen for Delta 2 (C4, C5, C6):\n {selected_bases_delta2}")
print("-" * 80)
for i, val in enumerate(best_coeffs, 1):
    print(f"C{i} = {val:.10f}")
print("-" * 80)
print(f"Delta 1 (Symbolic) = {sp.simplify(delta1_sym)}")
print(f"Delta 2 (Symbolic) = {sp.simplify(delta2_sym)}")
print(f"Minimum Residual Norm: {min_error:.2e}")
print("-" * 80)
print("="*80)

# تایید شرط مرزی F1(0) = 0 در خروجی
F1_at_zero = sp.simplify(F1_sym.subs(eta, 0))
print(f"Boundary Condition Check: F1(0) = {F1_at_zero}")
print("="*80)

# -------------------------------------------------
# Data Generation & Text Table
# -------------------------------------------------
eta_plot = np.linspace(0, 1, 100)
F_num = lambdify(eta, F_final_sym, 'numpy')

eta_table = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
exact_table = eta_table * np.exp(eta_table)
approx_table = F_num(eta_table)
error_table = np.abs(exact_table - approx_table)

# -------------------------------------------------
# True Residual Calculation (Using g_expr)
# -------------------------------------------------
# محاسبه تحلیلی و دقیق باقیمانده معادله (R) دقیقاً مانند مقاله
F_final_diff = sp.diff(F_final_sym, eta)
I_F_final = sp.integrate(F_final_sym.subs(eta, tau), (tau, 0, 1))

# R = F'(\eta) - g(\eta) - \int K(\eta, \tau) F(\tau) d\tau
True_Residual_sym = F_final_diff - g_expr - (eta * I_F_final)

R_num = lambdify(eta, True_Residual_sym, 'numpy')
final_residual_vec = R_num(eta_points)

# هندل کردن زمانی که خروجی lambdify به دلیل صفر شدن کامل متغیر به یک عدد ثابت (اسکالر) تبدیل می‌شود
if np.isscalar(final_residual_vec):
    final_residual_vec = np.full_like(eta_points, final_residual_vec)

# چاپ جدول ساده در کنسول
print(f"\n{'eta':<10} | {'Exact (eta*e^eta)':<20} | {'OAFM Approx':<20} | {'Absolute Error':<15}")
print("-" * 75)
for i in range(len(eta_table)):
    print(f"{eta_table[i]:<10.4f} | {exact_table[i]:<20.6f} | {approx_table[i]:<20.6f} | {error_table[i]:<15.2e}")

# -------------------------------------------------
# Error Metrics Calculation (L2, Max, Mean)
# -------------------------------------------------
l2_error_norm = np.linalg.norm(error_table)
max_error_norm = np.max(error_table)
mean_error_norm = np.mean(error_table)

print("-" * 75)
print(f"Error Norm (L2 Norm):        {l2_error_norm:.2e}")
print(f"Max Error (L-infinity):      {max_error_norm:.2e}")
print(f"Mean Error (MAE):            {mean_error_norm:.2e}")

# محاسبه و چاپ زمان کل
elapsed_time = time.time() - start_time
print("\n" + "-"*80)
print(f"Total Execution Time: {elapsed_time:.2f} seconds")
print("-" * 80)

# -------------------------------------------------
# Plots
# -------------------------------------------------
plt.figure(figsize=(14, 5))

# ---- نمودار اول: نمودار خطا ----
plt.subplot(1, 2, 1)
F_exact_plot = eta_plot * np.exp(eta_plot)
F_approx_plot = F_num(eta_plot)
plt.plot(eta_plot, np.abs(F_approx_plot - F_exact_plot) + 1e-20, color='blue', linewidth=2)
plt.yscale('log')
plt.title("Absolute Error (Log Scale)", fontweight="bold")
plt.xlabel("$\eta$")
plt.ylabel("$|\tilde{F} - \eta e^\eta|$")
plt.grid(True, which='both', linestyle='--', alpha=0.7)

# ---- نمودار دوم: نمودار باقیمانده ----
plt.subplot(1, 2, 2)
# در اینجا از باقیمانده تحلیلی دقیق (True_Residual) استفاده می‌شود
plt.plot(eta_points, final_residual_vec, 'ro--', linewidth=1.5, markersize=6, label="True Residual $R(\eta)$")
plt.axhline(0, color='black', linewidth=1)
plt.title("Equation True Residual Vector", fontweight="bold")
plt.xlabel("$\eta$")
plt.ylabel("Residual Value")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()