import time
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify
from itertools import combinations

# example 2 new - نسخه بدون متغیرهای واسط Q1 و Q2
# ثبت زمان شروع اجرا
start_time = time.time()

# -------------------------------------------------
# Symbols & Setup
# -------------------------------------------------
eta, tau, t = sp.symbols('eta tau t', real=True)

print("1. Initializing setup and symbolic expressions for Problem 2...")

# -------------------------------------------------
# F0 (Problem 2)
# -------------------------------------------------
F0 = sp.exp(-eta) - (eta**3)/3 + (2*eta**3)/(3*sp.exp(1))

# -------------------------------------------------
# N(F0) Calculation
# -------------------------------------------------
# N[F] = -eta^2 * integral_0^1 [tau * F(tau)] dtau
F0_tau = F0.subs(eta, tau)
I_0_val = sp.integrate(tau * F0_tau, (tau, 0, 1))
N_F0 = -eta**2 * I_0_val

# -------------------------------------------------
# Candidate basis (12 بردار پایه در مجموع)
# -------------------------------------------------
CandidateBasis = [sp.exp(0)]
# پایه های نمایی: eta*exp(-eta) تا eta^9*exp(-9*eta)
for i in range(1, 10):
    CandidateBasis.append(eta**i * sp.exp(-i * eta))
# پایه های چندجمله‌ای: eta^1 و eta^2
for i in range(1, 3):
    CandidateBasis.append(eta**i)

eta_points = np.linspace(0, 1, 25)

# -------------------------------------------------
# Execution: Step 1 (Symbolic Pre-calculations)
# -------------------------------------------------
print(f"2. Calculating symbolic vectors for {len(CandidateBasis)} bases (Please wait)...")

vec_d1_list = []
f1_d1_list = []
vec_d2_list = []
f1_d2_list = []

# بردار سمت راست معادله (b_vec)
# فرمول باقیمانده: R = F1' - eta^2 * I_tilde = 0  =>  Ac = eta^2 * I_0
b_vec_expr = eta**2 * I_0_val
b_vec_fn = lambdify(eta, b_vec_expr, 'numpy')
b_vec = b_vec_fn(eta_points)
if np.isscalar(b_vec):
    b_vec = np.full_like(eta_points, b_vec)

for phi in CandidateBasis:
    # ------------------ Delta 1 Contribution ------------------
    # عبارت زیر انتگرال مستقیماً داخل sp.integrate نوشته شده
    # (بدون تعریف متغیر واسط Q1)
    f1_1 = sp.integrate(
        -phi.subs(eta, t) * N_F0.subs(eta, t), (t, 0, eta)
    )
    # محاسبه سهم انتگرالی این پایه در باقیمانده
    I_f1_1 = sp.integrate(tau * f1_1.subs(eta, tau), (tau, 0, 1))
    # بردار ارزیابی باقیمانده: V1 = -phi*N_F0 - eta^2 * int(tau*F1)
    V1_expr = -phi * N_F0 - eta**2 * I_f1_1

    vec_v1 = lambdify(eta, V1_expr, 'numpy')(eta_points)
    if np.isscalar(vec_v1):
        vec_v1 = np.full_like(eta_points, vec_v1)
    vec_d1_list.append(vec_v1)
    f1_d1_list.append(f1_1)

    # ------------------ Delta 2 Contribution ------------------
    # عبارت زیر انتگرال مستقیماً داخل sp.integrate نوشته شده
    # (بدون تعریف متغیر واسط Q2)
    f1_2 = sp.integrate(-phi.subs(eta, t), (t, 0, eta))
    I_f1_2 = sp.integrate(tau * f1_2.subs(eta, tau), (tau, 0, 1))
    V2_expr = -phi - eta**2 * I_f1_2

    vec_v2 = lambdify(eta, V2_expr, 'numpy')(eta_points)
    if np.isscalar(vec_v2):
        vec_v2 = np.full_like(eta_points, vec_v2)
    vec_d2_list.append(vec_v2)
    f1_d2_list.append(f1_2)

# -------------------------------------------------
# Execution: Step 2 (Optimized Combinatorial Search)
# -------------------------------------------------
print("3. Searching best combination via matrix Least-Squares (Over 74,000 combinations)...")
best_d1_indices = None
best_d2_index = None
min_error = float('inf')
best_coeffs = None

# تخصیص حافظه برای سرعت بالاتر ماتریس 25 در 6 (5 ضریب دلتا 1 و 1 ضریب دلتا 2)
A_mat = np.empty((len(eta_points), 6))

for idx_combo_d1 in combinations(range(len(CandidateBasis)), 5):
    for idx_d2 in range(len(CandidateBasis)):

        # جلوگیری از تداخل پایه‌ها
        if idx_d2 in idx_combo_d1:
            continue

        A_mat[:, 0] = vec_d1_list[idx_combo_d1[0]]
        A_mat[:, 1] = vec_d1_list[idx_combo_d1[1]]
        A_mat[:, 2] = vec_d1_list[idx_combo_d1[2]]
        A_mat[:, 3] = vec_d1_list[idx_combo_d1[3]]
        A_mat[:, 4] = vec_d1_list[idx_combo_d1[4]]
        A_mat[:, 5] = vec_d2_list[idx_d2]

        coeffs, residuals, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

        err = residuals[0] if residuals.size > 0 else np.sum((A_mat @ coeffs - b_vec) ** 2)

        if err < min_error:
            min_error = err
            best_d1_indices = idx_combo_d1
            best_d2_index = idx_d2
            best_coeffs = coeffs.copy()

# -------------------------------------------------
# Final Reconstruction & Information Display
# -------------------------------------------------
c1, c2, c3, c4, c5, c6 = best_coeffs

selected_bases_delta1 = [CandidateBasis[i] for i in best_d1_indices]
selected_base_delta2 = CandidateBasis[best_d2_index]

delta1_sym = (c1 * selected_bases_delta1[0] + c2 * selected_bases_delta1[1] +
              c3 * selected_bases_delta1[2] + c4 * selected_bases_delta1[3] +
              c5 * selected_bases_delta1[4])
delta2_sym = c6 * selected_base_delta2

# بازسازی F1 و F_final
F1_sym = (c1 * f1_d1_list[best_d1_indices[0]] +
          c2 * f1_d1_list[best_d1_indices[1]] +
          c3 * f1_d1_list[best_d1_indices[2]] +
          c4 * f1_d1_list[best_d1_indices[3]] +
          c5 * f1_d1_list[best_d1_indices[4]] +
          c6 * f1_d2_list[best_d2_index])

F_final_sym = F0 + F1_sym

print("\n" + "=" * 80)
print("SELECTION INFORMATION (Fully Automated & Optimized):")
print(f"Bases chosen for Delta 1 (C1 to C5):\n {selected_bases_delta1}")
print(f"Basis chosen for Delta 2 (C6): {selected_base_delta2}")

print(f"C1: {c1:.16e}")
print(f"C2: {c2:.16e}")
print(f"C3: {c3:.16e}")
print(f"C4: {c4:.16e}")
print(f"C5: {c5:.16e}")
print(f"C6: {c6:.16e}")
print("-" * 80)

print(f"Delta 1 (Symbolic) = {sp.simplify(delta1_sym)}")
print(f"Delta 2 (Symbolic) = {sp.simplify(delta2_sym)}")
print(f"Minimum Residual Norm: {min_error:.2e}")
print("-" * 80)
print(f"F1 (Expanded) = \n{sp.expand(F1_sym)}")
print("=" * 80)

# تایید شرط مرزی F1(0) = 0 در خروجی
F1_at_zero = sp.simplify(F1_sym.subs(eta, 0))
print(f"Boundary Condition Check: F1(0) = {F1_at_zero}")
print("=" * 80)

# -------------------------------------------------
# Data Generation & Text Table
# -------------------------------------------------
eta_plot = np.linspace(0, 1, 150)
F_num = lambdify(eta, F_final_sym, 'numpy')

eta_table = np.array([0, 0.16, 0.32, 0.48, 0.64, 0.96, 1])
exact_table = np.exp(-eta_table)
approx_table = F_num(eta_table)

# ایجاد عبارت نمادی خطا برای استفاده در جدول و نمودار
Error_sym_expr = sp.Abs(sp.exp(-eta) - F_final_sym)

# استفاده از ارزیابی نمادین با دقت بالا (۳۰ رقم) جهت جلوگیری از Underflow
error_table = [float(Error_sym_expr.subs(eta, val).evalf(30)) for val in eta_table]

# --- محاسبه خطای دقیق شبکه 150 نقطه‌ای برای متریک‌ها و نمودار ---
error_plot_high_prec = np.array([float(Error_sym_expr.subs(eta, val).evalf(30)) for val in eta_plot])

# محاسبه بردار باقیمانده دقیق برای پلات
best_A_mat = np.column_stack([
    vec_d1_list[best_d1_indices[0]],
    vec_d1_list[best_d1_indices[1]],
    vec_d1_list[best_d1_indices[2]],
    vec_d1_list[best_d1_indices[3]],
    vec_d1_list[best_d1_indices[4]],
    vec_d2_list[best_d2_index]
])
final_residual_vec = best_A_mat @ best_coeffs - b_vec

# چاپ جدول ساده در کنسول
print(f"\n{'eta':<10} | {'Exact (e^-eta)':<20} | {'OAFM Approx':<20} | {'Absolute Error':<15}")
print("-" * 75)
for i in range(len(eta_table)):
    print(f"{eta_table[i]:<10.4f} | {exact_table[i]:<20.6f} | {approx_table[i]:<20.6f} | {error_table[i]:<15.6e}")

# =================================================================
# محاسبه و چاپ معیارهای خطا (بیشینه، میانگین، نرم اقلیدسی)
# =================================================================
max_err = np.max(error_plot_high_prec)
mean_err = np.mean(error_plot_high_prec)
l2_err = np.linalg.norm(error_plot_high_prec)

print("\n" + "=" * 80)
print("GLOBAL ERROR METRICS (Evaluated over 150 points):")
print(f"Maximum Absolute Error (L_inf): {max_err:.16e}")
print(f"Mean Absolute Error (MAE):      {mean_err:.16e}")
print(f"Euclidean Error Norm (L2):      {l2_err:.16e}")
print("=" * 80)

# محاسبه و چاپ زمان کل
elapsed_time = time.time() - start_time
print("\n" + "-" * 80)
print(f"Total Execution Time: {elapsed_time:.2f} seconds")
print("-" * 80)

# -------------------------------------------------
# Plots
# -------------------------------------------------
plt.figure(figsize=(14, 5))

# ---- نمودار اول: نمودار خطا ----
plt.subplot(1, 2, 1)
plt.plot(eta_plot, error_plot_high_prec + 1e-35, color='darkgreen', linewidth=2)
plt.yscale('log')
plt.title("Absolute Error (Log Scale)", fontweight="bold")
plt.xlabel("$\\eta$")
plt.ylabel("$|F_{approx} - F_{exact}|$")
plt.grid(True, which='both', linestyle='--', alpha=0.7)

# ---- نمودار دوم: نمودار باقیمانده ----
plt.subplot(1, 2, 2)
plt.plot(eta_points, final_residual_vec, 'ro--', linewidth=1.5, markersize=6, label="Residual $Ac - b$")
plt.axhline(0, color='black', linewidth=1)
plt.title("Equation Residual Vector", fontweight="bold")
plt.xlabel("$\\eta$")
plt.ylabel("Residual Value")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

plt.tight_layout()
plt.show()