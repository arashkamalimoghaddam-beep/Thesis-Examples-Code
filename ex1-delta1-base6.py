import time
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify
from itertools import combinations
# Example 1 increase 7
# ثبت زمان شروع اجرا
start_time = time.time()

# -------------------------------------------------
# Symbols & Setup
# -------------------------------------------------
eta, tau, t = sp.symbols('eta tau t', real=True)

print("1. Initializing setup and symbolic expressions for Problem 1...")

# استفاده از sp.Rational برای کسرها جهت جلوگیری از خطای ExactQuotientFailed
F0_expr = -sp.Rational(1, 2) * sp.exp(-2 - eta) * (1 - 3*sp.exp(2) - sp.exp(eta) + sp.exp(2 + eta))
F0 = sp.expand(F0_expr)

# N(F0) = - integral_0^1 F(tau)^2 dtau
F0_tau = F0.subs(eta, tau)
I_0_val = sp.integrate(F0_tau**2, (tau, 0, 1))
N_F0 = -I_0_val

# -------------------------------------------------
# Candidate basis (1 base for 1 + 10 exponential bases)
# -------------------------------------------------
CandidateBasis = [sp.exp(0)]  # پایه ثابت 1
for i in range(1, 11):
    CandidateBasis.append(eta**i * sp.exp(-i * eta))

eta_points = np.linspace(0, 1, 25)

# -------------------------------------------------
# Execution: Step 1 (Symbolic Pre-calculations)
# -------------------------------------------------
print(f"2. Calculating symbolic vectors for {len(CandidateBasis)} bases (Please wait)...")

vec_d1_list = []
f1_d1_list = []
vec_d2_list = []
f1_d2_list = []

# بردار ثابت b_vec
b_vec_expr = I_0_val
b_vec_fn = lambdify(eta, b_vec_expr, 'numpy')
b_vec = b_vec_fn(eta_points)
if np.isscalar(b_vec):
    b_vec = np.full_like(eta_points, b_vec)

for idx, phi in enumerate(CandidateBasis):
    # ------------------ Delta 1 Contribution ------------------
    integrand_1 = sp.expand((-phi.subs(eta, t) * N_F0) * sp.exp(t))
    f1_1 = sp.exp(-eta) * sp.integrate(integrand_1, (t, 0, eta))
    f1_1 = sp.expand(f1_1)

    integrand_tau_1 = sp.expand(F0_tau * f1_1.subs(eta, tau))
    I_f1_1 = -2 * sp.integrate(integrand_tau_1, (tau, 0, 1))

    V1_expr = (-phi * N_F0) + I_f1_1

    vec_v1 = lambdify(eta, V1_expr, 'numpy')(eta_points)
    if np.isscalar(vec_v1): vec_v1 = np.full_like(eta_points, vec_v1)
    vec_d1_list.append(vec_v1)
    f1_d1_list.append(f1_1)

    # ------------------ Delta 2 Contribution ------------------
    integrand_2 = sp.expand((-phi.subs(eta, t)) * sp.exp(t))
    f1_2 = sp.exp(-eta) * sp.integrate(integrand_2, (t, 0, eta))
    f1_2 = sp.expand(f1_2)

    integrand_tau_2 = sp.expand(F0_tau * f1_2.subs(eta, tau))
    I_f1_2 = -2 * sp.integrate(integrand_tau_2, (tau, 0, 1))

    V2_expr = (-phi) + I_f1_2

    vec_v2 = lambdify(eta, V2_expr, 'numpy')(eta_points)
    if np.isscalar(vec_v2): vec_v2 = np.full_like(eta_points, vec_v2)
    vec_d2_list.append(vec_v2)
    f1_d2_list.append(f1_2)

# -------------------------------------------------
# Execution: Step 2 (Optimized Combinatorial Search)
# -------------------------------------------------
print("3. Searching best combination via matrix Least-Squares with non-linear corrections...")

tau_fine = np.linspace(0, 1, 300)
f1_d1_fine = [lambdify(eta, f, 'numpy')(tau_fine) for f in f1_d1_list]
f1_d1_fine = [np.full_like(tau_fine, fn) if np.isscalar(fn) else fn for fn in f1_d1_fine]
f1_d2_fine = [lambdify(eta, f, 'numpy')(tau_fine) for f in f1_d2_list]
f1_d2_fine = [np.full_like(tau_fine, fn) if np.isscalar(fn) else fn for fn in f1_d2_fine]

best_d1_indices = None
best_d2_index = None
min_error = float('inf')
best_coeffs = None
best_int_F1_sq = 0

# تغییر ابعاد ماتریس به 7 ستون (6 تا برای دلتا 1 و 1 تا برای دلتا 2)
A_mat = np.empty((len(eta_points), 7))

# تغییر تعداد ترکیب‌ها برای دلتا 1 به 6 عدد (C1 تا C6)
for idx_combo_d1 in combinations(range(len(CandidateBasis)), 6):
    for idx_d2 in range(len(CandidateBasis)):
        if idx_d2 in idx_combo_d1:
            continue

        A_mat[:, 0] = vec_d1_list[idx_combo_d1[0]]
        A_mat[:, 1] = vec_d1_list[idx_combo_d1[1]]
        A_mat[:, 2] = vec_d1_list[idx_combo_d1[2]]
        A_mat[:, 3] = vec_d1_list[idx_combo_d1[3]]
        A_mat[:, 4] = vec_d1_list[idx_combo_d1[4]]
        A_mat[:, 5] = vec_d1_list[idx_combo_d1[5]]
        A_mat[:, 6] = vec_d2_list[idx_d2]

        b_eff = b_vec.copy()

        # اصلاح غیرخطی با ساختار تکرار ثابت (Picard Iteration)
        for _ in range(6):
            coeffs, residuals, _, _ = np.linalg.lstsq(A_mat, b_eff, rcond=None)

            F1_fine = (coeffs[0] * f1_d1_fine[idx_combo_d1[0]] +
                       coeffs[1] * f1_d1_fine[idx_combo_d1[1]] +
                       coeffs[2] * f1_d1_fine[idx_combo_d1[2]] +
                       coeffs[3] * f1_d1_fine[idx_combo_d1[3]] +
                       coeffs[4] * f1_d1_fine[idx_combo_d1[4]] +
                       coeffs[5] * f1_d1_fine[idx_combo_d1[5]] +
                       coeffs[6] * f1_d2_fine[idx_d2])

            int_F1_sq = np.trapz(F1_fine**2, tau_fine)
            b_eff = b_vec + int_F1_sq

        err = residuals[0] if residuals.size > 0 else np.sum((A_mat @ coeffs - b_eff)**2)

        if err < min_error:
            min_error = err
            best_d1_indices = idx_combo_d1
            best_d2_index = idx_d2
            best_coeffs = coeffs.copy()
            best_int_F1_sq = int_F1_sq

# -------------------------------------------------
# Final Reconstruction & Information Display
# -------------------------------------------------
# استخراج 7 ضریب نهایی
c1, c2, c3, c4, c5, c6, c7 = best_coeffs

selected_bases_delta1 = [CandidateBasis[i] for i in best_d1_indices]
selected_base_delta2 = CandidateBasis[best_d2_index]

delta1_sym = (c1 * selected_bases_delta1[0] +
              c2 * selected_bases_delta1[1] +
              c3 * selected_bases_delta1[2] +
              c4 * selected_bases_delta1[3] +
              c5 * selected_bases_delta1[4] +
              c6 * selected_bases_delta1[5])
delta2_sym = c7 * selected_base_delta2

F1_sym = (c1 * f1_d1_list[best_d1_indices[0]] +
          c2 * f1_d1_list[best_d1_indices[1]] +
          c3 * f1_d1_list[best_d1_indices[2]] +
          c4 * f1_d1_list[best_d1_indices[3]] +
          c5 * f1_d1_list[best_d1_indices[4]] +
          c6 * f1_d1_list[best_d1_indices[5]] +
          c7 * f1_d2_list[best_d2_index])

F_final_sym = F0 + F1_sym

print("\n" + "="*80)
print("SELECTION INFORMATION (Problem 1):")
print(f"Bases chosen for Delta 1 (C1 to C6):\n {selected_bases_delta1}")
print(f"Basis chosen for Delta 2 (C7): {selected_base_delta2}")
print("-" * 80)
print(f"C1: {c1:.16e}\nC2: {c2:.16e}\nC3: {c3:.16e}\nC4: {c4:.16e}\nC5: {c5:.16e}\nC6: {c6:.16e}\nC7: {c7:.16e}")
print("-" * 80)
print(f"Delta 1 (Symbolic) = {sp.simplify(delta1_sym)}")
print(f"Delta 2 (Symbolic) = {sp.simplify(delta2_sym)}")
print(f"Minimum Residual Norm: {min_error:.2e}")
print("-" * 80)
print(f"F1 (Expanded) = \n{sp.expand(F1_sym)}")
print("="*80)

F1_at_zero = sp.simplify(F1_sym.subs(eta, 0))
F_at_zero = sp.simplify(F_final_sym.subs(eta, 0))
print(f"Boundary Condition Check: F1(0) = {F1_at_zero}  (Target: 0)")
print(f"Boundary Condition Check: F(0)  = {F_at_zero}  (Target: 1)")
print("="*80)

# -------------------------------------------------
# Data Generation & Text Table
# -------------------------------------------------
eta_plot = np.linspace(0, 1, 150)
F_num = lambdify(eta, F_final_sym, 'numpy')

eta_table = np.array([0, 0.16, 0.32, 0.48, 0.64, 0.8, 0.96, 1])
exact_table = np.exp(-eta_table)
approx_table = F_num(eta_table)

Error_sym_expr = sp.Abs(sp.exp(-eta) - F_final_sym)

error_table = [float(Error_sym_expr.subs(eta, val).evalf(30)) for val in eta_table]
error_plot_high_prec = np.array([float(Error_sym_expr.subs(eta, val).evalf(30)) for val in eta_plot])

# -------------------------------------------------
# TRUE Continuous Residual Calculation
# -------------------------------------------------
F_final_prime_sym = sp.diff(F_final_sym, eta)
F_final_prime_num = lambdify(eta, F_final_prime_sym, 'numpy')

tau_ultra_fine = np.linspace(0, 1, 2500)
integral_val_true = np.trapz(F_num(tau_ultra_fine)**2, tau_ultra_fine)

true_residual_curve = F_final_prime_num(eta_plot) + F_num(eta_plot) - integral_val_true

print(f"\n{'eta':<10} | {'Exact (e^-eta)':<20} | {'OAFM Approx':<20} | {'Absolute Error':<15}")
print("-" * 75)
for i in range(len(eta_table)):
    print(f"{eta_table[i]:<10.4f} | {exact_table[i]:<20.6f} | {approx_table[i]:<20.6f} | {error_table[i]:<15.6e}")

max_err = np.max(error_plot_high_prec)
mean_err = np.mean(error_plot_high_prec)
l2_err = np.linalg.norm(error_plot_high_prec)

print("\n" + "="*80)
print("GLOBAL ERROR METRICS (Evaluated over 150 points):")
print(f"Maximum Absolute Error (L_inf): {max_err:.16e}")
print(f"Mean Absolute Error (MAE):      {mean_err:.16e}")
print(f"Euclidean Error Norm (L2):      {l2_err:.16e}")
print("="*80)
print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")
print("="*80)

# -------------------------------------------------
# Plots
# -------------------------------------------------
plt.figure(figsize=(14, 5))

# نمودار ۱: خطای مطلق
plt.subplot(1, 2, 1)
plt.plot(eta_plot, error_plot_high_prec, color='darkgreen', linewidth=2)
plt.yscale('log', nonpositive='clip')
plt.title("Absolute Error - Problem 1 (Log Scale)", fontweight="bold")
plt.xlabel("$\\eta$")
plt.ylabel("$|F_{approx} - F_{exact}|$")
plt.minorticks_on()
plt.grid(True, which='major', color='black', linestyle='-', alpha=0.3)
plt.grid(True, which='minor', color='gray', linestyle=':', alpha=0.5)

# نمودار ۲: باقیمانده واقعی عملگر دیفرانسیلی
ode_residual = np.abs(F_final_prime_num(eta_plot) + F_num(eta_plot))

plt.subplot(1, 2, 2)
plt.plot(eta_plot, ode_residual + 1e-17, 'r-', linewidth=2, label="|L(F)| = |F' + F|")
plt.yscale('log', nonpositive='clip')
plt.title("True Differential Residual $|F' + F|$ (Log Scale)", fontweight="bold")
plt.xlabel("$\\eta$")
plt.ylabel("Absolute Residual")
plt.legend()
plt.minorticks_on()
plt.grid(True, which='major', color='black', linestyle='-', alpha=0.3)
plt.grid(True, which='minor', color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()