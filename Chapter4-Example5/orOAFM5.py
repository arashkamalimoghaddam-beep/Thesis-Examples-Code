import time
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify
from itertools import combinations

# ثبت زمان شروع اجرا
start_time = time.time()

# -------------------------------------------------
# Symbols & Setup
# -------------------------------------------------
eta, tau, t = sp.symbols('eta tau t', real=True)

print("1. Initializing setup and symbolic expressions for Example 5 (Article 2)...")

# -------------------------------------------------
# F0 & Given Functions (Problem 5)
# -------------------------------------------------
# تابع g(eta) بر اساس مثال
g_eta = sp.exp(-eta) - sp.exp(-1) + 1

# محاسبه F0 با استفاده از F0' + g(eta) = 0 و شرط مرزی F0(0) = 1
F0 = 1 - sp.integrate(g_eta.subs(eta, t), (t, 0, eta))

# -------------------------------------------------
# N(F0) Calculation
# -------------------------------------------------
# N[F] = - integral_0^1 [F(tau)] dtau
F0_tau = F0.subs(eta, tau)
I_F0_val = sp.integrate(F0_tau, (tau, 0, 1))
N_F0 = -I_F0_val

# -------------------------------------------------
# Candidate basis (11 bases in total)
# -------------------------------------------------
# پایه‌های درخواستی: 1 و eta*exp(eta) تا eta*exp(10*eta)
CandidateBasis = [sp.sympify(1)]
for i in range(1, 11):
    CandidateBasis.append(eta * sp.exp(i * eta))

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
# باقیمانده: R = F_tilde' + g(eta) - int_0^1 F_tilde dtau = 0
# R0 = F0' + g(eta) - int_0^1 F0 dtau => b_vec = -R0
R0_expr = sp.diff(F0, eta) + g_eta - I_F0_val
b_vec_expr = -R0_expr
b_vec_fn = lambdify(eta, b_vec_expr, 'numpy')
b_vec = b_vec_fn(eta_points)
if np.isscalar(b_vec): b_vec = np.full_like(eta_points, b_vec)

for phi in CandidateBasis:
    # ------------------ Delta 1 Contribution ------------------
    # F1' contribution: Q1(t) = -phi(t) * N[F0](t)
    Q1_t = -phi.subs(eta, t) * N_F0
    # شرط مرزی F1(0) = 0
    f1_1 = sp.integrate(sp.expand(Q1_t), (t, 0, eta))
    
    # محاسبه سهم انتگرالی این پایه در باقیمانده
    I_f1_1 = sp.integrate(f1_1.subs(eta, tau), (tau, 0, 1))
    
    # بردار ارزیابی باقیمانده: V1 = Q1 - int(f1)
    V1_expr = Q1_t.subs(t, eta) - I_f1_1
    
    vec_v1 = lambdify(eta, V1_expr, 'numpy')(eta_points)
    if np.isscalar(vec_v1): vec_v1 = np.full_like(eta_points, vec_v1)
    vec_d1_list.append(vec_v1)
    f1_d1_list.append(f1_1)
    
    # ------------------ Delta 2 Contribution ------------------
    # F1' contribution: Q2(t) = -phi(t)
    Q2_t = -phi.subs(eta, t)
    # شرط مرزی F1(0) = 0
    f1_2 = sp.integrate(sp.expand(Q2_t), (t, 0, eta))
    
    I_f1_2 = sp.integrate(f1_2.subs(eta, tau), (tau, 0, 1))
    V2_expr = Q2_t.subs(t, eta) - I_f1_2
    
    vec_v2 = lambdify(eta, V2_expr, 'numpy')(eta_points)
    if np.isscalar(vec_v2): vec_v2 = np.full_like(eta_points, vec_v2)
    vec_d2_list.append(vec_v2)
    f1_d2_list.append(f1_2)

# -------------------------------------------------
# Execution: Step 2 (Optimized Combinatorial Search)
# -------------------------------------------------
print("3. Searching best combination via matrix Least-Squares...")
best_d1_indices = None
best_d2_index = None
min_error = float('inf')
best_coeffs = None

# تخصیص حافظه برای ماتریس (2 ضریب برای دلتا 1 و 1 ضریب برای دلتا 2)
A_mat = np.empty((len(eta_points), 3))

for idx_combo_d1 in combinations(range(len(CandidateBasis)), 2):
    for idx_d2 in range(len(CandidateBasis)):
        
        # جلوگیری از تداخل و تکرار پایه‌ها
        if idx_d2 in idx_combo_d1:
            continue
            
        A_mat[:, 0] = vec_d1_list[idx_combo_d1[0]]
        A_mat[:, 1] = vec_d1_list[idx_combo_d1[1]]
        A_mat[:, 2] = vec_d2_list[idx_d2]
        
        # حل معادلات برای یافتن C1, C2, C3
        coeffs, residuals, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        err = residuals[0] if residuals.size > 0 else np.sum((A_mat @ coeffs - b_vec)**2)

        if err < min_error:
            min_error = err
            best_d1_indices = idx_combo_d1
            best_d2_index = idx_d2
            best_coeffs = coeffs.copy() 

# -------------------------------------------------
# Final Reconstruction & Information Display
# -------------------------------------------------
c1, c2, c3 = best_coeffs

selected_bases_delta1 = [CandidateBasis[i] for i in best_d1_indices]
selected_base_delta2 = CandidateBasis[best_d2_index]

delta1_sym = c1 * selected_bases_delta1[0] + c2 * selected_bases_delta1[1]
delta2_sym = c3 * selected_base_delta2

# بازسازی F1 و تابع نهایی
F1_sym = (c1 * f1_d1_list[best_d1_indices[0]] + 
          c2 * f1_d1_list[best_d1_indices[1]] + 
          c3 * f1_d2_list[best_d2_index])

F_final_sym = F0 + F1_sym

print("\n" + "="*80)
print("SELECTION INFORMATION (Fully Automated & Optimized):")
print(f"Bases chosen for Delta 1 (C1, C2):\n {selected_bases_delta1}")
print(f"Basis chosen for Delta 2 (C3):\n {selected_base_delta2}")
print("-" * 80)
print(f"C1 = {c1:.15e}")
print(f"C2 = {c2:.15e}")
print(f"C3 = {c3:.15e}")
print("-" * 80)
print(f"Delta 1 (Symbolic) = {sp.simplify(delta1_sym)}")
print(f"Delta 2 (Symbolic) = {sp.simplify(delta2_sym)}")
print(f"Minimum Residual Norm: {min_error:.2e}")
print("-" * 80)
print(f"F1 (Expanded) = \n{sp.expand(F1_sym)}")
print("="*80)

# تایید شرط مرزی F(0) = 1
F_at_zero = sp.simplify(F_final_sym.subs(eta, 0))
print(f"Boundary Condition Check: Total F(0) = {F_at_zero}")
print("="*80)

# -------------------------------------------------
# Data Generation & Text Table
# -------------------------------------------------
eta_plot = np.linspace(0, 1, 100)
F_num = lambdify(eta, F_final_sym, 'numpy')

eta_table = np.linspace(0, 1, 11)
exact_table = np.exp(-eta_table) # Exact solution F_exact = e^(-eta)
approx_table = F_num(eta_table)

# محاسبه خطای مطلق به‌صورت نمادی جهت جلوگیری از صفر شدن ناشی از حدود دقت float64
error_sym = sp.Abs(sp.expand(F_final_sym - sp.exp(-eta)))
error_table = np.array([float(error_sym.subs(eta, val).evalf(30)) for val in eta_table])

# محاسبه بردار باقیمانده دقیق نهایی برای پلات
best_A_mat = np.column_stack([
    vec_d1_list[best_d1_indices[0]],
    vec_d1_list[best_d1_indices[1]],
    vec_d2_list[best_d2_index]
])
final_residual_vec = best_A_mat @ best_coeffs - b_vec

# چاپ جدول در کنسول با دقت بالا
print(f"\n{'eta':<10} | {'Exact (e^-eta)':<22} | {'OAFM Approx':<22} | {'Absolute Error':<20}")
print("-" * 80)
for i in range(len(eta_table)):
    print(f"{eta_table[i]:<10.4f} | {exact_table[i]:<22.10f} | {approx_table[i]:<22.10f} | {error_table[i]:<20.6e}")

# محاسبه معیارهای عددی خطا
max_error = np.max(error_table)
mean_error = np.mean(error_table)
norm_error = np.linalg.norm(error_table)

print("\n" + "="*80)
print("ERROR METRICS & F0 DISPLAY:")
print(f"F0(eta) = {sp.expand(F0)}")
print("-" * 80)
print(f"L2 Error Norm (نرم خطا): {norm_error:.6e}")
print(f"Maximum Absolute Error (بیشینه خطا): {max_error:.6e}")
print(f"Mean Absolute Error (میانگین خطا): {mean_error:.6e}")
print("="*80)

# محاسبه و چاپ زمان کل
elapsed_time = time.time() - start_time
print("\n" + "-"*80)
print(f"Total Execution Time: {elapsed_time:.2f} seconds")
print("-" * 80)

# -------------------------------------------------
# Plots (High Precision Evaluation)
# -------------------------------------------------
plt.figure(figsize=(14, 5))

# محاسبه خطای پیوسته با دقت نمادی بالا (30 رقم اعشار) برای نمودار
error_plot_hp = np.array([float(error_sym.subs(eta, val).evalf(30)) for val in eta_plot])

# محاسبه باقی‌مانده پیوسته معادله با دقت بالا
residual_sym = sp.diff(F_final_sym, eta) + g_eta - sp.integrate(F_final_sym.subs(eta, tau), (tau, 0, 1))
residual_plot_hp = np.array([float(residual_sym.subs(eta, val).evalf(30)) for val in eta_plot])
residual_pts_hp = np.array([float(residual_sym.subs(eta, val).evalf(30)) for val in eta_points])

# ---- نمودار اول: نمودار خطا (دقت بالا) ----
plt.subplot(1, 2, 1)
plt.plot(eta_plot, error_plot_hp, color='darkgreen', linewidth=2)
plt.yscale('log')
plt.title("Absolute Error (Log Scale - High Precision)", fontweight="bold")
plt.xlabel("$\eta$")
plt.ylabel("$|\tilde{F} - e^{-\eta}|$")
plt.grid(True, which='both', linestyle='--', alpha=0.7)

# ---- نمودار دوم: نمودار باقیمانده (دقت بالا) ----
plt.subplot(1, 2, 2)
plt.plot(eta_plot, residual_plot_hp, 'r-', linewidth=1.5, label="Continuous Residual $R(\eta)$")
plt.plot(eta_points, residual_pts_hp, 'ro', markersize=5, label="Collocation Points")
plt.axhline(0, color='black', linewidth=1)
plt.title("Equation Residual (High Precision)", fontweight="bold")
plt.xlabel("$\eta$")
plt.ylabel("Residual Value")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
