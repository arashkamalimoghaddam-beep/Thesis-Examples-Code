import time
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify
from itertools import combinations

#example 3
# ثبت زمان شروع اجرا
start_time = time.time()

# -------------------------------------------------
# Symbols
# -------------------------------------------------
eta, tau = sp.symbols('eta tau')
pi = sp.pi

# -------------------------------------------------
# F0 (Problem 3)
# -------------------------------------------------
F0 = 1 - eta**2/2 - eta**4/24

# -------------------------------------------------
# N(F0) Calculation
# -------------------------------------------------
F0_prime_tau = sp.diff(F0, eta).subs(eta, tau)
integral_N = sp.integrate((eta * tau) * F0_prime_tau, (tau, 0, pi/2))
C0 = F0  # مقدار C0 همان F0 است
N_F0 = -sp.sin(eta) + (eta * C0)

# -------------------------------------------------
# Candidate basis
# -------------------------------------------------
CandidateBasis = [
    eta**0, eta**1, eta**3, eta**4,
    eta**5, eta**6, eta**7, eta**8, eta**9, eta**10, eta**11, eta**2,
]

eta_points = np.linspace(0, 1, 25)

# -------------------------------------------------
# Helper Function
# -------------------------------------------------
def get_basis_residual_vector(term_type, phi_expr, N_F0_expr, eta_pts):
    if term_type == 'basis':
        source = -phi_expr * N_F0_expr
    elif term_type == 'aux':
        source = -phi_expr
    else:
        source = 0

    if term_type != 'base':
        A, B, C_const = sp.symbols('A B C_const')
        Fpp = sp.integrate(source, eta) + A
        Fp  = sp.integrate(Fpp, eta) + B
        Func = sp.integrate(Fp, eta) + C_const
        eqs = [sp.Eq(Fpp.subs(eta, 0), 0), sp.Eq(Fp.subs(eta, 0), 0), sp.Eq(Func.subs(eta, 0), 0)]
        sol = sp.solve(eqs, [A, B, C_const])
        F_comp = Func.subs(sol)
    else:
        F_comp = 0

    F_target = F0 + F_comp if term_type == 'base' else F_comp
    integral_part = sp.integrate((eta * tau) * sp.diff(F_target, eta).subs(eta, tau), (tau, 0, pi/2))

    if term_type == 'base':
        R_expr = sp.diff(F_target, eta, 3) - sp.sin(eta) + eta + integral_part
    else:
        R_expr = sp.diff(F_target, eta, 3) + integral_part

    func_num = lambdify(eta, R_expr, 'numpy')
    vec = func_num(eta_pts)
    if np.isscalar(vec): vec = np.full_like(eta_pts, vec)
    return vec, F_comp

# -------------------------------------------------
# Execution: Step 1
# -------------------------------------------------
print("1. Calculating vectors (This may take some time due to symbolic integration)...")
b_vec_raw, _ = get_basis_residual_vector('base', 0, N_F0, eta_points)
b_vec = -b_vec_raw

# لیست‌های جداگانه برای دلتا 1 (نوع basis) و دلتا 2 (نوع aux)
candidate_vectors_d1 = []
candidate_funcs_d1 = []
candidate_vectors_d2 = []
candidate_funcs_d2 = []

for base in CandidateBasis:
    # بردارهای مربوط به دلتا 1 (c1, c2, c3, c4, c5)
    vec_d1, f_expr_d1 = get_basis_residual_vector('basis', base, N_F0, eta_points)
    candidate_vectors_d1.append(vec_d1)
    candidate_funcs_d1.append(f_expr_d1)

    # بردارهای مربوط به دلتا 2 (c6)
    vec_d2, f_expr_d2 = get_basis_residual_vector('aux', base, N_F0, eta_points)
    candidate_vectors_d2.append(vec_d2)
    candidate_funcs_d2.append(f_expr_d2)

# -------------------------------------------------
# Execution: Step 2 (Optimized Search)
# -------------------------------------------------
print("2. Searching best combination (Optimized)...")
best_d1_indices = None
best_d2_index = None
min_error = float('inf')
best_coeffs = None

# تخصیص حافظه اولیه برای ماتریس (حالا 6 ستون دارد)
num_points = len(eta_points)
A_mat = np.empty((num_points, 6))

# تغییر به انتخاب 5 پایه برای دلتا 1
for idx_combo_d1 in combinations(range(len(CandidateBasis)), 5):
    for idx_d2 in range(len(CandidateBasis)):

        # 1. جلوگیری از انتخاب پایه تکراری بین دلتا 1 و دلتا 2
        if idx_d2 in idx_combo_d1:
            continue

        # 2. جایگذاری مستقیم ستون‌ها (5 ستون برای دلتا 1، 1 ستون برای دلتا 2)
        A_mat[:, 0] = candidate_vectors_d1[idx_combo_d1[0]]
        A_mat[:, 1] = candidate_vectors_d1[idx_combo_d1[1]]
        A_mat[:, 2] = candidate_vectors_d1[idx_combo_d1[2]]
        A_mat[:, 3] = candidate_vectors_d1[idx_combo_d1[3]]
        A_mat[:, 4] = candidate_vectors_d1[idx_combo_d1[4]]
        A_mat[:, 5] = candidate_vectors_d2[idx_d2]

        # حل معادله حداقل مربعات
        coeffs, residuals, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

        # محاسبه خطا
        err = residuals[0] if residuals.size > 0 else np.sum((A_mat @ coeffs - b_vec)**2)

        if err < min_error:
            min_error = err
            best_d1_indices = idx_combo_d1
            best_d2_index = idx_d2
            best_coeffs = coeffs.copy()

# -------------------------------------------------
# Final Reconstruction & Information Display
# -------------------------------------------------
selected_bases_delta1 = [CandidateBasis[i] for i in best_d1_indices]
selected_base_delta2 = CandidateBasis[best_d2_index]

delta1_sym = (best_coeffs[0] * selected_bases_delta1[0] + 
              best_coeffs[1] * selected_bases_delta1[1] + 
              best_coeffs[2] * selected_bases_delta1[2] + 
              best_coeffs[3] * selected_bases_delta1[3] + 
              best_coeffs[4] * selected_bases_delta1[4])
delta2_sym = best_coeffs[5] * selected_base_delta2

# بازسازی تابع نهایی
F_final_sym = F0
F_final_sym += best_coeffs[0] * candidate_funcs_d1[best_d1_indices[0]]
F_final_sym += best_coeffs[1] * candidate_funcs_d1[best_d1_indices[1]]
F_final_sym += best_coeffs[2] * candidate_funcs_d1[best_d1_indices[2]]
F_final_sym += best_coeffs[3] * candidate_funcs_d1[best_d1_indices[3]]
F_final_sym += best_coeffs[4] * candidate_funcs_d1[best_d1_indices[4]]
F_final_sym += best_coeffs[5] * candidate_funcs_d2[best_d2_index]

print("\n" + "="*60)
print("SELECTION INFORMATION (Fully Automated & Optimized):")
print(f"Bases chosen for Delta 1 (c1, c2, c3, c4, c5): {selected_bases_delta1}")
print(f"Basis chosen for Delta 2 (c6): {selected_base_delta2}")
print(f"Coefficient c1: {best_coeffs[0]:.6f}")
print(f"Coefficient c2: {best_coeffs[1]:.6f}")
print(f"Coefficient c3: {best_coeffs[2]:.6f}")
print(f"Coefficient c4: {best_coeffs[3]:.6f}")
print(f"Coefficient c5: {best_coeffs[4]:.6f}")
print(f"Coefficient c6: {best_coeffs[5]:.6f}")
print("-" * 60)
print(f"Delta 1 (Symbolic) = {sp.simplify(delta1_sym)}")
print(f"Delta 2 (Symbolic) = {sp.simplify(delta2_sym)}")
print(f"Minimum Residual Norm: {min_error:.2e}")
print("-" * 60)
# خط اضافه‌شده برای چاپ فرم گسترده F1
print(f"F1 (Expanded) = \n{sp.expand(F_final_sym)}")
print("="*60)

# -------------------------------------------------
# Data Generation & Text Table
# -------------------------------------------------
eta_plot = np.linspace(0, 1, 100)
F_num = lambdify(eta, F_final_sym, 'numpy')

eta_table = np.linspace(0, 1, 11)
exact_table = np.cos(eta_table)
approx_table = F_num(eta_table)
error_table = np.abs(exact_table - approx_table)

best_A_mat = np.column_stack([
    candidate_vectors_d1[best_d1_indices[0]],
    candidate_vectors_d1[best_d1_indices[1]],
    candidate_vectors_d1[best_d1_indices[2]],
    candidate_vectors_d1[best_d1_indices[3]],
    candidate_vectors_d1[best_d1_indices[4]],
    candidate_vectors_d2[best_d2_index]
])
final_residual_vec = best_A_mat @ best_coeffs - b_vec

# چاپ جدول ساده در کنسول
print(f"\n{'eta':<10} | {'Exact (cos)':<15} | {'Approx (F)':<15} | {'Absolute Error':<15}")
print("-" * 65)
for i in range(len(eta_table)):
    print(f"{eta_table[i]:<10.4f} | {exact_table[i]:<15.6f} | {approx_table[i]:<15.6f} | {error_table[i]:<15.2e}")

# محاسبه و چاپ نرم خطا، بیشینه خطا و میانگین خطا
l2_error_norm = np.linalg.norm(error_table)
max_error_norm = np.max(error_table)
mean_error = np.mean(error_table)

print("-" * 65)
print(f"Error Norm (L2 Norm):        {l2_error_norm:.2e}")
print(f"Max Error (L-infinity):      {max_error_norm:.2e}")
print(f"Mean Error (MAE):            {mean_error:.2e}")

# محاسبه و چاپ زمان کل صرف شده
elapsed_time = time.time() - start_time
print("\n" + "-"*60)
print(f"Total Execution Time: {elapsed_time:.2f} seconds")
print("-" * 60)

# -------------------------------------------------
# Plots
# -------------------------------------------------
plt.figure(figsize=(14, 5))

# ---- نمودار اول: نمودار خطا ----
plt.subplot(1, 2, 1)
F_exact_plot = np.cos(eta_plot)
F_approx_plot = F_num(eta_plot)
plt.semilogy(eta_plot, np.abs(F_approx_plot - F_exact_plot), color='blue', linewidth=2)
plt.title("Absolute Error (Log Scale)", fontweight="bold")
plt.xlabel("$\eta$")
plt.ylabel("$|F_{approx} - F_{exact}|$")
plt.grid(True, which='both', linestyle='--', alpha=0.7)

# ---- نمودار دوم: نمودار باقیمانده ----
plt.subplot(1, 2, 2)
plt.plot(eta_points, final_residual_vec, 'go--', linewidth=1.5, markersize=6, label="Residual $Ac - b$")
plt.axhline(0, color='black', linewidth=1)
plt.title("Equation Residual Vector", fontweight="bold")
plt.xlabel("$\eta$")
plt.ylabel("Residual Value")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
