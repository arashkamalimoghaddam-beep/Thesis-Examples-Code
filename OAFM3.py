import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# example 3 gemini=============================================================================
# 1. SETUP SYMBOLIC ENVIRONMENT
# =============================================================================
eta, tau, t = sp.symbols('eta tau t', real=True)
C1, C2, C3 = sp.symbols('C1 C2 C3', real=True)

print("--- Step 1: Analytical Derivation from Formula 35 ---")

# FORMULA 35: Initial zero-order approximation satisfying nonhomogeneous BCs
F0 = (24 - 12*eta**2 - eta**4) / 24

# =============================================================================
# 2. COMPUTE NONLINEAR OPERATOR N[F0] (Based on Eq 33)
# N[F] = -sin(eta) + integral_0^(pi/2) [eta * tau * F'(tau)] dtau
# =============================================================================
F0_prime_tau = F0.diff(eta).subs(eta, tau)
int_term_F0 = sp.integrate(eta * tau * F0_prime_tau, (tau, 0, sp.pi/2))
N_F0 = -sp.sin(eta) + int_term_F0

# =============================================================================
# 3. FIND F1(eta, C_i) BY CONSECUTIVE INTEGRAL DEFINITIONS
# FORMULA 37 & 38: F1'''(eta) = -(Delta_1 * N[F0] + Delta_2)
# Boundary Conditions: F1(0) = 0, F1'(0) = 0, F1''(0) = 0
# =============================================================================
Delta_1 = C1 * eta**2 + C2 * eta**4
Delta_2 = -C3
F1_triple_prime = -(Delta_1 * N_F0 + Delta_2)

# Performing consecutive integrations from 0 to eta to strictly observe BCs
F1_double_prime = sp.integrate(F1_triple_prime.subs(eta, t), (t, 0, eta))
F1_prime = sp.integrate(F1_double_prime.subs(eta, t), (t, 0, eta))
F1 = sp.integrate(F1_prime.subs(eta, t), (t, 0, eta))

# Displaying F1(eta, C_i) in its extended analytical form
F1_extended = sp.expand(F1)
print("\nExtended analytical form of F1(eta, C_i):")
print("=" * 80)
print(F1_extended)
print("=" * 80)

# =============================================================================
# 4. TOTAL SOLUTION AND RESIDUAL DEFINITION (Formulas 9 & 10)
# F_tilde = F0 + F1
# Formula 10: R(eta, C_i) = F_tilde'''(eta) + g(eta) + N[F_tilde(eta)]
# =============================================================================
print("\n--- Step 2: Least Squares Framework via Formulas 9 & 10 ---")
F_tilde = F0 + F1

F_tilde_prime_tau = F_tilde.diff(eta).subs(eta, tau)
int_term_tilde = sp.integrate(eta * tau * F_tilde_prime_tau, (tau, 0, sp.pi/2))
R = F_tilde.diff(eta, 3) + eta - sp.sin(eta) + int_term_tilde

# Discretizing the residual linearly into basis components: R = C1*phi1 + C2*phi2 + C3*phi3 + psi
phi1 = R.diff(C1)
phi2 = R.diff(C2)
phi3 = R.diff(C3)
psi = R.subs({C1: 0, C2: 0, C3: 0})

# Lambdifying symbolic expressions into numerical functions for quad execution
p1_num = sp.lambdify(eta, phi1, modules=['numpy'])
p2_num = sp.lambdify(eta, phi2, modules=['numpy'])
p3_num = sp.lambdify(eta, phi3, modules=['numpy'])
psi_num = sp.lambdify(eta, psi, modules=['numpy'])
phis = [p1_num, p2_num, p3_num]

# Formula 9: Minimizing J = integral_0^1 [R^2] d_eta -> Converting to A * C = B
A_mat = np.zeros((3, 3))
B_vec = np.zeros(3)

for i in range(3):
    for j in range(3):
        A_mat[i, j], _ = quad(lambda x: phis[i](x) * phis[j](x), 0, 1, limit=250)
    B_vec[i], _ = quad(lambda x: -psi_num(x) * phis[i](x), 0, 1, limit=250)

# Solving the linear system to find optimal parameters
C_opt = np.linalg.solve(A_mat, B_vec)
c1_val, c2_val, c3_val = C_opt

print(f"Optimized Convergence Control Parameters:")
print(f"C1 = {c1_val:.15f}")
print(f"C2 = {c2_val:.15f}")
print(f"C3 = {c3_val:.15f}")

# =============================================================================
# 5. DETERMINING APPROXIMATIONS AND CALCULATING ABSOLUTE ERROR
# =============================================================================
F_tilde_numeric = F_tilde.subs({C1: c1_val, C2: c2_val, C3: c3_val})
F_tilde_fn = sp.lambdify(eta, F_tilde_numeric, modules=['numpy'])

# =============================================================================
# Residual function with optimized convergence parameters
# =============================================================================
R_numeric = sp.expand(R.subs({C1: c1_val, C2: c2_val, C3: c3_val}))
R_fn = sp.lambdify(eta, R_numeric, modules=['numpy'])


eta_grid = np.linspace(0.0, 1.0, 11)
approximations = []
errors = []

print("\n" + "=" * 75)
print(f"{'eta':<5} | {'OAFM Approx':<18} | {'Exact Solution (cos)':<22} | {'Absolute Error':<15}")
print("-" * 75)

for e in eta_grid:
    approx_val = float(F_tilde_fn(e))
    exact_val = float(np.cos(e))
    abs_err = abs(approx_val - exact_val)

    approximations.append(approx_val)
    errors.append(abs_err)

    print(f"{e:<5.1f} | {approx_val:<18.12f} | {exact_val:<22.12f} | {abs_err:<15.5e}")
print("=" * 75)

# =============================================================================
# 6. GENERATING ERROR DIAGRAM
# =============================================================================
plt.figure(figsize=(7, 5))
plt.plot(eta_grid, errors, marker='o', linestyle='-', color='crimson', label=r'Absolute Error $| \tilde{F}(\eta) - \cos(\eta) |$')
plt.yscale('log')  # Standard logarithmic scale for handling convergence error presentation
plt.xlabel(r'$\eta$', fontsize=12)
plt.ylabel('Absolute Error', fontsize=12)
plt.title('Absolute Error Plot of First-Order OAFM for Problem 3', fontsize=11, fontweight='bold')
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

# =============================================================================
# 7. GENERATING RESIDUAL DIAGRAM
# =============================================================================
eta_plot = np.linspace(0.0, 1.0, 200)
residual_plot = [float(R_fn(e)) for e in eta_plot]

plt.figure(figsize=(7,5))
plt.plot(
    eta_plot,
    residual_plot,
    color='navy',
    linewidth=2,
    label=r'Residual $R(\eta)$'
)

plt.axhline(0, color='black', linestyle='--', linewidth=1)

plt.xlabel(r'$\eta$', fontsize=12)
plt.ylabel(r'$R(\eta)$', fontsize=12)
plt.title('Residual Plot of First-Order OAFM for Problem 3',
          fontsize=11,
          fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()