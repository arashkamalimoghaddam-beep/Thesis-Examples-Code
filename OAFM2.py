import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# problem 2 gemini=============================================================================
# 1. SETUP SYMBOLIC ENVIRONMENT
# =============================================================================
eta, tau, t = sp.symbols('eta tau t', real=True)
C1, C2, C3, C4, C5, C6 = sp.symbols('C1 C2 C3 C4 C5 C6', real=True)

print("--- Step 1: Analytical Derivation starting from Formula 25 ---")

# FORMULA 25: Initial zero-order approximation
# Evaluating F0(0) yields e^0 - 0 + 0 = 1. Boundary condition is strictly met.
F0 = sp.exp(-eta) - (eta**3)/3 + (2*eta**3)/(3*sp.E)

# =============================================================================
# 2. COMPUTE NONLINEAR OPERATOR N[F0] (Formula 26)
# N[F] = - integral_0^1 [eta^2 * tau * F(tau)] dtau
# =============================================================================
F0_tau = F0.subs(eta, tau)
# Extract eta^2 out of the integral since it's a constant with respect to tau
I_0 = sp.integrate(tau * F0_tau, (tau, 0, 1))
N_F0 = -eta**2 * I_0

# =============================================================================
# 3. FIND F1(eta, Ci) USING DEFINITE INTEGRATION TO SECURE BOUNDARY CONDITIONS
# Formula 27: F1'(eta) = -(Delta_1 * N[F0] + Delta_2)
# =============================================================================
# Formula 28: Auxiliary functions
Delta_1 = C1 + ((C2*eta + C3*eta**2 + C4*eta**3 + C5*eta**4) * sp.exp(-eta))
Delta_2 = -C6 * eta**3

# Right-Hand Side of the F1 ODE
F1_prime = -(Delta_1 * N_F0 + Delta_2)

# Integrate from 0 to eta.
# This mathematically guarantees F1(0) = 0 without needing manual constant subtraction.
print("Integrating F1'(t) from 0 to eta... (This may take a moment)")
F1_raw = sp.integrate(F1_prime.subs(eta, t), (t, 0, eta))
F1 = sp.expand(F1_raw)

print("\nExtended analytical form of F1(eta, C_i):")
print("=" * 80)
print(F1)
print("=" * 80)

# =============================================================================
# 4. TOTAL SOLUTION AND EXACT LINEAR RESIDUAL (Formulas 9 & 10)
# =============================================================================
print("\n--- Step 2: Continuous Least Squares Framework via Formulas 9 & 10 ---")
F_tilde = F0 + F1

# For Eq 21, the residual R = F_tilde' - g(eta) + N[F_tilde]
# Because F0' - g(eta) = 0, F_tilde' - g(eta) simplifies perfectly to F1'
# Therefore: R = F1' - eta^2 * integral_0^1 [tau * F_tilde(tau)] dtau
print("Computing the exact symbolic integration of the residual...")
I_tilde = sp.integrate(tau * F_tilde.subs(eta, tau), (tau, 0, 1))
R = F1_prime - eta**2 * I_tilde

# Since R is linear in C_i, we can map it to R = sum(C_i * phi_i) + psi
# Differentiate R with respect to each parameter to extract the basis components
phi1 = R.diff(C1)
phi2 = R.diff(C2)
phi3 = R.diff(C3)
phi4 = R.diff(C4)
phi5 = R.diff(C5)
phi6 = R.diff(C6)
psi = R.subs({C1: 0, C2: 0, C3: 0, C4: 0, C5: 0, C6: 0})

# Lambdify the expressions for high-speed SciPy numerical integration
p1_num = sp.lambdify(eta, phi1, modules=['numpy'])
p2_num = sp.lambdify(eta, phi2, modules=['numpy'])
p3_num = sp.lambdify(eta, phi3, modules=['numpy'])
p4_num = sp.lambdify(eta, phi4, modules=['numpy'])
p5_num = sp.lambdify(eta, phi5, modules=['numpy'])
p6_num = sp.lambdify(eta, phi6, modules=['numpy'])
psi_num = sp.lambdify(eta, psi, modules=['numpy'])
phis = [p1_num, p2_num, p3_num, p4_num, p5_num, p6_num]

# Formula 9: Minimizing J = integral_0^1 [R^2] d_eta -> Matrix System A * C = B
A_mat = np.zeros((6, 6))
B_vec = np.zeros(6)

print("Constructing the 6x6 Least Squares Matrix...")
for i in range(6):
    for j in range(6):
        A_mat[i, j], _ = quad(lambda x: phis[i](x) * phis[j](x), 0, 1, limit=200)
    B_vec[i], _ = quad(lambda x: -psi_num(x) * phis[i](x), 0, 1, limit=200)

# Solve the continuous linear system
C_opt = np.linalg.solve(A_mat, B_vec)
c1_val, c2_val, c3_val, c4_val, c5_val, c6_val = C_opt

print(f"\nOptimized Convergence Control Parameters:")
print(f"C1 = {c1_val:.15f}")
print(f"C2 = {c2_val:.15f}")
print(f"C3 = {c3_val:.15f}")
print(f"C4 = {c4_val:.15f}")
print(f"C5 = {c5_val:.15f}")
print(f"C6 = {c6_val:.15f}")

# =============================================================================
# 5. TABLE GENERATION AND ERROR CALCULATION
# =============================================================================
F_tilde_opt = F_tilde.subs({C1: c1_val, C2: c2_val, C3: c3_val,
                            C4: c4_val, C5: c5_val, C6: c6_val})
F_tilde_fn = sp.lambdify(eta, F_tilde_opt, modules=['numpy'])

# =============================================================================
# Residual function with optimized convergence parameters
# =============================================================================
R_opt = sp.expand(R.subs({
    C1: c1_val,
    C2: c2_val,
    C3: c3_val,
    C4: c4_val,
    C5: c5_val,
    C6: c6_val
}))

R_fn = sp.lambdify(eta, R_opt, modules=['numpy'])

# Grid matching the article's Table 2
eta_grid = np.array([0.16, 0.32, 0.48, 0.64, 0.80, 0.96])
errors = []

print("\n" + "=" * 70)
print(f"{'eta':<5} | {'OAFM Approx':<15} | {'Exact Solution (e^-eta)':<25} | {'Abs Error':<15}")
print("-" * 70)

for e in eta_grid:
    approx_val = float(F_tilde_fn(e))
    exact_val = float(np.exp(-e))
    abs_err = abs(approx_val - exact_val)
    errors.append(abs_err)
    print(f"{e:<5.2f} | {approx_val:<15.6f} | {exact_val:<25.6f} | {abs_err:<15.5e}")
print("=" * 70)

# =============================================================================
# 6. GENERATING ERROR DIAGRAM
# =============================================================================
# Generate a smoother curve for the plot
eta_plot = np.linspace(0, 1, 150)
err_plot = [abs(float(F_tilde_fn(e)) - np.exp(-e)) for e in eta_plot]

plt.figure(figsize=(8, 5))
plt.plot(eta_plot, err_plot, color='darkgreen', linewidth=2, label=r'Absolute Error $| \tilde{F}(\eta) - e^{-\eta} |$')
plt.yscale('log')
plt.xlabel(r'$\eta$', fontsize=12)
plt.ylabel('Absolute Error', fontsize=12)
plt.title('First-Order OAFM Absolute Error for Problem 2', fontsize=12, fontweight='bold')
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
# =============================================================================
# 7. GENERATING RESIDUAL DIAGRAM
# =============================================================================
residual_plot = [float(R_fn(e)) for e in eta_plot]

plt.figure(figsize=(8,5))
plt.plot(
    eta_plot,
    residual_plot,
    color='crimson',
    linewidth=2,
    label=r'Residual $R(\eta)$'
)

plt.axhline(0, color='black', linestyle='--', linewidth=1)

plt.xlabel(r'$\eta$', fontsize=12)
plt.ylabel(r'$R(\eta)$', fontsize=12)
plt.title('First-Order OAFM Residual for Problem 2',
          fontsize=12,
          fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()