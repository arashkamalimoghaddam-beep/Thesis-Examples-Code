import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# =============================================================================
# 1. SETUP SYMBOLIC ENVIRONMENT & CONSTANTS
# =============================================================================
eta, tau = sp.symbols('eta tau', real=True)

# Example 1 uses 6 convergence control parameters: C1, C2, C3, C4, C5, C6
C = sp.symbols('C1:7', real=True)

print("--- Step 1: Initial Zero-Order Approximation (Formula 20) ---")
# FORMULA 20: Initial approximation matching the boundary condition F0(0) = 0
F0 = eta * sp.exp(eta) - (1/2) * eta**2

print("F0(eta) =")
sp.pprint(F0)
print(f"Boundary Condition Check: F0(0) = {F0.subs(eta, 0)}")

# =============================================================================
# 2. COMPUTE INTEGRAL OPERATOR N[F0]
# N[F] = - integral_0^1 [eta * F(tau)] dtau
# =============================================================================
F0_tau = F0.subs(eta, tau)
I_F0 = sp.integrate(F0_tau, (tau, 0, 1))
N_F0 = -eta * I_F0

print("\nIntegral Operator N[F0] =")
sp.pprint(N_F0)

# =============================================================================
# 3. DERIVE F1(eta, C_i) VIA INTEGRATION & REWRITE IN EXTENDED FORM
# =============================================================================
print("\n--- Step 2: Analytical Derivation & Extension of F1(eta, C_i) ---")

# Auxiliary functions chosen in Formula 23
term = sp.exp(eta) - eta/2
Delta_1 = C[0]*term + C[1]*term**2 + C[2]*term**3
Delta_2 = C[3]*term**4 + C[4]*term**5 + C[5]*term**6

# Right-Hand Side of the F1 ODE: F1' = -(Delta_1 * N[F0] + Delta_2)
F1_prime = -(Delta_1 * N_F0 + Delta_2)

# First find the raw indefinite integral
F1_raw = sp.integrate(F1_prime, eta)

# Strictly observe boundary conditions: Adjust to ensure F1(0) = 0
F1_bc_value = F1_raw.subs(eta, 0)
F1 = F1_raw - F1_bc_value

# Rewrite in extended form and explicitly print it
F1_extended = sp.expand(F1)
print("\nExtended analytical form of F1(eta, C_i):")
print("=" * 80)
print(F1_extended)
print("=" * 80)

# Total first-order approximate solution components
F_tilde = F0 + F1
print(f"\nTotal Solution Boundary Condition Check: F_tilde(0) = {F_tilde.subs(eta, 0)}")

# =============================================================================
# 4. LEAST SQUARES OPTIMIZATION METHOD (FORMULAS 13 & 14)
# =============================================================================
print("\n--- Step 3: Least Squares Method Framework ---")

# FORMULA 14: Continuous Residual R(eta, C_i)
g_eta = eta * sp.exp(eta) + sp.exp(eta) - eta
I_tilde = sp.integrate(F_tilde.subs(eta, tau), (tau, 0, 1))
R = sp.expand(F_tilde.diff(eta) - g_eta - eta * I_tilde)

# Decompose the residual into linear basis components for the system matrix
phis = [R.diff(C[i]) for i in range(6)]
psi = R.subs({c: 0 for c in C})

# Convert symbolic functions to NumPy functions for fast numeric quad integration
phi_nums = [sp.lambdify(eta, phi, modules=['numpy']) for phi in phis]
psi_num = sp.lambdify(eta, psi, modules=['numpy'])

# FORMULA 13: System Matrix construction to minimize Residual Square J
A_mat = np.zeros((6, 6))
B_vec = np.zeros(6)

print("Constructing the 6x6 Least Squares Linear System...")
for i in range(6):
    for j in range(6):
        A_mat[i, j], _ = quad(lambda x: float(phi_nums[i](x) * phi_nums[j](x)), 0, 1, limit=100)
    B_vec[i], _ = quad(lambda x: float(-psi_num(x) * phi_nums[i](x)), 0, 1, limit=100)

# Solve system equations for optimal convergence parameters
C_opt = np.linalg.solve(A_mat, B_vec)

print("\nOptimized Convergence Control Parameters:")
for i, val in enumerate(C_opt, 1):
    print(f"C{i} = {val:.15e}")

# =============================================================================
# 5. GENERATE ACCURACY GRID VALUES (TABLE)
# =============================================================================
F_tilde_opt = F_tilde.subs({C[i]: C_opt[i] for i in range(6)})
F_tilde_fn = sp.lambdify(eta, F_tilde_opt, modules=['numpy'])

# Residual function with optimized convergence parameters
R_opt = R.subs({C[i]: C_opt[i] for i in range(6)})
R_opt = sp.expand(R_opt)
R_fn = sp.lambdify(eta, R_opt, modules=['numpy'])


eta_grid = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

print("\n" + "=" * 75)
print(f"{'eta':<5} | {'OAFM Approx':<15} | {'Exact Solution':<20} | {'Absolute Error':<15}")
print("-" * 75)

for e in eta_grid:
    approx_val = float(F_tilde_fn(e))
    exact_val = float(e * np.exp(e)) # Exact solution is eta * e^eta
    abs_err = abs(approx_val - exact_val)
    print(f"{e:<5.1f} | {approx_val:<15.6f} | {exact_val:<20.6f} | {abs_err:<15.5e}")
print("=" * 75)

# =============================================================================
# 6. DRAW ABSOLUTE ERROR DIAGRAM
# =============================================================================
eta_plot = np.linspace(0.0, 1.0, 100)
err_plot = [abs(float(F_tilde_fn(e)) - e * np.exp(e)) + 1e-20 for e in eta_plot]

plt.figure(figsize=(8, 5))
plt.plot(eta_plot, err_plot, color='blue', linewidth=2, label=r'Absolute Error $| \tilde{F}(\eta) - \eta e^\eta |$')
plt.yscale('log')
plt.xlabel(r'$\eta$', fontsize=12)
plt.ylabel('Absolute Error (Log Scale)', fontsize=12)
plt.title('First-Order OAFM Absolute Error for Example 1', fontsize=12, fontweight='bold')
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
# =============================================================================
# 7. DRAW RESIDUAL DIAGRAM
# =============================================================================
residual_plot = [float(R_fn(e)) for e in eta_plot]

plt.figure(figsize=(8,5))
plt.plot(eta_plot, residual_plot,
         color='red',
         linewidth=2,
         label=r'Residual $R(\eta)$')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel(r'$\eta$', fontsize=12)
plt.ylabel(r'$R(\eta)$', fontsize=12)
plt.title('First-Order OAFM Residual for Example 1',
          fontsize=12,
          fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()