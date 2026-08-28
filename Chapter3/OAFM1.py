import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
#example 1
# =============================================================================
# 1. SETUP SYMBOLIC ENVIRONMENT & CONSTANTS (FORMULA 11, 13, 14)
# =============================================================================
eta, tau = sp.symbols('eta tau', real=True)
C1, C2, C3, C4 = sp.symbols('C1 C2 C3 C4', real=True)

print("--- Step 1: Initial Approximation Setup ---")
# Exact solution for validation
exact_sol = sp.exp(-eta)

# Definitions based on Formula 13
# L(F) = F' + F, g(eta) = -0.5*(e^(-2) - 1), N(F) = - integral F^2
g_eta = -0.5 * (sp.exp(-2) - 1)

# Formula 14 provides the exact initial approximation satisfying F0(0) = 1
F0 = -0.5 * sp.exp(-2 - eta) * (1 - 3*sp.exp(2) - sp.exp(eta) + sp.exp(2 + eta))
F0 = sp.simplify(F0)

print(f"Boundary Condition Check: F0(0) = {F0.subs(eta, 0)}")

# =============================================================================
# 2. COMPUTE F1(eta, C_i) BY INTEGRAL AND REWRITE IN EXTENDED FORM
# =============================================================================
print("\n--- Step 2: Analytical Derivation & Extension of F1(eta, C_i) ---")

# Compute the non-linear operator on F0: N[F0] = - integral_0^1 F0(tau)^2 dtau
F0_tau = F0.subs(eta, tau)
N_F0 = -sp.integrate(F0_tau**2, (tau, 0, 1))

# Auxiliary functions defined for Example 1
Delta_1 = C1 * sp.exp(0) + C2 * eta * sp.exp(-eta) + C3 * eta**2 * sp.exp(-2*eta)
Delta_2 = -C4 * eta**3 * sp.exp(-3*eta)

# Differential relation for F1
# F1_prime(eta) + Delta_1 * N[F0] + Delta_2 = 0  =>  F1_prime = -(Delta_1 * N[F0] + Delta_2)
F1_prime = -(Delta_1 * N_F0 + Delta_2)

# Indefinite integration
F1_raw = sp.integrate(F1_prime, eta)

# Strictly enforce boundary condition: F1(0) = 0
F1_bc_value = F1_raw.subs(eta, 0)
F1 = F1_raw - F1_bc_value

# Rewrite and explicitly output the extended form of F1
F1_extended = sp.expand(F1)
print("\nExtended analytical form of F1(eta, C_i):")
print("=" * 80)
print(F1_extended)
print("=" * 80)

# Total first-order approximate solution: F_tilde = F0 + F1
F_tilde = F0 + F1
print(f"\nTotal Solution Boundary Condition Check: F_tilde(0) = {sp.simplify(F_tilde.subs(eta, 0))}")

# =============================================================================
# 3. LEAST SQUARES OPTIMIZATION METHOD
# =============================================================================
print("\n--- Step 3: Least Squares Optimization ---")
# To handle the non-linear Volterra-like integration in the residual rapidly,
# we lambdify the symbolic expressions and use scipy.optimize on the continuous residual

F_tilde_fn = sp.lambdify((eta, C1, C2, C3, C4), F_tilde, modules=['numpy'])
F_tilde_prime_fn = sp.lambdify((eta, C1, C2, C3, C4), sp.diff(F_tilde, eta), modules=['numpy'])
g_val = float(g_eta)

def N_operator(c1, c2, c3, c4):
    """Computes the integral term N[F_tilde] = - integral_0^1 F_tilde^2 dtau"""
    integrand = lambda t: F_tilde_fn(t, c1, c2, c3, c4)**2
    val, _ = quad(integrand, 0, 1, limit=100)
    return -val

def continuous_residual(t, c1, c2, c3, c4, n_val):
    """Computes the exact residual R(eta) = L(F_tilde) + g(eta) + N(F_tilde)"""
    L_val = F_tilde_prime_fn(t, c1, c2, c3, c4) + F_tilde_fn(t, c1, c2, c3, c4)
    return L_val + g_val + n_val

def objective_function(C):
    """Least Squares Objective: J = integral_0^1 R^2 d_eta"""
    c1, c2, c3, c4 = C
    n_val = N_operator(c1, c2, c3, c4)
    integrand = lambda t: continuous_residual(t, c1, c2, c3, c4, n_val)**2
    val, _ = quad(integrand, 0, 1, limit=100)
    return val

# Run optimization using an initial guess close to the reported article roots
print("Minimizing the squared residual integral...")
initial_guess = [1.1, 0.0, 0.0, 0.0]
result = minimize(objective_function, initial_guess, method='Nelder-Mead', tol=1e-15)

c1_opt, c2_opt, c3_opt, c4_opt = result.x
print(f"Optimized Convergence Control Parameters:")
print(f"C1 = {c1_opt:.15e}")
print(f"C2 = {c2_opt:.15e}")
print(f"C3 = {c3_opt:.15e}")
print(f"C4 = {c4_opt:.15e}")

# =============================================================================
# 4. GENERATE ACCURACY GRID VALUES & TABLE
# =============================================================================
eta_grid = np.array([0.0, 0.16, 0.32, 0.48, 0.64, 0.8, 0.96, 1.0])

print("\n" + "=" * 75)
print(f"{'eta':<5} | {'OAFM Approx':<15} | {'Exact Solution':<15} | {'Abs Error':<15}")
print("-" * 75)

for e in eta_grid:
    approx_val = float(F_tilde_fn(e, c1_opt, c2_opt, c3_opt, c4_opt))
    exact_val = float(np.exp(-e))
    abs_err = abs(approx_val - exact_val)
    print(f"{e:<5.2f} | {approx_val:<15.6f} | {exact_val:<15.6f} | {abs_err:<15.5e}")
print("=" * 75)

# =============================================================================
# 5. DRAW ERROR AND RESIDUAL DIAGRAMS
# =============================================================================
eta_plot = np.linspace(0.0, 1.0, 200)
n_val_opt = N_operator(c1_opt, c2_opt, c3_opt, c4_opt)

# Generate data arrays for plotting
abs_errors = [abs(F_tilde_fn(e, c1_opt, c2_opt, c3_opt, c4_opt) - np.exp(-e)) for e in eta_plot]
residuals = [continuous_residual(e, c1_opt, c2_opt, c3_opt, c4_opt, n_val_opt) for e in eta_plot]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Absolute Error
ax1.plot(eta_plot, abs_errors, color='navy', linewidth=2, label='Abs errors')
ax1.set_xlabel(r'$\eta$', fontsize=12)
ax1.set_ylabel('Absolute Error', fontsize=12)
ax1.set_title('First-order OAFM absolute error', fontsize=12, fontweight='bold')
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()

# Plot 2: Residual
ax2.plot(eta_plot, residuals, color='navy', linewidth=2, label='Residual')
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_xlabel(r'$\eta$', fontsize=12)
ax2.set_ylabel('Residual', fontsize=12)
ax2.set_title('Residual of first-order OAFM', fontsize=12, fontweight='bold')
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()
