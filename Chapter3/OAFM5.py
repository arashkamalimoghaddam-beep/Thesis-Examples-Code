import sympy as sp
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt
#example 5
def solve_oafm_example_3():
    # ---------------------------------------------------------
    # 1. Symbolic Setup & Problem Definition
    # ---------------------------------------------------------
    eta, tau, C1, C2, C3 = sp.symbols('eta tau C1 C2 C3', real=True)
    
    # Exact solution for comparison
    F_exact = sp.exp(-eta)
    
    # The source equation component g(eta) derived from F0
    g_eta = sp.exp(-eta) - sp.exp(-1) + 1
    
    # ---------------------------------------------------------
    # 2. Initial Approximation F0(eta)
    # ---------------------------------------------------------
    # Solving F0'(eta) + g(eta) = 0
    F0_prime = -g_eta
    F0_indef = sp.integrate(F0_prime, eta)
    
    # Observe Boundary Condition: F0(0) = 1
    C_0 = 1 - F0_indef.subs(eta, 0)
    F0 = F0_indef + C_0
    
    print("==================================================")
    print("1. Value of F_0(eta):")
    print(f"F_0(eta) = {F0}")
    print("==================================================\n")

    # ---------------------------------------------------------
    # 3. Find F1(eta, C_i) via Integral and Rewrite in Extended Form
    # ---------------------------------------------------------
    # Nonlinear component evaluated at F0
    N_F0 = -sp.integrate(F0.subs(eta, tau), (tau, 0, 1))
    
    # Auxiliary functions (Bases selection)
    Delta1 = C1 * eta * sp.exp(eta) + C2 * eta * sp.exp(2*eta)
    Delta2 = -C3
    
    print("==================================================")
    print("2. Bases selected for the Deltas:")
    print(f"Delta_1 = {Delta1}")
    print(f"Delta_2 = {Delta2}")
    print("==================================================\n")
    
    # F1'(eta) formulation
    F1_prime = -Delta1 * N_F0 - Delta2
    
    # Observe Boundary Condition for F1: F1(0) MUST be 0.
    x = sp.symbols('x')
    F1 = sp.integrate(F1_prime.subs(eta, x), (x, 0, eta))
    
    # Rewrite in extended form
    F1_extended = sp.expand(F1)
    print("==================================================")
    print("3. Extended numerical value of F_1(eta, C_i):")
    print(F1_extended)
    print("==================================================\n")

    # ---------------------------------------------------------
    # 4. Formulate the Residual (Formula 14)
    # ---------------------------------------------------------
    F_tilde = F0 + F1
    F_tilde_prime = sp.diff(F_tilde, eta)
    
    # Integral of F_tilde over [0, 1]
    int_F_tilde = sp.integrate(F_tilde.subs(eta, tau), (tau, 0, 1))
    
    # Residual R(eta, C1, C2, C3)
    R = F_tilde_prime + sp.exp(-eta) - sp.exp(-1) + 1 - int_F_tilde
    
    # Convert Symbolic Residual to a numerical function
    R_func = sp.lambdify((eta, C1, C2, C3), R, 'numpy')

    # ---------------------------------------------------------
    # 5. Least Squares Method (Formula 13)
    # ---------------------------------------------------------
    def cost_function(C_vals):
        c1, c2, c3 = C_vals
        # Integral of R^2
        integrand = lambda e: R_func(e, c1, c2, c3)**2
        J_val, _ = quad(integrand, 0, 1)
        return J_val

    # Minimize J to find optimal C1, C2, C3
    initial_guess = [0.0, 0.0, 0.0]
    result = minimize(cost_function, initial_guess, method='Nelder-Mead', tol=1e-15)
    opt_C1, opt_C2, opt_C3 = result.x
    
    print("==================================================")
    print("4. Numerical values of C_i (Least Squares Optimal):")
    print(f"C_1 = {opt_C1:.15e}")
    print(f"C_2 = {opt_C2:.15e}")
    print(f"C_3 = {opt_C3:.15e}")
    print("==================================================\n")

    # ---------------------------------------------------------
    # 6. Determine Approximations and Errors for each eta (Results Table)
    # ---------------------------------------------------------
    F_exact_func = sp.lambdify(eta, F_exact, 'numpy')
    F_tilde_func = sp.lambdify(eta, F_tilde.subs({C1: opt_C1, C2: opt_C2, C3: opt_C3}), 'numpy')
    F_tilde_prime_num = sp.lambdify((eta, C1, C2, C3), F_tilde_prime, 'numpy')
    F_tilde_num = sp.lambdify((eta, C1, C2, C3), F_tilde, 'numpy')

    # Compute numerical integral to preserve precision scale point-by-point
    opt_int_val, _ = quad(lambda t: F_tilde_num(t, opt_C1, opt_C2, opt_C3), 0, 1)

    eta_test_points = np.arange(0.1, 1.0, 0.1)
    
    print("==================================================")
    print("5. Results Table:")
    print(f"{'eta':<5} | {'Exact Solution':<18} | {'OAFM Approx':<18} | {'Absolute Error':<18} | {'Residual Value':<18}")
    print("-" * 90)
    
    for val in eta_test_points:
        exact_val = F_exact_func(val)
        approx_val = F_tilde_func(val)
        error_val = abs(exact_val - approx_val)
        
        # Calculate pointwise numeric residual value
        res_val = F_tilde_prime_num(val, opt_C1, opt_C2, opt_C3) + np.exp(-val) - np.exp(-1) + 1 - opt_int_val
        
        print(f"{val:.1f}   | {exact_val:.16f} | {approx_val:.16f} | {error_val:.16e} | {res_val:.16e}")
    print("==================================================\n")

    # ---------------------------------------------------------
    # 7. Draw the Pointwise Residual and Error Diagrams
    # ---------------------------------------------------------
    eta_plot = np.linspace(0, 1, 200)
    
    # Calculate point-by-point floating-point residuals to show true granularity
    residual_plot = np.array([
        F_tilde_prime_num(e, opt_C1, opt_C2, opt_C3) + np.exp(-e) - np.exp(-1) + 1 - opt_int_val 
        for e in eta_plot
    ])
    
    # Calculate Error values across the domain
    exact_plot = F_exact_func(eta_plot)
    approx_plot = F_tilde_func(eta_plot)
    error_plot = np.abs(exact_plot - approx_plot)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Pointwise Residual Graph
    ax1.plot(eta_plot, residual_plot, 'g-', linewidth=2, label='Pointwise Residual')
    ax1.set_xlabel(r'$\eta$')
    ax1.set_ylabel(r'$R(\eta)$')
    ax1.set_title('Residual Graph for Example 3')
    ax1.grid(True)
    
    # Adjust limits tightly around structural variance to prevent flat flattening
    y_min, y_max = np.min(residual_plot), np.max(residual_plot)
    y_range = y_max - y_min if (y_max - y_min) > 1e-17 else 1e-16
    ax1.set_ylim(y_min - y_range * 0.2, y_max + y_range * 0.2)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 

    # Plot 2: Absolute Error Graph
    ax2.plot(eta_plot, error_plot, 'b-', linewidth=2)
    ax2.set_xlabel(r'$\eta$')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Absolute Error Diagram')
    ax2.grid(True)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    solve_oafm_example_3()
