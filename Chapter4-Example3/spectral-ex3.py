import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
#example 3
class ChelyshkovCollocationSolver:
    r"""
    حل‌کننده معادلات دیفرانسیل-انتگرال مرتبه سوم بر اساس روش هم‌محلی چلیشکوف
    استفاده از متدولوژی Projection (تصویرسازی توابع روی پایه‌های چندجمله‌ای با استفاده از ماتریس گرام)
    برای معادله:
    F'''(\eta) = \sin(\eta) - \eta - \int_{0}^{\pi/2} \eta \tau F'(\tau) d\tau
    با شروط اولیه: F(0)=1, F'(0)=0, F''(0)=-1
    """
    def __init__(self, N, M=30, alpha=1, nu=1):
        self.N = N
        self.M = M        # تعداد نقاط گرهی گاوس برای انتگرال‌گیری عددی (M > N)
        self.alpha = alpha
        self.nu = nu
        self.x = sp.Symbol('x')
        self.W = sp.symbols(f'w0:{N+1}')
        # ساخت بردارهای پایه
        self.Phi = [self.chelyshkov_poly(self.N, i, self.x) for i in range(self.N + 1)]

    def chelyshkov_poly(self, N, n, x):
        poly = 0
        for j in range(n, N + 1):
            term1 = (-1)**(j - n)
            term2 = sp.binomial(N - n, j - n)
            term3 = sp.binomial(N + j + 1, N - n)
            poly += term1 * term2 * term3 * x**(j * self.nu)
        return sp.simplify(poly)

    def compute_P_matrix(self):
        P = np.zeros((self.N + 1, self.N + 1))
        for n in range(self.N + 1):
            for k in range(self.N + 1):
                theta_nk = 0
                for j in range(n, self.N + 1):
                    term1 = (-1)**(j - n) * sp.binomial(self.N - n, j - n) * sp.binomial(self.N + j + 1, self.N - n)
                    beta_val = sp.beta(self.alpha, j * self.nu + 1)

                    xi_kj = 0
                    for l in range(k, self.N + 1):
                        num = (-1)**(l - k) * sp.binomial(self.N - k, l - k) * sp.binomial(self.N + l + 1, self.N - k)
                        den = (j + l + 1) * self.nu + self.alpha
                        xi_kj += num / den
                    xi_kj *= self.nu * (2 * k + 1)

                    theta_nk += term1 * float(beta_val) * float(xi_kj)
                P[n, k] = theta_nk
        return P

    def compute_gram_matrix(self):
        """ تشکیل ماتریس جرم (Gram Matrix) با استفاده از کوآدراتور گاوس """
        Q = np.zeros((self.N + 1, self.N + 1))
        z, w = roots_legendre(50) 
        z_mapped = 0.5 * z + 0.5
        w_mapped = 0.5 * w
        
        Phi_vals = np.array([[float(self.Phi[i].subs(self.x, p)) for p in z_mapped] for i in range(self.N + 1)])
        
        for i in range(self.N + 1):
            for j in range(self.N + 1):
                Q[i, j] = np.sum(w_mapped * Phi_vals[i] * Phi_vals[j])
        return Q

    def project_function(self, func_expr, Q_inv):
        """ تصویر کردن یک تابع روی فضای پایه‌های چلیشکوف A^{-1} * b """
        b = np.zeros(self.N + 1)
        z, w = roots_legendre(50)
        z_mapped = 0.5 * z + 0.5
        w_mapped = 0.5 * w
        
        if self.x in func_expr.free_symbols:
            f_func = sp.lambdify(self.x, func_expr, 'numpy')
            f_vals = f_func(z_mapped)
        else:
            f_vals = np.full_like(z_mapped, float(func_expr))
            
        Phi_vals = np.array([[float(self.Phi[i].subs(self.x, p)) for p in z_mapped] for i in range(self.N + 1)])
        
        for i in range(self.N + 1):
            b[i] = np.sum(w_mapped * f_vals * Phi_vals[i])
            
        return Q_inv @ b

    def solve(self):
        Phi_sym = sp.Matrix(self.Phi)
        P_mat = sp.Matrix(self.compute_P_matrix())
        P2_mat = P_mat**2
        P3_mat = P_mat**3

        # 1. تشکیل ماتریس جرم و معکوس آن
        Q = self.compute_gram_matrix()
        Q_inv = np.linalg.inv(Q)

        # 2. محاسبه بردارهای C (تصویر شروط اولیه روی پایه‌ها)
        # شرط اولیه F(0)=1, F'(0)=0, F''(0)=-1 منجر به بسط 1 - x^2/2 می‌شود
        c_expr = 1 - (self.x**2) / 2
        C_vec = self.project_function(c_expr, Q_inv)
        C_mat = sp.Matrix(C_vec).T

        c_prime_expr = -self.x
        C_prime_vec = self.project_function(c_prime_expr, Q_inv)
        C_prime_mat = sp.Matrix(C_prime_vec).T

        # 3. محاسبه بردار G (تصویر ترم غیر همگن g(x) = sin(x) - x)
        g_expr = sp.sin(self.x) - self.x
        G_vec = self.project_function(g_expr, Q_inv)
        G_mat = sp.Matrix(G_vec).T

        W_mat = sp.Matrix(self.W).T

        # تقریب‌ها بر اساس روابط تصویرشده 
        F_triple_prime_sym = (W_mat * Phi_sym)[0]
        F_prime_sym = ((C_prime_mat + W_mat * P2_mat) * Phi_sym)[0]
        F_sym = ((C_mat + W_mat * P3_mat) * Phi_sym)[0]
        g_sym = (G_mat * Phi_sym)[0]  

        # 4. نقاط هم‌محلی بر اساس ریشه‌های چندجمله‌ای پایه N+1
        C_Nplus1 = self.chelyshkov_poly(self.N + 1, 0, self.x)
        roots = sp.nroots(C_Nplus1)
        x_hat = np.sort([float(sp.re(r)) for r in roots if abs(sp.im(r)) < 1e-10])

        # 5. محاسبه انتگرال فردهلم در بازه [0, pi/2]
        z_leg, w_leg = roots_legendre(self.M)
        z_l = (np.pi / 4.0) * (z_leg + 1.0)
        w_l = (np.pi / 4.0) * w_leg

        tau_integral = 0
        for l in range(self.M):
            F_prime_val = F_prime_sym.subs(self.x, z_l[l])
            # هسته انتگرال شامل \tau است
            tau_integral += w_l[l] * z_l[l] * F_prime_val

        integral_sym = self.x * tau_integral

        # 6. رابطه باقیمانده (Residual)
        # F'''(\eta) - g(\eta) + \eta \int_0^{\pi/2} \tau F'(\tau) d\tau = 0
        R_sym = F_triple_prime_sym - g_sym + integral_sym

        system = []
        for i in range(self.N + 1):
            eq = R_sym.subs(self.x, x_hat[i])
            system.append(eq)

        system_func = sp.lambdify([self.W], system, 'numpy')
        J_sym = sp.Matrix(system).jacobian(self.W)
        J_func = sp.lambdify([self.W], J_sym, 'numpy')

        # 7. مقدار اولیه نیوتن
        W_num = np.zeros(self.N + 1)

        # حلقه روش نیوتن
        for iteration in range(15):
            F_val_vec = np.array(system_func(W_num))
            J_val_mat = np.array(J_func(W_num))
            delta = np.linalg.solve(J_val_mat, -F_val_vec)
            W_num += delta

            if np.linalg.norm(delta) < 1e-14:
                break

        self.W_num = W_num
        self.F_sym = F_sym

    def evaluate(self, x_vals):
        F_num_func = sp.lambdify(self.x, self.F_sym.subs(dict(zip(self.W, self.W_num))), 'numpy')
        return F_num_func(x_vals)

# ==========================================
# اجرای الگوریتم، رسم نمودارها و چاپ جداول
# ==========================================
if __name__ == "__main__":
    N_degree = 5
    solver = ChelyshkovCollocationSolver(N=N_degree, M=30)
    solver.solve()

    table_points = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    approx_vals = solver.evaluate(table_points)
    exact_vals = np.cos(table_points)  # تابع دقیق F(eta) = cos(eta)
    errors = np.abs(exact_vals - approx_vals)

    print("=" * 72)
    print(f"{'eta':<8} | {'Exact F(eta)':<20} | {'Approx F(eta)':<20} | {'Absolute Error':<15}")
    print("=" * 72)
    for i in range(len(table_points)):
        print(f"{table_points[i]:<8.2f} | {exact_vals[i]:<20.10e} | {approx_vals[i]:<20.10e} | {errors[i]:<15.4e}")
    print("=" * 72)

    # جدول LaTeX
    print("\n% LaTeX Table Code:")
    print(r"\begin{table}[ht]")
    print(r"    \centering")
    print(r"    \caption{Comparison of Exact and Approximate Solutions for 3rd order IDE with $N=" + str(N_degree) + r"$}")
    print(r"    \vspace{0.2cm}")
    print(r"    \begin{tabular}{cccc}")
    print(r"        \hline")
    print(r"        $\eta$ & Exact $F(\eta)$ & Approx $F(\eta)$ & Absolute Error \\ \hline")
    for i in range(len(table_points)):
        print(f"        {table_points[i]:.2f} & {exact_vals[i]:.10e} & {approx_vals[i]:.10e} & {errors[i]:.4e} \\\\")
    print(r"        \hline")
    print(r"    \end{tabular}")
    print(r"\end{table}" + "\n")

    # رسم نمودارها
    x_test = np.linspace(0, 1, 100)
    y_approx = solver.evaluate(x_test)
    y_exact = np.cos(x_test)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_test, y_exact, 'b-', linewidth=2, label=r'Exact: $F(\eta) = \cos(\eta)$')
    plt.plot(x_test, y_approx, 'r--', linewidth=2, label=f'App. Sol (N={N_degree})')
    plt.xlabel(r'$\eta$', fontsize=12)
    plt.ylabel(r'$F(\eta)$', fontsize=12)
    plt.title('Comparison of Exact and Approximate Solution')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.subplot(1, 2, 2)
    absolute_error = np.abs(y_exact - y_approx)
    plt.plot(x_test, absolute_error, 'k-', linewidth=1.5, label='Absolute Error')
    plt.yscale('log')
    plt.xlabel(r'$\eta$', fontsize=12)
    plt.ylabel(r'Error $||E_N||$', fontsize=12)
    plt.title('Logarithmic Absolute Error')
    plt.legend()
    plt.grid(True, which="both", linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()
