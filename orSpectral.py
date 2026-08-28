import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
#example 0
class ChelyshkovCollocationSolver:
    r"""
    حل‌کننده معادلات دیفرانسیل-انتگرال بر اساس روش هم‌محلی چلیشکوف
    نسخه منطبق بر متدولوژی Projection برای معادله کسری:
    D^{1/2} y(x) = \sqrt{\pi}/2 - 1/4 + 1/2 \int_{0}^{1} y^2(t) dt
    با شرط اولیه: y(0) = 0
    جواب دقیق: y(x) = x^{0.5}
    """
    def __init__(self, N, M=30, alpha=0.5, nu=0.5):
        self.N = N
        self.M = M        # تعداد نقاط گرهی گاوس برای انتگرال‌گیری
        self.alpha = alpha # مرتبه مشتق کسری
        self.nu = nu       # پارامتر توان چلیشکوف
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
        """ماتریس عملیاتی انتگرال‌گیری کسری با اصلاح گاما"""
        P = np.zeros((self.N + 1, self.N + 1))
        for n in range(self.N + 1):
            for k in range(self.N + 1):
                theta_nk = 0
                for j in range(n, self.N + 1):
                    term1 = (-1)**(j - n) * sp.binomial(self.N - n, j - n) * sp.binomial(self.N + j + 1, self.N - n)
                    
                    # --- اعمال اصلاح گامای آلفا بر اساس فرمول انتگرال کسری ---
                    beta_val = sp.beta(self.alpha, j * self.nu + 1) / sp.gamma(self.alpha)

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
            # پشتیبانی از مقادیر ثابت با evalf برای عبارات دارای نماد پی
            f_vals = np.full_like(z_mapped, float(func_expr.evalf()))
            
        Phi_vals = np.array([[float(self.Phi[i].subs(self.x, p)) for p in z_mapped] for i in range(self.N + 1)])
        
        for i in range(self.N + 1):
            b[i] = np.sum(w_mapped * f_vals * Phi_vals[i])
            
        return Q_inv @ b

    def solve(self):
        Phi_sym = sp.Matrix(self.Phi)
        P_mat = sp.Matrix(self.compute_P_matrix())

        # 1. تشکیل ماتریس جرم و معکوس آن
        Q = self.compute_gram_matrix()
        Q_inv = np.linalg.inv(Q)

        # 2. محاسبه بردار C (تصویر شرط اولیه y(0) = 0)
        c_expr = sp.sympify(0.0)
        C_vec = self.project_function(c_expr, Q_inv)
        C_mat = sp.Matrix(C_vec).T

        # 3. محاسبه بردار G (تصویر ترم غیر همگن)
        # ترم غیر همگن از تصویر: g(x) = \sqrt{\pi}/2 - 1/4
        g_expr = sp.sqrt(sp.pi)/2 - sp.Rational(1, 4)
        G_vec = self.project_function(g_expr, Q_inv)
        G_mat = sp.Matrix(G_vec).T

        W_mat = sp.Matrix(self.W).T

        # تقریب‌ها بر اساس روابط متدولوژی (W نشان‌دهنده مشتق کسری است)
        D_alpha_y_sym = (W_mat * Phi_sym)[0]
        y_sym = ((C_mat + W_mat * P_mat) * Phi_sym)[0]
        g_sym = (G_mat * Phi_sym)[0]  

        # 4. نقاط هم‌محلی منطبق بر کد اولیه شما
        x_hat = np.array([((i + 1) / (self.N + 2))**(1 / self.nu) for i in range(self.N + 1)])

        # 5. محاسبه انتگرال ترم غیرخطی با نقاط گاوس
        z_leg, w_leg = roots_legendre(self.M)
        z_l = 0.5 * z_leg + 0.5
        w_l = 0.5 * w_leg

        integral_sym = 0
        for l in range(self.M):
            y_val = y_sym.subs(self.x, z_l[l])
            # ضریب 1/2 در انتگرال y^2(t) dt از معادله استخراج شده است
            integral_sym += 0.5 * w_l[l] * (y_val ** 2)

        # رابطه باقیمانده: D^{1/2}y(x) - g(x) - 1/2 \int_0^1 y^2(t) dt = 0
        R_sym = D_alpha_y_sym - g_sym - integral_sym

        system = []
        for i in range(self.N + 1):
            eq = R_sym.subs(self.x, x_hat[i])
            system.append(eq)

        system_func = sp.lambdify([self.W], system, 'numpy')
        J_sym = sp.Matrix(system).jacobian(self.W)
        J_func = sp.lambdify([self.W], J_sym, 'numpy')

        # 6. مقدار اولیه نیوتن 
        W_num = np.zeros(self.N + 1)

        # حلقه روش نیوتن برای حل دستگاه غیرخطی
        for iteration in range(15):
            F_val_vec = np.array(system_func(W_num))
            J_val_mat = np.array(J_func(W_num))
            delta = np.linalg.solve(J_val_mat, -F_val_vec)
            W_num += delta

            if np.linalg.norm(delta) < 1e-14:
                break

        self.W_num = W_num
        self.y_sym = y_sym

    def evaluate(self, x_vals):
        y_num_func = sp.lambdify(self.x, self.y_sym.subs(dict(zip(self.W, self.W_num))), 'numpy')
        return y_num_func(x_vals)

# ==========================================
# اجرای الگوریتم، رسم نمودارها و چاپ جداول
# ==========================================
if __name__ == "__main__":
    N_degree = 3  # درجه تنظیم شده مطابق کد اول شما
    solver = ChelyshkovCollocationSolver(N=N_degree, M=30, alpha=0.5, nu=0.5)
    solver.solve()

    # نقاط ارزیابی مطابق با تنظیمات eval_points در کد اول شما
    table_points = np.linspace(0.1, 1.0, 10)

    # محاسبه مقادیر و خطاها
    approx_vals = solver.evaluate(table_points)
    exact_vals = np.sqrt(table_points)  # جواب دقیق y(x) = x^(0.5)
    errors = np.abs(exact_vals - approx_vals)

    print("=" * 72)
    print(f"{'x':<8} | {'Exact y(x)':<20} | {'Approx y_N(x)':<20} | {'Absolute Error':<15}")
    print("=" * 72)
    for i in range(len(table_points)):
        print(f"{table_points[i]:<8.2f} | {exact_vals[i]:<20.10e} | {approx_vals[i]:<20.10e} | {errors[i]:<15.4e}")
    print("=" * 72)

    # کد جدول LaTeX 
    print("\n% LaTeX Table Code:")
    print(r"\begin{table}[ht]")
    print(r"    \centering")
    print(r"    \caption{Comparison of Exact and Approximate Solutions for Example 1 with $N=" + str(N_degree) + r"$}")
    print(r"    \vspace{0.2cm}")
    print(r"    \begin{tabular}{cccc}")
    print(r"        \hline")
    print(r"        $x$ & Exact $y(x)$ & Approx $y(x)$ & Absolute Error \\ \hline")
    for i in range(len(table_points)):
        print(f"        {table_points[i]:.2f} & {exact_vals[i]:.10e} & {approx_vals[i]:.10e} & {errors[i]:.4e} \\\\")
    print(r"        \hline")
    print(r"    \end{tabular}")
    print(r"\end{table}" + "\n")

    # رسم نمودارها
    x_test = np.linspace(0, 1, 100)
    y_approx = solver.evaluate(x_test)
    y_exact = np.sqrt(x_test)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_test, y_exact, 'b-', linewidth=2, label=r'Exact: $y(x) = \sqrt{x}$')
    plt.plot(x_test, y_approx, 'r--', linewidth=2, label=f'App. Sol (N={N_degree})')
    plt.xlabel(r'$x$', fontsize=12)
    plt.ylabel(r'$y(x)$', fontsize=12)
    plt.title('Comparison of Exact and Approximate Solution')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.subplot(1, 2, 2)
    absolute_error = np.abs(y_exact - y_approx)
    plt.plot(x_test, absolute_error, 'k-', linewidth=1.5, label='Absolute Error')
    plt.yscale('log')
    plt.xlabel(r'$x$', fontsize=12)
    plt.ylabel(r'Error $||E_N||$', fontsize=12)
    plt.title('Logarithmic Absolute Error')
    plt.legend()
    plt.grid(True, which="both", linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()