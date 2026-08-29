import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

# =example 4=========================================
# 1. ساخت فضای RKHS و زنجیره استخراج ریاضی با SymPy
# ==========================================
n = 11
# تنظیم گره‌ها از 0 تا 1 با گام 0.1 مطابق درخواست
x_nodes = np.linspace(0, 1, n)

# تعریف متغیرهای نمادین ریاضی برای استخراج خودکار فرمول‌ها
x_sym, y_sym, t_sym, s = sp.symbols('x y t s', real=True)

# ---> تعریف کرنل بازتولیدکننده W_2^2 (مرتبه اول)
# این کرنل به طور ذاتی پس از اعمال عملگر مشتق، شرط F(0)=0 را ارضا می‌کند
K_1 = 1 + x_sym * y_sym + 0.5 * x_sym * y_sym**2 - sp.Rational(1, 6) * y_sym**3
K_2 = 1 + x_sym * y_sym + 0.5 * x_sym**2 * y_sym - sp.Rational(1, 6) * x_sym**3

K_sym = sp.Piecewise(
    (K_1, x_sym >= y_sym),
    (K_2, x_sym < y_sym)
)

def get_phi_i(xi):
    """ \phi_i(x) = K(x, xi) """
    return K_sym.subs(y_sym, xi)

def apply_L_star(expr, var):
    """
    چون عملگر اصلی مسئله L(F) = F' است،
    عملگر الحاقی آن معادل 1 بار مشتق‌گیری نسبت به متغیر است.
    """
    return sp.diff(expr, var, 1)

# استخراج خودکار w_i از روی کرنل و L*
w_sym = apply_L_star(K_sym, y_sym)  # w_i(x) = L_y K(x, y)

# محاسبه ضرب داخلی مستقیماً از تعریف فضای RKHS
inner_prod_sym = apply_L_star(w_sym, x_sym) 

def inner_product_w(i, j):
    """ محاسبه ضرب داخلی با ارزیابی فرمول استخراج‌شده در نقاط """
    xi, xj = x_nodes[i], x_nodes[j]
    val = inner_prod_sym.subs({x_sym: xi, y_sym: xj}).doit()
    return float(val)

def int_t_w(m):
    """ \int_0^1 w_m(\tau) d\tau """
    xm = x_nodes[m]
    # برای پایداری در محاسبه تحلیلی، بازه انتگرال دو تکه می‌شود
    w_1 = sp.diff(K_1, y_sym, 1).subs(y_sym, xm) 
    w_2 = sp.diff(K_2, y_sym, 1).subs(y_sym, xm) 
    
    w_1_t = w_1.subs(x_sym, t_sym)
    w_2_t = w_2.subs(x_sym, t_sym)
    
    # انتگرال‌گیری روی تابع F(\tau) (که در اینجا تقریب آن w_m است)
    int1 = sp.integrate(w_2_t, (t_sym, 0, xm)).doit()
    int2 = sp.integrate(w_1_t, (t_sym, xm, 1)).doit()
    
    return float(int1 + int2)

# تبدیل توابع نمادین به توابع عددی (NumPy)
w_num_func = sp.lambdify((x_sym, y_sym), w_sym, modules=['numpy'])
w_deriv_sym = sp.diff(w_sym, x_sym, 1) # مشتق اول برای باقی‌مانده معادله
w_deriv_num_func = sp.lambdify((x_sym, y_sym), w_deriv_sym, modules=['numpy'])

def w_i_val(xi, X):
    X_arr = np.asarray(X, dtype=float)
    return w_num_func(X_arr, xi)

def w_i_deriv(xi, X):
    X_arr = np.asarray(X, dtype=float)
    return w_deriv_num_func(X_arr, xi)

# ==========================================
# 2. متعامدسازی گرام-اشمیت (Gram-Schmidt)
# ==========================================
beta = np.zeros((n, n))
c_matrix = np.zeros((n, n))

for i in range(n):
    for k in range(i):
        c_matrix[i, k] = sum(beta[k, m] * inner_product_w(i, m) for m in range(k + 1))
    
    norm_w_i_sq = inner_product_w(i, i)
    sum_c_sq = sum(c_matrix[i, k]**2 for k in range(i))
    d_i = np.sqrt(norm_w_i_sq - sum_c_sq)
    
    beta[i, i] = 1.0 / d_i
    
    for j in range(i):
        sum_c_beta = sum(c_matrix[i, k] * beta[k, j] for k in range(j, i))
        beta[i, j] = - (1.0 / d_i) * sum_c_beta

# ==========================================
# 3. ساخت و حل دستگاه معادلات خطی 
# ==========================================
I_wbar = np.zeros(n)
for j in range(n):
    I_wbar[j] = sum(beta[j, m] * int_t_w(m) for m in range(j + 1))

M = np.zeros((n, n))
RHS = np.zeros(n)

for i in range(n):
    # f(\eta) = \eta * e^\eta + e^\eta - \eta
    B_f_i = sum(beta[i, k] * (x_nodes[k] * np.exp(x_nodes[k]) + np.exp(x_nodes[k]) - x_nodes[k]) for k in range(i + 1))
    # ضریب بیرون انتگرال در معادله \eta است (با انتقال به سمت چپ منفی می‌شود که در فرمول M ضرب شده)
    B_x_i = sum(beta[i, k] * (x_nodes[k]) for k in range(i + 1))
    
    RHS[i] = B_f_i
    for j in range(n):
        M[i, j] = - B_x_i * I_wbar[j]
        if i == j:
            M[i, j] += 1.0

B_coeffs = np.linalg.solve(M, RHS)

A_coeffs = np.zeros(n)
for m in range(n):
    A_coeffs[m] = sum(B_coeffs[j] * beta[j, m] for j in range(m, n))

# ==========================================
# 4. توابع بازسازی جواب و مشتق‌گیری
# ==========================================
def un_approx(X):
    return sum(A_coeffs[m] * w_i_val(x_nodes[m], X) for m in range(n))

def un_deriv_approx(X):
    return sum(A_coeffs[m] * w_i_deriv(x_nodes[m], X) for m in range(n))

# ==========================================
# 5. ارزیابی خطا و باقی‌مانده برای نمودارها
# ==========================================
x_plot = np.linspace(0, 1, 200)

# جواب دقیق F(\eta) = \eta * e^\eta
exact_plot = x_plot * np.exp(x_plot)
approx_plot = un_approx(x_plot)
error_plot = np.abs(exact_plot - approx_plot)

int_tu_n = sum(A_coeffs[m] * int_t_w(m) for m in range(n))
# محاسبه بخش راست معادله برای استخراج باقی‌مانده
RHS_eq = x_plot * np.exp(x_plot) + np.exp(x_plot) - x_plot + x_plot * int_tu_n
LHS_eq = un_deriv_approx(x_plot)
residual_plot = np.abs(LHS_eq - RHS_eq)

# ==========================================
# 6. رسم نمودارها
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(x_plot, error_plot, color='red', linewidth=2, label='Absolute Error $|F(\eta) - F_n(\eta)|$')
ax1.set_title('Absolute Error over [0, 1]')
ax1.set_xlabel('$\eta$')
ax1.set_ylabel('Error')
ax1.set_yscale('log')
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()

ax2.plot(x_plot, residual_plot, color='blue', linewidth=2, label='Residual $R(\eta)$')
ax2.set_title('Equation Residual over [0, 1]')
ax2.set_xlabel('$\eta$')
ax2.set_ylabel('Residual')
ax2.set_yscale('log')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 7. نمایش جدول نهایی 
# ==========================================
# تولید دقیق نقاط بررسی از 0 تا 1 با گام 0.1
test_points = np.linspace(0, 1, 11) 
exact_vals = test_points * np.exp(test_points)
approx_vals = un_approx(test_points)
errors = np.abs(exact_vals - approx_vals)

df_results = pd.DataFrame({
    'x': test_points,
    'Exact Solution': exact_vals,
    'RKHS Solution': approx_vals,
    'Absolute Error': errors
})

df_results['x'] = df_results['x'].apply(lambda val: f"{val:.2f}")
df_results['Exact Solution'] = df_results['Exact Solution'].apply(lambda val: f"{val:.20f}")
df_results['RKHS Solution'] = df_results['RKHS Solution'].apply(lambda val: f"{val:.20f}")
df_results['Absolute Error'] = df_results['Absolute Error'].apply(lambda val: f"{val:.2e}")

print("\n--- Numerical Results (Strict RKHS Theory Extraction) ---")
print(df_results.to_string(index=False))