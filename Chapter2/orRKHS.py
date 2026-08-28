import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

# ==========================================
# 1. ساخت فضای RKHS و زنجیره استخراج ریاضی با SymPy
# (پاسخ به تمامی ۶ ایراد مطرح شده در تصاویر)
# ==========================================
n = 11
x_nodes = np.linspace(0, 1, n)

# تعریف متغیرهای نمادین ریاضی برای استخراج خودکار فرمول‌ها
x_sym, y_sym, t_sym = sp.symbols('x y t', real=True)

# ---> پاسخ به ایراد 1: تعریف کرنل بازتولیدکننده به عنوان نقطه شروع واقعی
K_sym = sp.Piecewise(
    (1 + x_sym * y_sym + 0.5 * x_sym * y_sym**2 - sp.Rational(1, 6) * y_sym**3, x_sym >= y_sym),
    (1 + x_sym * y_sym + 0.5 * x_sym**2 * y_sym - sp.Rational(1, 6) * x_sym**3, x_sym < y_sym)
)

# ---> پاسخ به ایراد 5: پیاده‌سازی دقیق تابع \phi_i
def get_phi_i(xi):
    """ \phi_i(x) = K(x, xi) """
    return K_sym.subs(y_sym, xi)

# ---> پاسخ به ایراد 6: تعریف عملگر الحاقی L* به عنوان یک تابع عملگر واقعی
def apply_L_star(expr, var):
    """
    چون عملگر اصلی مسئله L(u) = u' است،
    عملگر الحاقی آن معادل مشتق‌گیری نسبت به متغیر است.
    """
    return sp.diff(expr, var)

# ---> پاسخ به ایراد 2: استخراج خودکار w_i از روی کرنل و L*
w_sym = apply_L_star(K_sym, y_sym)  # w_i(x) = L_y K(x, y)

# ---> پاسخ به ایراد 3: محاسبه ضرب داخلی مستقیماً از تعریف فضای RKHS
# <w_i, w_j> = L_x L_y K(x,y)
inner_prod_sym = sp.diff(w_sym, x_sym) 

def inner_product_w(i, j):
    """ محاسبه ضرب داخلی با ارزیابی فرمول استخراج‌شده در نقاط بدون فرمول دستی """
    xi, xj = x_nodes[i], x_nodes[j]
    val = inner_prod_sym.subs({x_sym: xi, y_sym: xj})
    return float(val)

# ---> پاسخ به ایراد 4: محاسبه انتگرال با عمل انتگرال‌گیری واقعی 
def int_t_w(m):
    """ \int_0^1 t * w_m(t) dt """
    xm = x_nodes[m]
    # جایگذاری y = x_m برای به دست آوردن تابع w_m(x)
    w_m_expr = w_sym.subs(y_sym, xm)
    # ساخت تابع داخل انتگرال (t * w_m(t))
    integrand = t_sym * w_m_expr.subs(x_sym, t_sym)
    # محاسبه تحلیلی انتگرال توسط سیمپای
    integral_val = sp.integrate(integrand, (t_sym, 0, 1))
    return float(integral_val)

# تبدیل توابع نمادین به توابع عددی (NumPy) برای سرعت بالا در محاسبات و رسم نمودار
w_num_func = sp.lambdify((x_sym, y_sym), w_sym, modules=['numpy'])
w_deriv_sym = sp.diff(w_sym, x_sym)
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
    B_f_i = sum(beta[i, k] * (1.0 - x_nodes[k] / 3.0) for k in range(i + 1))
    B_x_i = sum(beta[i, k] * x_nodes[k] for k in range(i + 1))
    
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

exact_plot = x_plot
approx_plot = un_approx(x_plot)
error_plot = np.abs(exact_plot - approx_plot)

int_tu_n = sum(A_coeffs[m] * int_t_w(m) for m in range(n))
RHS_eq = 1.0 - (x_plot / 3.0) + x_plot * int_tu_n
LHS_eq = un_deriv_approx(x_plot)
residual_plot = np.abs(LHS_eq - RHS_eq)

# ==========================================
# 6. رسم نمودارها
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(x_plot, error_plot, color='red', linewidth=2, label='Absolute Error $|u(x) - u_n(x)|$')
ax1.set_title('Absolute Error over [0, 1]')
ax1.set_xlabel('$x$')
ax1.set_ylabel('Error')
ax1.set_yscale('log')
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()

ax2.plot(x_plot, residual_plot, color='blue', linewidth=2, label='Residual $R(x)$')
ax2.set_title('Equation Residual over [0, 1]')
ax2.set_xlabel('$x$')
ax2.set_ylabel('Residual')
ax2.set_yscale('log')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 7. نمایش جدول نهایی 
# ==========================================
test_points = np.array([0.16, 0.32, 0.48, 0.64, 0.80, 0.96])
exact_vals = test_points
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
