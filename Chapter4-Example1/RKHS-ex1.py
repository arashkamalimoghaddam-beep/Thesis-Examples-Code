import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

# ==========================================
# 1. ساخت فضای RKHS و زنجیره استخراج ریاضی با SymPy
# ==========================================
n = 11
# با توجه به بازه انتگرال در تصویر، دامنه [0, 1] است
x_nodes = np.linspace(0, 1, n)

# تعریف متغیرهای نمادین
x_sym, y_sym = sp.symbols('x y', real=True)

# ---> تعریف کرنل بازتولیدکننده W_2^2 روی بازه [0,1]
# سازگار با شرط اولیه u(0) = 0 (حاصل از تغییر متغیر F(eta) - 1)
K_1 = x_sym * y_sym + (x_sym**2 * y_sym) / 2 - (x_sym**3) / 6
K_2 = x_sym * y_sym + (x_sym * y_sym**2) / 2 - (y_sym**3) / 6

K_sym = sp.Piecewise(
    (K_1, x_sym <= y_sym),
    (K_2, x_sym > y_sym)
)

def apply_L(expr, var):
    """
    عملگر اصلی مسئله L(u) = u' + u است.
    """
    return sp.diff(expr, var, 1) + expr

# استخراج خودکار w_i(x) از روی کرنل
# w_i(x) = L_y K(x, y)
w_1 = apply_L(K_1, y_sym)
w_2 = apply_L(K_2, y_sym)

# محاسبه ضرب داخلی \langle w_i, w_j \rangle = L_x L_y K(x, y)
inner_11 = apply_L(w_1, x_sym)
inner_22 = apply_L(w_2, x_sym)

def inner_product_w(i, j):
    """ محاسبه ضرب داخلی با جایگذاری گره‌ها """
    xi, xj = x_nodes[i], x_nodes[j]
    if xi <= xj:
        val = inner_11.subs({x_sym: xi, y_sym: xj}).doit()
    else:
        val = inner_22.subs({x_sym: xi, y_sym: xj}).doit()
    return float(val)

# تبدیل توابع نمادین به توابع عددی سریع (NumPy)
w_1_num = sp.lambdify((x_sym, y_sym), w_1, modules=['numpy'])
w_2_num = sp.lambdify((x_sym, y_sym), w_2, modules=['numpy'])

w_1_deriv = sp.diff(w_1, x_sym, 1)
w_2_deriv = sp.diff(w_2, x_sym, 1)
w_1_deriv_num = sp.lambdify((x_sym, y_sym), w_1_deriv, modules=['numpy'])
w_2_deriv_num = sp.lambdify((x_sym, y_sym), w_2_deriv, modules=['numpy'])

def w_i_val(xi, X):
    X_arr = np.asarray(X, dtype=float)
    is_scalar = X_arr.ndim == 0
    if is_scalar: X_arr = np.array([X_arr])
    # ضرب در np.ones_like برای جلوگیری از خطای اسکالر شدن در lambdify
    res = np.where(X_arr <= xi,
                   w_1_num(X_arr, xi) * np.ones_like(X_arr),
                   w_2_num(X_arr, xi) * np.ones_like(X_arr))
    return res[0] if is_scalar else res

def w_i_deriv(xi, X):
    X_arr = np.asarray(X, dtype=float)
    is_scalar = X_arr.ndim == 0
    if is_scalar: X_arr = np.array([X_arr])
    res = np.where(X_arr <= xi,
                   w_1_deriv_num(X_arr, xi) * np.ones_like(X_arr),
                   w_2_deriv_num(X_arr, xi) * np.ones_like(X_arr))
    return res[0] if is_scalar else res

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
# 3. حل ساختار غیرخطی با رهیافت جبری دقیق
# ==========================================
# از آنجا که طرف راست معادله پس از انتگرال‌گیری معین، یک "عدد ثابت" است،
# ابتدا مسئله را به ازای ثابت واحد (L(u) = 1) حل می‌کنیم.
RHS_unit = np.ones(n)
B_unit = np.zeros(n)

for i in range(n):
    B_unit[i] = sum(beta[i, k] * RHS_unit[k] for k in range(i + 1))

A_unit = np.zeros(n)
for m in range(n):
    A_unit[m] = sum(B_unit[j] * beta[j, m] for j in range(m, n))

def u_unit_approx(X):
    return sum(A_unit[m] * w_i_val(x_nodes[m], X) for m in range(n))

# محاسبه پارامترهای انتگرالی جواب پایه
tau_vals = np.linspace(0, 1, 500)
u_unit_vals = np.array([u_unit_approx(t) for t in tau_vals])

I_2 = np.trapz(u_unit_vals**2, tau_vals)
I_1 = np.trapz(u_unit_vals, tau_vals)

# حال معادله تبدیل به حل معادله درجه دوم زیر برای یافتن ثابت C* می‌شود:
# (I_2) * C^2 + (2 * I_1 - 1) * C + 0.5 * (e^{-2} - 1) = 0
a_quad = I_2
b_quad = 2 * I_1 - 1
c_quad = 0.5 * (np.exp(-2) - 1)

delta = b_quad**2 - 4 * a_quad * c_quad
C_star_1 = (-b_quad + np.sqrt(delta)) / (2 * a_quad)
C_star_2 = (-b_quad - np.sqrt(delta)) / (2 * a_quad)

# ریشه‌ای را انتخاب می‌کنیم که از نظر فیزیک مسئله صحیح است (نزدیک به -1)
C_star = C_star_1 if abs(C_star_1 - (-1)) < abs(C_star_2 - (-1)) else C_star_2

# ضرایب نهایی مدل ساخته می‌شود
A_coeffs = C_star * A_unit

# ==========================================
# 4. توابع بازسازی جواب نهایی
# ==========================================
def un_approx(X):
    """ u(\eta) """
    return sum(A_coeffs[m] * w_i_val(x_nodes[m], X) for m in range(n))

def un_deriv_approx(X):
    """ u'(\eta) """
    return sum(A_coeffs[m] * w_i_deriv(x_nodes[m], X) for m in range(n))

def F_approx(X):
    """ F(\eta) = u(\eta) + 1 """
    return un_approx(X) + 1.0

def F_deriv_approx(X):
    """ F'(\eta) = u'(\eta) """
    return un_deriv_approx(X)

# ==========================================
# 5. ارزیابی خطا و باقی‌مانده معادله
# ==========================================
x_plot = np.linspace(0, 1, 200)

exact_plot = np.exp(-x_plot)
approx_plot = F_approx(x_plot)
error_plot = np.abs(exact_plot - approx_plot)

# محاسبه باقی‌مانده معادله دیفرانسیل:
# R = | F' + F - (0.5*(e^-2 - 1) + \int F^2) |
LHS_eq = F_deriv_approx(x_plot) + F_approx(x_plot)
int_F_sq = np.trapz(F_approx(tau_vals)**2, tau_vals) # انتگرال عددی F^2
RHS_eq = 0.5 * (np.exp(-2) - 1) + int_F_sq

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
#example 1
plt.tight_layout()
plt.show()

# ==========================================
# 7. نمایش جدول نهایی
# ==========================================
test_points =  np.array([0.0, 0.16, 0.32, 0.48, 0.64, 0.80, 0.96, 1.0])
exact_vals = np.exp(-test_points)
approx_vals = F_approx(test_points)
errors = np.abs(exact_vals - approx_vals)

df_results = pd.DataFrame({
    'eta': test_points,
    'Exact F(eta)': exact_vals,
    'RKHS F(eta)': approx_vals,
    'Absolute Error': errors
})

df_results['eta'] = df_results['eta'].apply(lambda val: f"{val:.2f}")
df_results['Exact F(eta)'] = df_results['Exact F(eta)'].apply(lambda val: f"{val:.15f}")
df_results['RKHS F(eta)'] = df_results['RKHS F(eta)'].apply(lambda val: f"{val:.15f}")
df_results['Absolute Error'] = df_results['Absolute Error'].apply(lambda val: f"{val:.2e}")

print("\n--- Numerical Results for Non-Linear IDE (RKHS + Algebraic Fix) ---")
print(df_results.to_string(index=False))
