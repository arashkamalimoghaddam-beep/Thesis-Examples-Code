import numpy as np
import matplotlib.pyplot as plt
import decimal

# تنظیم دقت محاسبات برای استخراج دقیق و واقعی خطای عددی
decimal.getcontext().prec = 50

# ---------- توابع پایه روش طیفی (چبیشف) ----------
def cheb(N):
    """ماتریس مشتق‌گیری چبیشف روی نقاط گاوس-لوباتو x در بازه [-1,1]"""
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2., np.ones(N - 1), 2.]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1. / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x

def clencurt(N):
    """وزن‌های انتگرال‌گیری کلنشا-کرتیس روی همان نقاط چبیشف در [-1,1]"""
    theta = np.pi * np.arange(N + 1) / N
    x = np.cos(theta)
    w = np.zeros(N + 1)
    ii = np.arange(1, N)
    v = np.ones(N - 1)
    if N % 2 == 0:
        w[0] = 1.0 / (N ** 2 - 1)
        w[N] = w[0]
        for k in range(1, N // 2):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k ** 2 - 1)
        v -= np.cos(N * theta[ii]) / (N ** 2 - 1)
    else:
        w[0] = 1.0 / N ** 2
        w[N] = w[0]
        for k in range(1, (N - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k ** 2 - 1)
    w[ii] = 2 * v / N
    return x, w

def bary_interp(x_nodes, f_nodes, x_eval):
    """درون‌یابی بری‌سنتریک لاگرانژ (مناسب برای نقاط چبیشف)"""
    N = len(x_nodes) - 1
    w = np.hstack([0.5, np.ones(N - 1), 0.5]) * (-1) ** np.arange(N + 1)
    x_eval = np.atleast_1d(x_eval).astype(float)
    f_eval = np.zeros_like(x_eval)
    for j, xe in enumerate(x_eval):
        diff = xe - x_nodes
        hit = np.where(np.abs(diff) < 1e-13)[0]
        if hit.size > 0:
            f_eval[j] = f_nodes[hit[0]]
        else:
            num = np.sum(w * f_nodes / diff)
            den = np.sum(w / diff)
            f_eval[j] = num / den
    return f_eval

# ---------- تنظیم مسئله ----------
N = 20                          # تعداد نقاط طیفی -> دقت نزدیک به ماشین
D, x = cheb(N)                  # x در [-1,1]
eta = (1 - x) / 2               # نگاشت به [0,1]
D_eta = -2 * D                  # d/deta = -2*D

_, wcc = clencurt(N)
w_eta = wcc / 2                 # وزن‌های انتگرال‌گیری روی [0,1]

rhs = eta**2 * (2*np.exp(-1) - 1) - np.exp(-eta)

w2 = w_eta * eta
M = np.outer(eta**2, w2)        # ماتریس جمله انتگرالی

A = D_eta - M                    # ماتریس ضرایب
b = rhs.copy()

# اعمال شرط مرزی F(0)=1
A[0, :] = 0.0
A[0, 0] = 1.0
b[0] = 1.0

F_nodes = np.linalg.solve(A, b)

# ---------- ارزیابی در نقاط 0 تا 1 با دقت مضاعف برای محاسبه خطای واقعی ----------
eta_eval = np.array([0.0, 0.16, 0.32, 0.48, 0.64, 0.80, 0.96, 1.0])
F_num = bary_interp(eta, F_nodes, eta_eval)

print(f"{'eta':>6} {'F_numeric':>16} {'F_exact':>16} {'|error|':>14}")

err_exact_list = []
for e, fn in zip(eta_eval, F_num):
    # 1. محاسبه مقدار دقیق ریاضی (e^-eta) با دقت ۵۰ رقم برای جلوگیری از خطای ماشین
    exact_dec = (-decimal.Decimal(float(e))).exp()
    
    # 2. استخراج مقدار دقیق عددیِ ذخیره شده در حافظه بدون گرد کردن
    num_dec = decimal.Decimal(float(fn))
    
    # 3. محاسبه تفاوت دقیق بدون صفر شدن مصنوعی
    err_dec = abs(num_dec - exact_dec)
    err_float = float(err_dec)
    err_exact_list.append(err_float)
    
    fe = float(exact_dec)
    print(f"{e:6.2f} {fn:16.12f} {fe:16.12f} {err_float:14.3e}")

err = np.array(err_exact_list)

# ---------- رسم نمودارها ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

eta_fine = np.linspace(0, 1, 300)
F_fine = bary_interp(eta, F_nodes, eta_fine)

axes[0].plot(eta_fine, np.exp(-eta_fine), 'b-', lw=2, label=r'جواب دقیق $F(\eta)=e^{-\eta}$')
axes[0].plot(eta_eval, F_num, 'ro', ms=6, label='جواب طیفی (نقاط 0.1)')
axes[0].set_xlabel(r'$\eta$')
axes[0].set_ylabel(r'$F(\eta)$')
axes[0].set_title('مقایسه جواب طیفی و جواب دقیق')
axes[0].legend()
axes[0].grid(alpha=0.3)

# در نمودار خطا دیگر نیازی به وارد کردن مقادیر غیرواقعی مثل 1e-18 نیست
axes[1].semilogy(eta_eval, err, 'ks-', lw=1.5, ms=6)
axes[1].set_xlabel(r'$\eta$')
axes[1].set_ylabel('خطای مطلق  |F_num - F_exact|')
axes[1].set_title('خطای روش طیفی در نقاط $\\eta$')
axes[1].grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/spectral_solution3.png', dpi=150)
print("\nplot saved.")