import numpy as np
import matplotlib.pyplot as plt

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
D, x = cheb(N)                   # x در [-1,1]   (x[0]=1 ... x[N]=-1)
eta = (1 - x) / 2                # نگاشت به [0,1]:  eta=0 <-> x=1 , eta=1 <-> x=-1
D_eta = -2 * D                   # d/deta = (dx/deta)*d/dx = -2*D

_, wcc = clencurt(N)
w_eta = wcc / 2                  # وزن‌های انتگرال‌گیری روی [0,1]

# معادله (با علامت اصلاح‌شده که با جواب دقیق سازگار است):
#   F'(eta) = -exp(-eta) + exp(-1) - 1 + C ,   C = ∫_0^1 F(tau) d tau
rhs = -np.exp(-eta) + np.exp(-1) - 1.0

A = D_eta - np.outer(np.ones(N + 1), w_eta)   # ماتریس ضرایب سیستم خطی
b = rhs.copy()

# اعمال شرط مرزی F(0)=1 (اندیس eta=0 برابر با اندیس 0 در آرایه است)
A[0, :] = 0.0
A[0, 0] = 1.0
b[0] = 1.0

F_nodes = np.linalg.solve(A, b)

# ---------- ارزیابی در نقاط 0 تا 1 با گام 0.1 ----------
eta_eval = np.round(np.arange(0, 1.0001, 0.1), 2)
F_num = bary_interp(eta, F_nodes, eta_eval)
F_exact = np.exp(-eta_eval)
err = np.abs(F_num - F_exact)

print(f"{'eta':>6} {'F_numeric':>16} {'F_exact':>16} {'|error|':>14}")
for e, fn, fe, er in zip(eta_eval, F_num, F_exact, err):
    print(f"{e:6.2f} {fn:16.12f} {fe:16.12f} {er:14.3e}")

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

axes[1].semilogy(eta_eval, err + 1e-18, 'ks-', lw=1.5, ms=6)
axes[1].set_xlabel(r'$\eta$')
axes[1].set_ylabel('خطای مطلق  |F_num - F_exact|')
axes[1].set_title('خطای روش طیفی در نقاط $\\eta$')
axes[1].grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/spectral_solution.png', dpi=150)
print("\nplot saved.")
