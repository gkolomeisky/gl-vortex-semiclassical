"""
figure2_convergence.py
----------------------
Reproduces Figure 2 of:
  "Exact Single-Scale Outer Solution of the Abrikosov Vortex
   in the Extreme Type-II Limit"

Two-panel figure showing convergence of GL solutions to the
asymptotically exact Bessel function outer solution as kappa -> inf.

Upper panel: kappa*b vs K_0(r)  [log-log]
Lower panel: kappa^2*(1-R^2) vs K_1^2(r)  [log-log]

for kappa = 5, 10, 20, 40, units lambda=1.

Dependencies: numpy, scipy, matplotlib
Usage: python figure2_convergence.py
Output: vortex_convergence_final.pdf
"""

import numpy as np
from scipy.special import k1 as K1, k0 as K0
from scipy.integrate import solve_bvp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'mathtext.default': 'bf',
})

r_max  = 20.0
kappas = [5, 10, 20, 40]
colors = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7']

# ---------- GL BVP solver ----------
def solve_gl(kappa):
    def gl_RW(s, y):
        R, Rs, W, Ws = y
        e2s = np.exp(2*s)
        return np.array([Rs,
                         kappa**2*R*(W**2 + e2s*(R**2 - 1)),
                         Ws,
                         2*Ws + e2s*W*R**2])

    def bc_GL(ya, yb):
        sqrt2k = np.sqrt(2)*kappa
        return np.array([ya[1] - ya[0],
                         ya[2] - 1.0/kappa,
                         yb[0] - (1.0 - K0(sqrt2k*r_max)),
                         yb[2] - (1.0/kappa)*(1.0 + r_max*K1(r_max))])

    s_g = np.linspace(np.log(5e-4), np.log(r_max), 1500)
    r_g = np.exp(s_g)
    W_g = np.clip((1.0/kappa)*(1.0 + r_g*K1(np.maximum(r_g, 1e-6))),
                  0, 1.0/kappa + 0.005)
    R_g = np.tanh(kappa*r_g/np.sqrt(2))
    sol = solve_bvp(gl_RW, bc_GL, s_g,
                    np.array([R_g, np.gradient(R_g, s_g),
                               W_g, np.gradient(W_g, s_g)]),
                    tol=1e-9, max_nodes=20000, verbose=0)
    assert sol.success, f"GL BVP did not converge for kappa={kappa}"
    return sol

print("Solving GL equations for kappa =", kappas, "...")
solutions = {k: solve_gl(k) for k in kappas}
print("Done.")

# ---------- Evaluate curves ----------
r_pts    = np.logspace(np.log10(1e-3), np.log10(10.0), 5000)
K0_ref   = K0(r_pts)
K1sq_ref = K1(r_pts)**2

# ---------- Figure ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 12), sharex=True)
fig.subplots_adjust(hspace=0.06)

# Bessel reference curves
ax1.loglog(r_pts, K0_ref,   'k-', lw=3.5, label='$K_0(r)$',   zorder=5)
ax2.loglog(r_pts, K1sq_ref, 'k-', lw=3.5, label='$K_1^2(r)$', zorder=5)

for kappa, col in zip(kappas, colors):
    sol      = solutions[kappa]
    kb_arr   = []
    k2R2_arr = []
    for r in r_pts:
        s  = np.log(max(r, 5e-4))
        yg = sol.sol(s)
        R  = max(float(yg[0]), 0.0)
        Ws = float(yg[3])
        b  = -Ws/r**2
        kb_arr.append(kappa*b)
        k2R2_arr.append(kappa**2*max(1.0 - R**2, 0.0))

    kb_arr   = np.array(kb_arr)
    k2R2_arr = np.array(k2R2_arr)
    mask_b   = kb_arr > 0

    lbl = f'$\\kappa={kappa}$'
    ax1.loglog(r_pts[mask_b], kb_arr[mask_b], color=col, lw=2.5,
               label=lbl, zorder=4)
    ax2.loglog(r_pts,         k2R2_arr,        color=col, lw=2.5,
               label=lbl, zorder=4)

    # Core boundary vertical lines
    rc = 1.0/kappa
    ax1.axvline(rc, color=col, lw=1.4, ls=':', zorder=3)
    ax2.axvline(rc, color=col, lw=1.4, ls=':', zorder=3)

# lambda marker
for ax in (ax1, ax2):
    ax.axvline(1.0, color='gray', lw=1.3, ls='--', alpha=0.7, zorder=2)

# Core boundary and lambda labels — top of upper panel only
top_y = 1.002
for kappa, col in zip(kappas, colors):
    rc = 1.0/kappa
    ax1.text(rc, top_y, f'$\\frac{{1}}{{{kappa}}}$',
             ha='center', va='bottom', fontsize=16, color=col,
             fontweight='bold', transform=ax1.get_xaxis_transform())
ax1.text(1.0, top_y, '$\\lambda$', ha='center', va='bottom',
         fontsize=17, color='gray', fontweight='bold',
         transform=ax1.get_xaxis_transform())

# Axes
ax1.set_xlim(1e-3, 10.0);  ax1.set_ylim(1e-4, 10)
ax1.set_ylabel(r'$\kappa\, b$', fontsize=20, fontweight='bold')
ax1.tick_params(labelsize=15, width=1.5)
ax1.grid(True, which='both', alpha=0.2, lw=0.5)
for spine in ax1.spines.values(): spine.set_linewidth(1.5)

ax2.set_xlim(1e-3, 10.0);  ax2.set_ylim(1e-4, 1e4)
ax2.set_ylabel(r'$\kappa^2(1-R^2)$', fontsize=20, fontweight='bold')
ax2.set_xlabel('$r$', fontsize=20, fontweight='bold')
ax2.tick_params(labelsize=15, width=1.5)
ax2.grid(True, which='both', alpha=0.2, lw=0.5)
for spine in ax2.spines.values(): spine.set_linewidth(1.5)

xt = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
xt_labels = {0.001:'0.001', 0.005:'0.005', 0.01:'0.01',
             0.05:'0.05',   0.1:'0.1',     0.5:'0.5',
             1:'1',         2:'2',          5:'5',    10:'10'}
ax2.set_xticks(xt)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, p: xt_labels.get(round(x, 3), '')))

for ax in (ax1, ax2):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

ax1.legend(fontsize=14, loc='lower left', framealpha=0.9,
           facecolor='#F5F5F0', edgecolor='#CCCCCC',
           prop={'weight':'bold', 'size':14})
ax2.legend(fontsize=14, loc='lower left', framealpha=0.9,
           facecolor='#F5F5F0', edgecolor='#CCCCCC',
           prop={'weight':'bold', 'size':14})

fig.savefig('vortex_convergence_final.pdf',
            format='pdf', bbox_inches='tight', facecolor='white')
print("Figure 2 saved: vortex_convergence_final.pdf")
