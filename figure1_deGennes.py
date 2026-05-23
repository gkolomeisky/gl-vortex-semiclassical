"""
figure1_deGennes.py
-------------------
Reproduces Figure 1 of:
  "Exact Single-Scale Outer Solution of the Abrikosov Vortex
   in the Extreme Type-II Limit"

de Gennes-style log-r, linear-y plot of R^2 and b/b(0)
for the full GL solution at kappa=20, units lambda=1.

Dependencies: numpy, scipy, matplotlib
Usage: python figure1_deGennes.py
Output: gl_logr_linear_vortex_kappa20_final.pdf
"""

import numpy as np
from scipy.special import k1 as K1, k0 as K0
from scipy.integrate import solve_bvp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'mathtext.default': 'bf',
})

kappa = 20.0
xi    = 1.0/kappa
r_max = 20.0

# ---------- GL BVP ----------
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

s_g = np.linspace(np.log(5e-4), np.log(r_max), 1200)
r_g = np.exp(s_g)
W_g = np.clip((1.0/kappa)*(1.0 + r_g*K1(np.maximum(r_g, 1e-6))),
              0, 1.0/kappa + 0.005)
R_g = np.tanh(kappa*r_g/np.sqrt(2))

sol_gl = solve_bvp(gl_RW, bc_GL, s_g,
                   np.array([R_g, np.gradient(R_g, s_g),
                              W_g, np.gradient(W_g, s_g)]),
                   tol=1e-9, max_nodes=15000, verbose=0)
assert sol_gl.success, "GL BVP solver did not converge"

# b(0) normalization
yg0   = sol_gl.sol(np.log(5e-4))
b_max = -float(yg0[3])/(5e-4)**2

# ---------- Evaluate curves ----------
r_pts = np.logspace(np.log10(0.009), np.log10(5.4), 1200)
R2_arr, b_arr = [], []
for r in r_pts:
    yg = sol_gl.sol(np.log(max(r, 5e-4)))
    R2_arr.append(float(yg[0])**2)
    b_arr.append(-float(yg[3])/r**2 / b_max)
R2_arr = np.array(R2_arr)
b_arr  = np.array(b_arr)

# ---------- Figure ----------
blue  = '#378ADD'
amber = '#BA7517'

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.axvspan(0.008, xi, color='gray', alpha=0.12, hatch='////', linewidth=0)
ax.axvline(xi,  color='#555555', lw=1.2, ls=(0,(5,3)))
ax.axvline(1.0, color='#888888', lw=1.0, ls=(0,(5,3)))
ax.axhline(1.0, color='black',   lw=2.2, zorder=1)
for yval in [0.25, 0.50, 0.75]:
    ax.axhline(yval, color='#cccccc', lw=0.6, ls=(0,(2,5)), zorder=0)

ax.semilogx(r_pts, R2_arr, color=blue,  lw=2.8, zorder=3)
ax.semilogx(r_pts, b_arr,  color=amber, lw=2.8, zorder=3)

ax.set_xlabel('$r$', fontsize=18, fontweight='bold')
ax.set_xlim(0.008, 5.5)
ax.set_ylim(-0.02, 1.08)

ax.set_ylabel('$R^2$', fontsize=18, fontweight='bold', color=blue)
ax.tick_params(axis='y', labelcolor=blue, labelsize=14, width=1.5)
ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
ax.set_yticklabels(['0','0.25','0.50','0.75','1'],
                   color=blue, fontsize=14, fontweight='bold')

ax2 = ax.twinx()
ax2.set_ylim(-0.02, 1.08)
ax2.set_ylabel('$b/b(0)$', fontsize=18, fontweight='bold', color=amber)
ax2.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
ax2.set_yticklabels(['0','0.25','0.50','0.75','1'],
                    color=amber, fontsize=14, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=amber, labelsize=14, width=1.5)

ax.set_xticks([0.01,0.02,0.05,0.1,0.2,0.5,1,2,5])
ax.set_xticklabels(['0.01','0.02','0.05','0.1','0.2','0.5','1','2','5'],
                   fontsize=13, fontweight='bold')

ax.text(xi,  1.04, r'$\xi$',     ha='center', va='bottom', fontsize=16,
        fontweight='bold', transform=ax.get_xaxis_transform())
ax.text(1.0, 1.04, r'$\lambda$', ha='center', va='bottom', fontsize=16,
        fontweight='bold', transform=ax.get_xaxis_transform())
ax.text(0.022, 0.50, 'core', ha='center', va='center', fontsize=13,
        color='#666666', style='italic', fontweight='bold')

ax.annotate('$R^2$',
            xy=(0.28, float(R2_arr[np.argmin(np.abs(r_pts-0.28))])),
            xytext=(0.55, 0.78),
            arrowprops=dict(arrowstyle='->', color=blue, lw=1.8),
            fontsize=16, color=blue, fontweight='bold', ha='center')

r_arrow    = 0.8
b_at_arrow = float(b_arr[np.argmin(np.abs(r_pts-r_arrow))])
ax.annotate('$b/b(0)$',
            xy=(r_arrow, b_at_arrow),
            xytext=(1.8, 0.38),
            arrowprops=dict(arrowstyle='->', color=amber, lw=1.8),
            fontsize=16, color=amber, fontweight='bold', ha='center')

for spine in ax.spines.values():  spine.set_linewidth(1.5)
for spine in ax2.spines.values(): spine.set_linewidth(1.5)
ax.minorticks_off()
ax2.minorticks_off()
fig.tight_layout()
fig.savefig('gl_logr_linear_vortex_kappa20_final.pdf',
            format='pdf', dpi=300, bbox_inches='tight')
print("Figure 1 saved: gl_logr_linear_vortex_kappa20_final.pdf")
