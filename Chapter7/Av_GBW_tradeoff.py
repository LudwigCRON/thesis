#!/usr/bin/env python3

import numpy as np
from matplotlib import cm
from matplotlib import rcParams

params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'savefig.format': 'pdf',
   'figure.dpi': 200,
   'figure.figsize': [4, 3],
   'mathtext.fontset': 'stix',
   'font.family':'STIXGeneral',
   'axes.color_cycle': ['#AA3939', '#AA9739', '#403075', '#2D882D', '#246B61', '#AA6839', '#333333']
}
rcParams.update(params)

import matplotlib.pyplot as plt

def formatter(r):
    t = np.log2(1/(r))
    print(r, t)
    return "{}-bits".format(int(t))

if __name__ == "__main__":
    t_T = np.linspace(5, 20, 300)
    Av  = np.power(10, np.linspace(2.5, 6, 100))
    X, Y = np.meshgrid(t_T, Av)
    error = np.add(np.exp(-X), np.multiply(1, np.divide(1, Y)))
    norm = cm.colors.Normalize(vmax=abs(error).max(), vmin=-abs(error).max())
    cmap = cm.PRGn

    N = np.linspace(16, 10, 4)
    levels = np.power(0.5, N)
    plt.figure()
    cs = plt.contour(X, 20*np.log10(Y), error, levels,
        cmap=cm.get_cmap(cmap, len(levels) - 1), linewidths=2)
    # find pareto front
    curves_y = []
    curves_x = []
    for c in cs.collections:
        curve = []
        for idx, path in enumerate(c.__dict__["_paths"]):
            curve.extend(path.__dict__["_vertices"].tolist())
        curve = np.array(curve)
        curve = curve[np.argsort(curve[:,0])]
        curves_y.append(curve)

    plt.clabel(cs, inline=True, inline_spacing=20, fontsize=10, fmt=formatter, manual=[c[-40] for c in curves_y])

    for i, c in enumerate(curves_y):
        dn  = np.divide(np.diff(c[:,1]), np.diff(c[:,0]))
        ddn = np.divide(np.diff(dn), np.diff(np.diff(c[:,0])))
        ddn = list(filter(np.isfinite, ddn))
        idx = ddn.index(np.nanmax(ddn))
        if i == 1:
            idx -= 7
        plt.plot(c[idx,0], c[idx,1], marker='o', color=cs.collections[i].__dict__["_edgecolors"][0])
        plt.text(c[idx,0]+0.5, c[idx+7,1]+4, "({:.2f}, {:.0f} dB)".format(c[idx,0], c[idx,1]))
    
    plt.xlabel(r"$t/\tau$")
    plt.ylabel(r"$A_v$ [dB]")
    plt.tight_layout()
    plt.savefig("./Figs/ota_spec."+params['savefig.format'], format=params['savefig.format'], dpi=params['figure.dpi'], bbox_inches="tight")
    plt.show()