#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import gamma
from matplotlib import rcParams

params = {
   'axes.labelsize': 12,
   'font.size': 12,
   'legend.fontsize': 16,
   'xtick.labelsize': 16,
   'ytick.labelsize': 16,
   'text.usetex': True,
   'savefig.format': 'pgf',
   'figure.dpi': 200,
   'figure.figsize': [4, 3],
   'mathtext.fontset': 'stix',
   'font.family':'STIXGeneral',
   'axes.color_cycle': ['#AA3939', '#AA9739', '#403075', '#2D882D', '#246B61', '#AA6839', '#333333'],
   "pgf.texsystem": "pdflatex", 
   "pgf.preamble": [
     r"\usepackage{gensymb}",
     r"\usepackage[utf8x]{inputenc}", # use utf8 fonts because your computer can handle it :)
     r"\usepackage[T1]{fontenc}", # plots will be generated using this preamble
   ]
}
rcParams.update(params)

import matplotlib.pyplot as plt

def ISD(vin, a):
    global Nstep, Cs_Ci
    g = (1+1/a)/(1+(1+Cs_Ci)/a)
    h = (Cs_Ci/(1+(1+Cs_Ci)/a))
    vo = 0
    b  = []
    for i in range(Nstep):
        if i == 0:
            b.extend([vin > 0.5, vin > -0.5])
        else:
            b.extend([vo+vin > 0.5, vo+vin > -0.5])
        vo = vo*g+h*(vin-(np.sum(b[-2:])-1))
    return (vo, np.sum(b)/Nstep/2)


def FormattedGain(a):
    return "{:.0f} dB".format(20*np.log10(a))

if __name__ == "__main__":
    N     = 2**12
    Npts  = 10*N
    Ngain = 10
    NAmax = 6
    NAmin = 2
    Nstep = 5
    Cs_Ci = 1

    vin   = np.linspace(-1, 1, Npts)
    codes = np.linspace(0, 1, N)

    A     = np.power(10, np.linspace(NAmin, NAmax, Ngain))

    INL   = np.zeros((Ngain, Npts))
    Res   = np.zeros((Ngain, Npts))
    Codes = np.zeros((Ngain, Npts))
    for ax, a in enumerate(A):
        for av, v in enumerate(vin):
            r, c = ISD(v, a)
            Res[ax][av]   = r
            Codes[ax][av] = c
    
    plt.figure()
    for ax, a in enumerate(A):
        plt.plot(vin, Res[ax], label=FormattedGain(a))
    plt.tight_layout()

    vout = np.add(Codes, np.divide(Res, 2*Nstep))
    plt.figure()
    for ax, a in enumerate(A):
        plt.plot(vin, vout[ax], label=FormattedGain(a))
    plt.tight_layout()
    
    plt.show()
