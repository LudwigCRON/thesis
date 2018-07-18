#!/usr/bin/env python3

import numpy as np
from matplotlib import cm
from matplotlib import rcParams

params = {
   'axes.labelsize': 12,
   'font.size': 12,
   'legend.fontsize': 12,
   'xtick.labelsize': 12,
   'ytick.labelsize': 12,
   'text.usetex': True,
   'savefig.format': 'pgf',
   'figure.dpi': 200,
   'figure.figsize': [4, 3],
   'mathtext.fontset': 'stix',
   #'font.family':'STIXGeneral',
   'axes.color_cycle': ['#AA3939', '#AA9739', '#403075', '#2D882D', '#246B61', '#AA6839', '#333333']
}
rcParams.update(params)

import matplotlib.pyplot as plt

def formatter(r):
    t = np.log2(1/(r))
    print(r, t)
    return "{}-bits".format(int(t))

def algo_vout(a, b, vth, vin, N):
    vo = vin
    for i in range(N):
        if vo > vth:
            vo = a*vo-b
        elif vo < -vth:
            vo = a*vo+b
        else:
            vo = a*vo
    return vo

def algo_error(g, f, vin, vth, A, N):
    k = np.linspace(0, N, N+1)
    if isinstance(g, (int, float)):
        if isinstance(f, (int, float)):
            err = 0.
        else:
            err = np.zeros((1, len(f)))
    else:
        if isinstance(f, (int, float)):
            err = np.zeros((len(g), 1))
        else:
            err = np.zeros((len(g), len(f)))
    #a = [(1-(1/(1+(2*g+f-1)/A))**ki)*2**(ki-1) for ki in k]
    #for i in k:
    #    err = err + a[int(i)]
    #err = err + vin*g*(1-(1/(1+(2*g+f-1)/A))**N)*2**N
    ratio = 1+(2*g+f-1)/A
    return (algo_vout(2., f, g*vin, vth, N)-algo_vout(2./ratio, f/ratio, g*vin, vth, N))

def max_algo_error(g, f, vinmax, vth, A, N):
    e = []
    for vin in np.linspace(1e-2, vinmax, 128):
        e.append(algo_error(g, f, vin, vth, A, N))
    return np.max(e)

if __name__ == "__main__":
    g = np.linspace(1, 2, 5) # interstage gain
    f = np.linspace(1, 2, 5) # feedback gain
    X, Y = np.meshgrid(g, f)

    N = 5 # number of clock cycle
    Av  = np.power(10, np.linspace(2.5, 6, 50)) # DC gain of the OTA
    #errors = []
    #for A in Av:
    #    errors.append(error(X, Y, 0.5, A, N))

    plt.figure()
    plt.plot(20*np.log10(Av), [1e3*max_algo_error(1, 2.0, 0.5, 0.5, A, N) for A in Av], label="b = 2")
    plt.plot(20*np.log10(Av), [1e3*max_algo_error(1, 1.6, 0.5, 0.4, A, N) for A in Av], label="b = 1.6")
    plt.plot(20*np.log10(Av), [1e3*max_algo_error(1, 1.2, 0.5, 0.3, A, N) for A in Av], label="b = 1.2")
    plt.plot(20*np.log10(Av), [1e3*max_algo_error(1, 1.0, 0.5, 0.25, A, N) for A in Av], label="b = 1")
    plt.xlabel("DC Gain [dB]")
    plt.ylabel("Error [mV]")
    plt.axis([65, 90, 0, 16])
    plt.legend()
    plt.tight_layout()
    plt.savefig("./Figs/algo_error_CB_impact."+params['savefig.format'], format=params['savefig.format'], dpi=params['figure.dpi'], bbox_inches="tight")

    plt.figure()
    #plt.plot(range(1,N+1), [algo_error(1,1.6,0.5,0.4,2500, i) for i in range(N)])
    plt.plot(20*np.log10(Av), [1e3*max_algo_error(2   , 1.2, 0.5, 0.3, A, N) for A in Av], label="g = 2")
    plt.plot(20*np.log10(Av), [1e3*max_algo_error(1.5 , 1.2, 0.5, 0.3, A, N) for A in Av], label="g = 1.5")
    plt.plot(20*np.log10(Av), [1e3*max_algo_error(1.25, 1.2, 0.5, 0.3, A, N) for A in Av], label="g = 1.25")
    plt.plot(20*np.log10(Av), [1e3*max_algo_error(1   , 1.2, 0.5, 0.3, A, N) for A in Av], label="g = 1")
    plt.xlabel("DC Gain [dB]")
    plt.ylabel("Error [mV]")
    plt.axis([65, 90, 0, 16])
    plt.legend()
    plt.tight_layout()
    plt.savefig("./Figs/algo_error_CG_impact."+params['savefig.format'], format=params['savefig.format'], dpi=params['figure.dpi'], bbox_inches="tight")

    plt.show()
