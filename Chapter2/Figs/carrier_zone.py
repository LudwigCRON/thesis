#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
from scipy.integrate import quad
from matplotlib import rcParams
mpl.use('pgf')

params = {
   'axes.labelsize': 9,
   'text.fontsize': 9,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': True,
   'text.latex.unicode': True,
   'figure.dpi': 150,
   'figure.figsize': [4, 3],
   'mathtext.fontset': 'stix',
   'font.family':'STIXGeneral',
   'savefig.format': 'pgf',
   "pgf.texsystem": "pdflatex", 
   "pgf.preamble": [
     r"\usepackage{gensymb}",
     r"\usepackage[utf8x]{inputenc}", # use utf8 fonts because your computer can handle it :)
     r"\usepackage[T1]{fontenc}", # plots will be generated using this preamble
   ]
}
rcParams.update(params)

import matplotlib.pyplot as plt

TMAX = 1000
TMIN = -240
T0   = 27

m0 = 9.109e-31
me = 1.08*m0
mh = 0.81*m0

kB = 1.38e-23
q  = 1.609e-19
kB_q = kB/q
h  = 6.63e-34
hb = h/(2*np.pi)
kB_h2 = 3.1446e43

ND = 1e16   # cm^-3
NA = 1e14

def EgT(t):
    return 1.166 - 4.73e-4*(t**2)/(t+636)

def EcT(t):
    return (EgT(0)+EgT(t))/2

def EvT(t):
    return (EgT(0)-EgT(t))/2

if __name__ == "__main__":
    Nb = 100
    T = np.linspace(273+TMIN,273+TMAX, Nb)
    # Eg from Varshni equation
    Eg0 = EgT(0) # eV
    Eg  = EgT(T) # eV

    # Ec and Ev
    Ec = EcT(T)
    Ev = EvT(T)
    Ei = Eg/2+3/4*kB_q*T*np.log(mh/me)

    #https://ecee.colorado.edu/~bart/book/book/chapter2/ch2_6.htm
    # Density of States [Checked]
    NC0 = 2e-6*(2*np.pi*me*kB_h2)**(1.5) # electrons/cm^-3/K^-3/2
    NV0 = 2e-6*(2*np.pi*mh*kB_h2)**(1.5) # electrons/cm^-3/K^-3/2
    NC  = NC0 * T**1.5
    NV  = NV0 * T**1.5
    ni  = np.sqrt(NC*NV*np.exp(-Eg/(kB_q*T)))
    n0  = (ND-NA)/2 + np.sqrt(((ND-NA)/2)**2+ni**2)
    p0  = (NA-ND)/2 + np.sqrt(((NA-ND)/2)**2+ni**2)
    Efn  = Eg0/2+kB_q*T*np.log(n0/ni)
    Efp  = Eg0/2+kB_q*T*np.log(p0/ni)

    # Donnors and Acceptors [Checked]
    Ea = 50e-3 # eV
    Ed = 50e-3 # eV
    Efs = np.linspace(-1,Eg0+0.5,1000)
    Ef = []
    for t in T:
        Ndp = ND/(1+2*np.exp(np.divide(np.subtract(Efs,np.subtract(EgT(t),Ed)),kB_q*t)))
        Nam = NA/(1+4*np.exp(np.subtract(Ea, Efs)/(kB_q*t)))
        n   = NC0*(t**1.5)*np.exp(np.divide(np.subtract(Efs, EgT(t)), kB_q*t))
        p   = NV0*(t**1.5)*np.exp(-np.divide(Efs, kB_q*t))
        # find fermi level
        f = n+Nam
        g = p+Ndp
        idx = np.argwhere(np.diff(np.sign(f - g)) != 0).reshape(-1) + 0
        if idx >= 0:
            Ef.append(Efs[idx][0])
        else:
            Ef.append(min(Efs))
        # display fermi level calculation
        #plt.figure()
        #plt.semilogy(Efs, p+Ndp, 'r')
        #plt.semilogy(Efs, n+Nam, 'b')
        #plt.semilogy(Efs, Nam, 'k')
        #plt.semilogy(Efs, Ndp, 'g')
        #plt.xlabel("Fermi Energy [eV]")
        #plt.ylabel("Carrier Density [$cm^-3$]")
        #plt.tight_layout()

    # update values
    Ndp = ND/(1+0.5*np.exp(np.divide(np.subtract(Ef,np.subtract(Eg,Ed)),kB_q*T)))
    Nam = NA/(1+0.25*np.exp(np.divide(np.subtract(Ea, Ef),kB_q*T)))
    n   = NC*np.exp(np.divide(np.subtract(Ef,Eg), kB_q*T))
    p   = NV*np.exp(-np.divide(Ef,kB_q*T))

    # display energy
    plt.figure()
    plt.plot(T, Eg, 'b-', linewidth=2., label="Band Gap")
    ax = plt.gca()
    ax.annotate('', xy=(452, 1.12), xycoords='data',
                xytext=(300, 1.12), textcoords='data',
                va="center", ha="center",
                arrowprops=dict(
                    arrowstyle="->",
                    facecolor='black',
                    lw=1))
    ax.annotate('', xy=(450, 1.077), xycoords='data',
                xytext=(450, 1.12), textcoords='data',
                va="center", ha="center",
                arrowprops=dict(
                    arrowstyle="->",
                    facecolor='black',
                    lw=1))
    ax.text(350, 1.122, '400 ppm/$\degree$K')
    plt.xlabel("Temperature [$\degree K$]")
    plt.ylabel("Energy [eV]")
    plt.axis([250, 475, 1.07, 1.14])
    plt.tight_layout()
    plt.savefig("./Vector/bandgap.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
    plt.show()

    # display Fermi
    plt.figure()
    plt.plot(T, Efn/Eg0, 'b-', linewidth=2., label="Fermi N")
    #plt.plot(T, Efp/Eg0, 'g-', linewidth=2., label="Fermi P")
    plt.plot(T, np.divide(Ef,Eg0), 'k-.', linewidth=2., label="Fermi Found")
    plt.legend()
    plt.xlabel("Temperature [$\degree K$]")
    plt.ylabel("Energy [eV/Eg($T_0$)]")
    plt.axis([250, 475, 0, 1])
    plt.tight_layout()
    plt.savefig("./Vector/fermi.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
    plt.show()

    # display carrier
    Tdes_rng = [1000/t for t in T if (t >= 250 and t <= 450)]
    ndes_rng = [n[i]/ND for i in range(len(T)) if (T[i] >= 250 and T[i] <= 450)]
    plt.figure()
    plt.semilogy(1000/T, n/ND, 'b-', linewidth=2., label="electrons density")
    plt.semilogy(Tdes_rng, ndes_rng, 'g-', linewidth=2., label="temperature range for the design")
    ax = plt.gca()
    ax.annotate('Intrinsic', xy=(1000/T[-3], n[-3]/ND), xycoords='data',
                xytext=(5, 100), textcoords='data',
                va="center", ha="center",
                arrowprops=dict(
                    arrowstyle="->",
                    facecolor='black',
                    lw=1))
    ax.annotate('Extrinsic', xy=(1000/T[25], n[25]/ND), xycoords='data',
                xytext=(7, 10), textcoords='data',
                va="center", ha="center",
                arrowprops=dict(
                    arrowstyle="->",
                    facecolor='black',
                    lw=1))
    ax.annotate('Freeze-Out', xy=(1000/T[2], n[2]/ND), xycoords='data',
                xytext=(20, 1), textcoords='data',
                va="center", ha="center",
                arrowprops=dict(
                    arrowstyle="->",
                    facecolor='black',
                    lw=1))
    ax.axvline(x=1.8, linewidth=1, color='r', linestyle='--')
    ax.axvline(x=9.5, linewidth=1, color='r', linestyle='--')
    plt.xlabel("1000/Temperature [$1000/\degree K$]")
    plt.ylabel("Electron Density [$n/N_D$]")
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig("./Vector/carrier_density.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
    plt.show()