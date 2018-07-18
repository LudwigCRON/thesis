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


def quantizer_offset(VDD, C0, dC0, ki, Cp):
    sqt = np.sum(list(map(np.sqrt, ki)))
    t = np.sum(ki)
    ans = VDD*dC0*sqt/t
    ans = (1-VDD)*(C0*dC0*sqt)**2/(Cp+C0*t)/t/C0
    return ans

if __name__ == "__main__":
    WL      = np.linspace(1,64,50)
    Ns      = 6    # sigma process variations
    Acmim   = 4e-3 # 0.4 %.um
    CMIMdes = 1    # 1 fF/um**2
    C0      = WL*CMIMdes # fF
    dC0     = Ns/2*Acmim*CMIMdes*(WL)**(1/2) # fF divided by 2 since +dCo on positive side and -dCo on negative side

    offset = quantizer_offset(1.8, C0, dC0, [1, 2, 2], 10)

    fig, ax1 = plt.subplots()
    ax1.plot(WL, 100*dC0/C0, 'r-')
    ax1.set_xlabel("Area [$\mu m^2$]")
    ax1.set_ylabel("${}\sigma$  Capacitor Variation [%/$\mu m^2$]".format(Ns))
    ax2 = ax1.twinx()
    ax2.plot(WL, C0, 'b-')
    ax2.plot(WL, dC0, 'g-')
    ax2.set_ylabel("Capacitor [fF]")
    plt.tight_layout()

    plt.figure()
    plt.plot(WL, offset)
    plt.xlabel("Area [$\mu m^2$]")
    plt.show()