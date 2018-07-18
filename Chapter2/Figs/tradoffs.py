#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.signal import medfilt
from matplotlib import rcParams
mpl.use('pgf')

params = {
   'axes.labelsize': 9,
   'text.fontsize': 9,
   'legend.fontsize': 8,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': True,
   'text.latex.unicode': True,
   'figure.dpi': 150,
   'figure.figsize': [6, 4],
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

subplotparams = {
   'axes.labelsize': 12,
   'text.fontsize': 12,
   'legend.fontsize': 10,
   'xtick.labelsize': 14,
   'ytick.labelsize': 14,
}
rcParams.update(params)

import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.figure()
    # axis
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(np.arange(3), ('Weak','Moderate','Strong'))
    plt.yticks(np.arange(3), ('', '', ''))
    plt.xlabel("Inversion Region")
    plt.ylabel("Channel Length $L$")
    # arrows
    Ars = 0.5
    Arx = 1
    Ary = 1
    for theta in [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]:
        ax.annotate("", xy=(Arx, Ary), xycoords='data',
                    xytext=(Arx+Ars*np.cos(theta), Ary+Ars*4/3*np.sin(theta)), textcoords='data', color='k',
                    va="center", ha="center", size=9,
                    arrowprops=dict(
                        arrowstyle="<-",
                        facecolor='black',
                        lw=2))
    # texts
    linespacing = 0.1
    xmargin     = 0.05
    # weak inversion whatsoever the L
    ax.text(xmargin, Ary+2*linespacing, r"Best:", weight='bold')
    ax.text(xmargin, Ary+1*linespacing, r"- $V_{gs}-V_{th}$ (min.)")
    ax.text(xmargin, Ary+0*linespacing, r"- $V_{dsSAT}$ (min.)")
    ax.text(xmargin, Ary-1*linespacing, r"- $g_m/I_D$ (max.)") 
    ax.text(xmargin, Ary-2*linespacing, r"- white $e_n$ (min.)")
    ax.text(xmargin, Ary-3*linespacing, r"- $f_T$ temperature", color='blue') 
    ax.text(xmargin, Ary-4*linespacing, r"  sensitivity (min.)", color='blue') 
    # weak inversion larger L
    ax.text(xmargin, .75+Ary+2*linespacing, r"Best:", weight='bold')
    ax.text(xmargin+0.175, .75+Ary+2*linespacing, r"low-power", weight='bold', style='italic')
    ax.text(xmargin, .75+Ary+1*linespacing, r"- DC Gain (max.)")
    ax.text(xmargin, .75+Ary+0*linespacing, r"- Matching")
    ax.text(xmargin, .75+Ary-1*linespacing, r"- flicker $e_n$ (min.)")
    ax.text(xmargin, .75+Ary-2*linespacing, r"- DC Gain temperature", color='blue') 
    ax.text(xmargin, .75+Ary-3*linespacing, r"  sensitivity (min.)", color='blue') 
    # Moderate inversion larger L
    ax.text(Arx-0.2, .75+Ary+1*linespacing, r"Best:", weight='bold')
    ax.text(Arx-0.2, .75+Ary+0*linespacing, r"- $r_{ds}$ (max.)")
    # Strong Inversion
    ax.text(Arx+Ars+0.025, Ary+2*linespacing, r"Best:", weight='bold')
    ax.text(Arx+Ars+0.025, Ary+1*linespacing, r"- $g_m$ linearity (max.)")
    ax.text(Arx+Ars+0.025, Ary+0*linespacing, r"- Temperature improves", color='blue')
    ax.text(Arx+Ars+0.025, Ary-1*linespacing, r"  linearity", color='blue')
    ax.text(Arx+Ars+0.025, Ary-2*linespacing, r"- DC Gain temperature", color='blue') 
    ax.text(Arx+Ars+0.025, Ary-3*linespacing, r"  sensitivity (min.)", color='blue') 
    # Strong Inversion smaller L
    ax.text(Arx+Ars-0.075, Ary-4*linespacing, r"Best:", weight='bold')
    ax.text(Arx+Ars-0.075+0.175,   Ary-4*linespacing, r"high-speed", weight='bold', style='italic')
    ax.text(Arx+Ars-0.075, Ary-5*linespacing, r"- $f_T$ (max.)")
    ax.text(Arx+Ars-0.075, Ary-6*linespacing, r"- C (min.)")
    ax.text(Arx+Ars-0.075, Ary-7*linespacing, r"- Layout Area (min.)")
    # fill area
    plt.tight_layout()
    plt.savefig("./Vector/tradeoffs.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
    plt.show()