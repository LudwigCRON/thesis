#!/usr/bin/env python3

import numpy as np
import pandas as pd
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

subplotparams = {
   'axes.labelsize': 12,
   'text.fontsize': 12,
   'legend.fontsize': 14,
   'xtick.labelsize': 14,
   'ytick.labelsize': 14,
}
rcParams.update(params)

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # load vthn temp tm case
    #db = pd.read_csv("./nel_1000_180_vth_temp.csv")
    #keys = [k for k in db]
    #temp = db[keys[0]]
    #vthn = db[keys[1]]
    # display vthn temp tm case
    #plt.figure()
    #plt.plot(273+temp, vthn, 'b-', linewidth=2., label="nel 1$\mu$m/180nm")
    #plt.xlabel("Temperature [$\degree K$]")
    #plt.ylabel("Voltage [V]")
    #plt.tight_layout()
    #plt.savefig("./Raster/vth_nel.png", bbox_inches="tight", dpi=150)
    #plt.show()
    # load vthn temp mc_g 3s
    db = pd.read_csv("./nel_1000_180_vth_mc_temp.csv")
    temp = db.groupby(['temperature'])['temperature'].mean().values
    vthn_mean = db.groupby(['temperature'])['Vthn'].mean().values
    vthn_std = db.groupby(['temperature'])['Vthn'].std().values
    vthnm3s = vthn_mean-3*vthn_std
    vthnp3s = vthn_mean+3*vthn_std
    # display vthn temp mc_g 3s
    rcParams.update(subplotparams)
    plt.figure()
    ax = plt.gca()
    ax.fill_between(temp, vthnm3s, vthnp3s, color='b', alpha=0.25, linewidth=0.1)
    plt.plot(temp, vthn_mean, 'b-', linewidth=2., label="nel 1$\mu$m/180nm")
    plt.xlabel("Temperature [$\degree C$]")
    plt.ylabel("Voltage [V]")
    plt.tight_layout()
    plt.savefig("./Vector/vth_nel.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
    #plt.show()
    # load vthn ids gm ron temp tm case
    db = pd.read_csv("./nel_1000_180_ids_gm_ron_vthn_temp.csv")
    gdb  = db.groupby(['temperature'])
    temp = gdb['temperature'].mean().values
    vov40 = db.where(db.temperature == -40)['vgs']#-db.where(db.temperature == -40)['Vthn']
    jds40 = db.where(db.temperature == -40)['Ids']
    vov27 = db.where(db.temperature == 27)['vgs']#-db.where(db.temperature == 27)['Vthn']
    jds27 = db.where(db.temperature == 27)['Ids']
    vov175 = db.where(db.temperature == 175)['vgs']#-db.where(db.temperature == 175)['Vthn']
    jds175 = db.where(db.temperature == 175)['Ids']
    rcParams.update(params)
    plt.figure()
    plt.semilogy(vov40, jds40, 'k-', label="nel $-40 \degree C$")
    plt.semilogy(vov27, jds27, 'k--', label="nel $27 \degree C$")
    plt.semilogy(vov175, jds175, 'k-.', label="nel $175 \degree C$")
    plt.plot(0.7909, 121.478e-6, 'mo', alpha=0.5)
    plt.xlabel("Gate Voltage [V]")
    plt.ylabel("Current Density [A/m]")
    plt.axis([0.5, 1,  10e-6, 300e-6])
    plt.legend()
    ax = plt.gca()
    ax.annotate(r'$\frac{\partial J_{DS}}{\partial T} \approx 0 $', xy=(0.7909, 121.478e-6), xycoords='data',
                xytext=(0.7909, 200e-6), textcoords='data', color='k',
                va="center", ha="center", size=9,
                arrowprops=dict(
                    arrowstyle="->",
                    facecolor='black',
                    lw=1))
    ax.axvline(x=0.7909, linewidth=1, color='k', linestyle='--', alpha=0.3)
    ax.annotate("Drift Dominates", xy=(0.7909, 20e-6), xycoords='data',
                xytext=(0.85, 20e-6), textcoords='data', color='b',
                va="center", ha="left", size=9,
                arrowprops=dict(
                    arrowstyle="<-",
                    facecolor='b',
                    lw=1))
    ax.annotate("Diffusion Dominates", xy=(0.7909, 20e-6), xycoords='data',
                xytext=(0.75, 20e-6), textcoords='data', color='r',
                va="center", ha="right", size=9,
                arrowprops=dict(
                    arrowstyle="<-",
                    facecolor='r',
                    lw=1))
    plt.tight_layout()
    plt.savefig("./Vector/jds_nel.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
    plt.show()
    