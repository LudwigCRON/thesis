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
    # load gm_id data for 0.18 um
    db = pd.read_csv("./nel_1000_180_ids_gm_ron_vthn_temp.csv")
    gdb  = db.groupby(['temperature'])
    temp = gdb['temperature'].mean().values
    vgs40_180 = db.where(db.temperature == -40)['vgs']#-db.where(db.temperature == -40)['Vthn']
    Ids40_180 = db.where(db.temperature == -40)['Ids']
    gm40_180 = db.where(db.temperature == -40)['gmn']
    ro40_180 = db.where(db.temperature == -40)['Ron']
    gm_id40_180 = np.divide(gm40_180, Ids40_180)
    A40_180 = np.multiply(gm40_180, ro40_180)
    vgs27_180 = db.where(db.temperature == 27)['vgs']#-db.where(db.temperature == 27)['Vthn']
    Ids27_180 = db.where(db.temperature == 27)['Ids']
    gm27_180 = db.where(db.temperature == 27)['gmn']
    ro27_180 = db.where(db.temperature == 27)['Ron']
    gm_id27_180 = np.divide(gm27_180, Ids27_180)
    A27_180 = np.multiply(gm27_180, ro27_180)
    vgs175_180 = db.where(db.temperature == 175)['vgs']#-db.where(db.temperature == 175)['Vthn']
    Ids175_180 = db.where(db.temperature == 175)['Ids']
    gm175_180 = db.where(db.temperature == 175)['gmn']
    ro175_180 = db.where(db.temperature == 175)['Ron']
    gm_id175_180 = np.divide(gm175_180, Ids175_180)
    A175_180 = np.multiply(gm175_180, ro175_180)
    vt_180 = db.groupby('temperature').mean()['Vthn']
    # load gm_id data for 0.36 um
    db = pd.read_csv("./nel_1000_360_ids_gm_ron_vthn_temp.csv")
    gdb  = db.groupby(['temperature'])
    temp = gdb['temperature'].mean().values
    vgs40_360 = db.where(db.temperature == -40)['vgs']#-db.where(db.temperature == -40)['Vthn']
    Ids40_360 = db.where(db.temperature == -40)['Ids']
    gm40_360 = db.where(db.temperature == -40)['gmn']
    ro40_360 = db.where(db.temperature == -40)['Ron']
    gm_id40_360 = np.divide(gm40_360, Ids40_360)
    A40_360 = np.multiply(gm40_360, ro40_360)
    vgs27_360 = db.where(db.temperature == 27)['vgs']#-db.where(db.temperature == 27)['Vthn']
    Ids27_360 = db.where(db.temperature == 27)['Ids']
    gm27_360 = db.where(db.temperature == 27)['gmn']
    ro27_360 = db.where(db.temperature == 27)['Ron']
    gm_id27_360 = np.divide(gm27_360, Ids27_360)
    A27_360 = np.multiply(gm27_360, ro27_360)
    vgs175_360 = db.where(db.temperature == 175)['vgs']#-db.where(db.temperature == 175)['Vthn']
    Ids175_360 = db.where(db.temperature == 175)['Ids']
    gm175_360 = db.where(db.temperature == 175)['gmn']
    ro175_360 = db.where(db.temperature == 175)['Ron']
    gm_id175_360 = np.divide(gm175_360, Ids175_360)
    A175_360 = np.multiply(gm175_360, ro175_360)
    vt_360 = db.groupby('temperature').mean()['Vthn']
    # load gm_id data for 0.54 um
    db = pd.read_csv("./nel_1000_540_ids_gm_ron_vthn_temp.csv")
    gdb  = db.groupby(['temperature'])
    temp = gdb['temperature'].mean().values
    vgs40_540 = db.where(db.temperature == -40)['vgs']#-db.where(db.temperature == -40)['Vthn']
    Ids40_540 = db.where(db.temperature == -40)['Ids']
    gm40_540 = db.where(db.temperature == -40)['gmn']
    ro40_540 = db.where(db.temperature == -40)['Ron']
    gm_id40_540 = np.divide(gm40_540, Ids40_540)
    A40_540 = np.multiply(gm40_540, ro40_540)
    vgs27_540 = db.where(db.temperature == 27)['vgs']#-db.where(db.temperature == 27)['Vthn']
    Ids27_540 = db.where(db.temperature == 27)['Ids']
    gm27_540 = db.where(db.temperature == 27)['gmn']
    ro27_540 = db.where(db.temperature == 27)['Ron']
    gm_id27_540 = np.divide(gm27_540, Ids27_540)
    A27_540 = np.multiply(gm27_540, ro27_540)
    vgs175_540 = db.where(db.temperature == 175)['vgs']#-db.where(db.temperature == 175)['Vthn']
    Ids175_540 = db.where(db.temperature == 175)['Ids']
    gm175_540 = db.where(db.temperature == 175)['gmn']
    ro175_540 = db.where(db.temperature == 175)['Ron']
    gm_id175_540 = np.divide(gm175_540, Ids175_540)
    A175_540 = np.multiply(gm175_540, ro175_540)
    vt_540 = db.groupby('temperature').mean()['Vthn']
    # load gm_id data for 0.72 um
    db = pd.read_csv("./nel_1000_720_ids_gm_ron_vthn_temp.csv")
    gdb  = db.groupby(['temperature'])
    temp = gdb['temperature'].mean().values
    vgs40_720 = db.where(db.temperature == -40)['vgs']#-db.where(db.temperature == -40)['Vthn']
    Ids40_720 = db.where(db.temperature == -40)['Ids']
    gm40_720 = db.where(db.temperature == -40)['gmn']
    ro40_720 = db.where(db.temperature == -40)['Ron']
    gm_id40_720 = np.divide(gm40_720, Ids40_720)
    A40_720 = np.multiply(gm40_720, ro40_720)
    vgs27_720 = db.where(db.temperature == 27)['vgs']#-db.where(db.temperature == 27)['Vthn']
    Ids27_720 = db.where(db.temperature == 27)['Ids']
    gm27_720 = db.where(db.temperature == 27)['gmn']
    ro27_720 = db.where(db.temperature == 27)['Ron']
    gm_id27_720 = np.divide(gm27_720, Ids27_720)
    A27_720 = np.multiply(gm27_720, ro27_720)
    vgs175_720 = db.where(db.temperature == 175)['vgs']#-db.where(db.temperature == 175)['Vthn']
    Ids175_720 = db.where(db.temperature == 175)['Ids']
    gm175_720 = db.where(db.temperature == 175)['gmn']
    ro175_720 = db.where(db.temperature == 175)['Ron']
    gm_id175_720 = np.divide(gm175_720, Ids175_720)
    A175_720 = np.multiply(gm175_720, ro175_720)
    vt_720 = db.groupby('temperature').mean()['Vthn']

    plt.figure()
    p1 = plt.plot(gm_id40_180, A40_180, '-', label="nel $-40 \degree C$")
    p2 = plt.plot(gm_id27_180, A27_180, '--', label="nel $27 \degree C$")
    p3 = plt.plot(gm_id175_180, A175_180, '-.', label="nel $175 \degree C$")
    plt.plot(np.nanmax(gm_id40_180), np.nanmax(A40_180), 'o', color=p1[0].get_color())
    plt.plot(np.nanmax(gm_id27_180), np.nanmax(A27_180), 'o', color=p2[0].get_color())
    plt.plot(np.nanmax(gm_id175_180), np.nanmax(A175_180), 'o', color=p3[0].get_color())
    ax = plt.gca()
    ax.annotate(r'$A_{max} @ -40 \degree C$', xy=(np.nanmax(gm_id40_180), np.nanmax(A40_180)), xycoords='data',
                xytext=(np.nanmax(gm_id40_180)-10, np.nanmax(A40_180)-2), textcoords='data',
                va="center", ha="center", size=12,
                arrowprops=dict(
                    arrowstyle="->",
                    facecolor='black',
                    lw=1))
    ax.annotate(r'$A_{max} @ 27 \degree C$', xy=(np.nanmax(gm_id27_180), np.nanmax(A27_180)), xycoords='data',
                xytext=(np.nanmax(gm_id27_180)-10, np.nanmax(A27_180)-2), textcoords='data',
                va="center", ha="center", size=12,
                arrowprops=dict(
                    arrowstyle="->",
                    facecolor='black',
                    lw=1))
    ax.annotate(r'$A_{max} @ 175 \degree C$', xy=(np.nanmax(gm_id175_180), np.nanmax(A175_180)), xycoords='data',
                xytext=(np.nanmax(gm_id175_180)-10, np.nanmax(A175_180)-2), textcoords='data',
                va="center", ha="center", size=12,
                arrowprops=dict(
                    arrowstyle="->",
                    facecolor='black',
                    lw=1))
    plt.xlabel("$g_m/I_D$ [S/A]")
    plt.ylabel("Intrinsic Gain [V/V]")
    plt.tight_layout()
    plt.savefig("./Vector/gmro_nel.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
    plt.figure()
    plt.semilogx(Ids40_180, vgs40_180, '-', label="nel $-40 \degree C$")
    plt.semilogx(Ids27_180, vgs27_180, '--', label="nel $27 \degree C$")
    plt.semilogx(Ids175_180, vgs175_180, '-.', label="nel $175 \degree C$")
    plt.xlabel("Drain-Source Current [A]")
    plt.ylabel("Gate Voltage [V]")
    plt.tight_layout()
    plt.savefig("./Vector/gatebias_nel.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
    plt.figure()
    plt.semilogx(Ids40_180, gm_id40_180, '-', label="nel $-40 \degree C$")
    plt.semilogx(Ids27_180, gm_id27_180, '--', label="nel $27 \degree C$")
    plt.semilogx(Ids175_180, gm_id175_180, '-.', label="nel $175 \degree C$")
    plt.xlabel("Drain-Source Current [A]")
    plt.ylabel("$g_m/I_D$ [S/A]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./Vector/gm_id_nel.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
    rcParams.update(subplotparams)
    plt.figure()
    plt.plot(temp[:-1], np.diff(vt_180)/np.diff(temp)*1e3, '-', label="L = 180 nm")
    plt.plot(temp[:-1], np.diff(vt_360)/np.diff(temp)*1e3, '--', label="L = 360 nm")
    plt.plot(temp[:-1], np.diff(vt_540)/np.diff(temp)*1e3, '-.', label="L = 540 nm")
    plt.plot(temp[:-1], np.diff(vt_720)/np.diff(temp)*1e3, ':', label="L = 720 nm")
    plt.xlabel("Temperature [$\degree$C]")
    plt.ylabel(r"$\frac{\partial V_{th}}{\partial T}$ [mV/$\degree$C]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./Vector/vth_nel_T.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
    rcParams.update(params)
    plt.figure()
    plt.semilogx(Ids40_180, ro40_180, '-', label="nel $-40 \degree C$")
    plt.semilogx(Ids27_180, ro27_180, '--', label="nel $27 \degree C$")
    plt.semilogx(Ids175_180, ro175_180, '-.', label="nel $175 \degree C$")
    plt.xlabel("Drain-Source Current [A]")
    plt.ylabel("$r_{ds}$ [$\Omega$]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./Vector/gm_id_nel.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
    plt.show()
    