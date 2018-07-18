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
   'legend.fontsize': 10,
   'xtick.labelsize': 14,
   'ytick.labelsize': 14,
}
rcParams.update(params)

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # load gm_id data
    db = pd.read_csv("./xt018_gm_id.csv", na_values=['', '#N/A', '\#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null', 'sim err', 'eval err'], dtype={'gmn_raw': np.float32, 'idn': np.float32, 'gmp_raw': np.float32, 'idp': np.float32, 'vgsn_raw': np.float32, 'vgsp_raw': np.float32, 'gdsp_raw': np.float32, 'gdsn_raw': np.float32, 'Ftn': np.float32, 'Ftp': np.float32})
    gdb  = db.groupby(['temperature'])
    temp = gdb['temperature'].mean().values
    print([k for k in db], temp)
    db_180 = db.where((db['M0.l'] == 180e-9) & (db['VDS'] == 0.9) & (db.temperature == 27))
    db_360 = db.where((db['M0.l'] == 360e-9) & (db['VDS'] == 0.9) & (db.temperature == 27))
    db_540 = db.where((db['M0.l'] == 540e-9) & (db['VDS'] == 0.9) & (db.temperature == 27))
    db_720 = db.where((db['M0.l'] == 720e-9) & (db['VDS'] == 0.9) & (db.temperature == 27))
    db_1000 = db.where((db['M0.l'] == 1000e-9) & (db['VDS'] == 0.9) & (db.temperature == 27))
    idn_180  = pd.to_numeric(db_180['idn'], errors='coerce').dropna()
    idn_360  = pd.to_numeric(db_360['idn'], errors='coerce').dropna()
    idn_540  = pd.to_numeric(db_540['idn'], errors='coerce').dropna()
    idn_720  = pd.to_numeric(db_720['idn'], errors='coerce').dropna()[:-1]
    idn_1000  = pd.to_numeric(db_1000['idn'], errors='coerce').dropna()
    ftn_180  = pd.to_numeric(db_180['Ftn'], errors='coerce').dropna()
    ftn_360  = pd.to_numeric(db_360['Ftn'], errors='coerce').dropna()
    ftn_540  = pd.to_numeric(db_540['Ftn'], errors='coerce').dropna()
    ftn_720  = pd.to_numeric(db_720['Ftn'], errors='coerce').dropna()[:-1]
    ftn_1000  = pd.to_numeric(db_1000['Ftn'], errors='coerce').dropna()
    gmn_180  = pd.to_numeric(db_180['gmn_raw'], errors='coerce').dropna()
    gmn_360  = pd.to_numeric(db_360['gmn_raw'], errors='coerce').dropna()
    gmn_540  = pd.to_numeric(db_540['gmn_raw'], errors='coerce').dropna()
    gmn_720  = pd.to_numeric(db_720['gmn_raw'], errors='coerce').dropna()[:-1]
    gmn_1000  = pd.to_numeric(db_1000['gmn_raw'], errors='coerce').dropna()
    gdsn_180  = pd.to_numeric(db_180['gdsn_raw'], errors='coerce').dropna()
    gdsn_360  = pd.to_numeric(db_360['gdsn_raw'], errors='coerce').dropna()
    gdsn_540  = pd.to_numeric(db_540['gdsn_raw'], errors='coerce').dropna()
    gdsn_720  = pd.to_numeric(db_720['gdsn_raw'], errors='coerce').dropna()[:-1]
    gdsn_1000  = pd.to_numeric(db_1000['gdsn_raw'], errors='coerce').dropna()
    # display ron for different L
    RScale = 1
    plt.figure()
    plt.loglog(idn_180, 1/gdsn_180*RScale, label="L = 180 nm")
    plt.loglog(idn_360, 1/gdsn_360*RScale, label="L = 360 nm")
    plt.loglog(idn_540, 1/gdsn_540*RScale, label="L = 540 nm")
    plt.loglog(idn_720, 1/gdsn_720*RScale, label="L = 720 nm")
    plt.loglog(idn_1000, 1/gdsn_1000*RScale, label="L = 1 $\mu$m")
    plt.xlabel("Drain-Source Current [A]")
    plt.ylabel("$R_{ds}$ [$\Omega$]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./Vector/Rds_nel_L.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
    # display Ftn for different L
    FtScale = 1
    rcParams.update(subplotparams)
    plt.figure()
    plt.loglog(idn_180, ftn_180*FtScale, label="L = 180 nm")
    plt.loglog(idn_360, ftn_360*FtScale, label="L = 360 nm")
    plt.loglog(idn_540, ftn_540*FtScale, label="L = 540 nm")
    plt.loglog(idn_720, ftn_720*FtScale, label="L = 720 nm")
    plt.loglog(idn_1000, ftn_1000*FtScale, label="L = 1 $\mu$m")
    plt.xlabel("Drain-Source Current [A]")
    plt.ylabel("Transit Frequency [Hz]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./Vector/ft_nel_L.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
    #display ft temperature sensitivity for 25 uA current
    sFtScale = 1
    ids = db.where(db['IDS'] < 1e-4).groupby('IDS')['IDS'].mean().values
    db_180 = db.where((db['M0.l'] == 180e-9) & (db['VDS'] < 0.9) & (db['IDS'] < 1e-4)).groupby('IDS')
    db_360 = db.where((db['M0.l'] == 360e-9) & (db['VDS'] < 0.9) & (db['IDS'] < 1e-4)).groupby('IDS')
    db_540 = db.where((db['M0.l'] == 540e-9) & (db['VDS'] < 0.9) & (db['IDS'] < 1e-4)).groupby('IDS')
    db_720 = db.where((db['M0.l'] == 720e-9) & (db['VDS'] < 0.9) & (db['IDS'] < 1e-4)).groupby('IDS')
    db_1000 = db.where((db['M0.l'] == 1000e-9) & (db['VDS'] < 0.9) & (db['IDS'] < 1e-4)).groupby('IDS')
    sftn_180 = (db_180.max()['Ftn']-db_180.min()['Ftn'])/(max(temp)-min(temp))/db_180.mean()['Ftn']
    sftn_360 = (db_360.max()['Ftn']-db_360.min()['Ftn'])/(max(temp)-min(temp))/db_360.mean()['Ftn']
    sftn_540 = (db_540.max()['Ftn']-db_540.min()['Ftn'])/(max(temp)-min(temp))/db_540.mean()['Ftn']
    sftn_720 = (db_720.max()['Ftn']-db_720.min()['Ftn'])/(max(temp)-min(temp))/db_720.mean()['Ftn']
    sftn_1000 = (db_1000.max()['Ftn']-db_1000.min()['Ftn'])/(max(temp)-min(temp))/db_1000.mean()['Ftn']
    plt.figure()
    plt.semilogx(ids, sftn_180*sFtScale, label="L = 180 nm")
    plt.semilogx(ids, sftn_360*sFtScale, label="L = 360 nm")
    plt.semilogx(ids, sftn_540*sFtScale, label="L = 540 nm")
    plt.semilogx(ids, sftn_720*sFtScale, label="L = 720 nm")
    plt.semilogx(ids, sftn_1000*sFtScale, label="L = 1 $\mu$m")
    plt.xlabel("Drain-Source Current [A]")
    plt.ylabel("$\\frac{\partial f_T}{f_T \partial T}$ [Hz.Hz$^{-1}$.T$^{-1}$]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./Vector/ft_nel_T.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
#
    # display the intrinsic gain of nel transistor
    Av_180 = db_180.apply(lambda r: np.divide(r['gmn_raw'],r['gdsn_raw']))
    Av_360 = db_360.apply(lambda r: np.divide(r['gmn_raw'],r['gdsn_raw']))
    Av_540 = db_540.apply(lambda r: np.divide(r['gmn_raw'],r['gdsn_raw']))
    Av_720 = db_720.apply(lambda r: np.divide(r['gmn_raw'],r['gdsn_raw']))
    Av_1000 = db_1000.apply(lambda r: np.divide(r['gmn_raw'],r['gdsn_raw']))
    sAv_180 = np.divide(Av_180.diff(), Av_180).values.reshape((len(ids), len(temp)))
    sAv_360 = np.divide(Av_360.diff(), Av_360).values.reshape((len(ids), len(temp)))
    sAv_540 = np.divide(Av_540.diff(), Av_540).values.reshape((len(ids), len(temp)))
    sAv_720 = np.divide(Av_720.diff(), Av_720).values.reshape((len(ids), len(temp)))
    sAv_1000 = np.divide(Av_1000.diff(), Av_1000).values.reshape((len(ids), len(temp)))
    Av_180 = Av_180.values.reshape(len(ids), len(temp))
    Av_360 = Av_360.values.reshape(len(ids), len(temp))
    Av_540 = Av_540.values.reshape(len(ids), len(temp))
    Av_720 = Av_720.values.reshape(len(ids), len(temp))
    Av_1000 = Av_1000.values.reshape(len(ids), len(temp))

    plt.figure()
    plt.semilogx(ids, [i[1] for i in Av_180], '-', alpha=1, label="T = {:.0f}$\degree$C".format(temp[1]))
    plt.semilogx(ids, [i[5] for i in Av_180], '-.', alpha=0.85, label="T = {:.0f}$\degree$C".format(temp[5]))
    plt.semilogx(ids, [i[7] for i in Av_180], '--', alpha=0.7, label="T = {:.0f}$\degree$C".format(temp[7]))
    plt.semilogx(ids, [i[-4] for i in Av_180], '--.', alpha=0.55, label="T = {:.0f}$\degree$C".format(temp[-4]))
    plt.semilogx(ids, [i[-1] for i in Av_180], '.-', alpha=0.4, label="T = {:.0f}$\degree$C".format(temp[-1]))
    plt.xlabel("Drain Current [A]")
    plt.ylabel("Intrinsic Gain [V/V]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./Vector/Av_nel_id.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
#
    plt.figure()
    plt.semilogx(ids, [i[1] for i in Av_720], '-', alpha=1, label="T = {:.0f}$\degree$C".format(temp[1]))
    plt.semilogx(ids, [i[5] for i in Av_720], '-.', alpha=0.85, label="T = {:.0f}$\degree$C".format(temp[5]))
    plt.semilogx(ids, [i[7] for i in Av_720], '--', alpha=0.7, label="T = {:.0f}$\degree$C".format(temp[7]))
    plt.semilogx(ids, [i[-4] for i in Av_720], '--.', alpha=0.55, label="T = {:.0f}$\degree$C".format(temp[-4]))
    plt.semilogx(ids, [i[-1] for i in Av_720], '.-', alpha=0.4, label="T = {:.0f}$\degree$C".format(temp[-1]))
    plt.xlabel("Drain Current [A]")
    plt.ylabel("Intrinsic Gain [V/V]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./Vector/Av_nel_id_720.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
    
    # display the intrinsic gain of nel transistor over temp
    plt.figure()
    i = 4 # or 9
    print(ids[i])
    def lissage(Lx, Ly, n):
        dLy = np.diff(Ly)
        ddLy = np.diff(Ly)
        ldLy = len(dLy)
        def crit(idx):
            print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(Lx[idx],Ly[idx], dLy[idx] if idx < ldLy else 0, ddLy[idx] if idx < ldLy-1 else 0))
            return (idx-1 >= ldLy) or (idx > -1 and dLy[idx-1] < n)
        # remove jump high
        ret  = [l for idx, l in enumerate(Ly) if crit(idx)]
        retx = [l for idx, l in enumerate(Lx) if crit(idx)]
        # remove jump low
        ret = medfilt(ret)
        #ret = np.cumsum(Ly, dtype=np.float)
        #ret[n:] = ret[n:] - ret[:-n]
        #print(Ly,ret[n-1:]/n)
        #return ret[n-1:]/n
        #return medfilt(Ly)
        return (retx, ret)
    x, y = lissage(temp[1:], sAv_180[i][1:], 1e-2)
    plt.plot(x, y, label="L = 180 nm")
    x, y = lissage(temp[1:], sAv_360[i][1:], 1e-2)
    plt.plot(x, y, label="L = 360 nm")
    x, y = lissage(temp[1:], sAv_540[i][1:], 1e-2)
    plt.plot(x, y, label="L = 720 nm")
    x, y = lissage(temp[1:], sAv_720[i][1:], 1e-2)
    plt.plot(x, y, label="L = 1 $\mu$m")
    plt.xlabel("Temperature [$\degree$C]")
    plt.ylabel(r"$\frac{\partial g_m/g_{ds}}{g_m/g_{ds}\partial T}$ [V/V/$\degree$C]")
    plt.legend()
    ax = plt.gca()
    # for 200 nA
    ax.annotate(r'L increasing', xy=(25, -2e-3), xycoords='data',
                xytext=(70, -7e-3), textcoords='data', color='k',
                va="center", ha="center", size=12,
                arrowprops=dict(
                    arrowstyle="<-",
                    facecolor='black',
                    lw=1))
    # for 10 uA
    #ax.annotate(r'L increasing', xy=(100, -9e-2), xycoords='data',
    #            xytext=(140, -10.5e-2), textcoords='data', color='k',
    #            va="center", ha="center", size=12,
    #            arrowprops=dict(
    #                arrowstyle="<-",
    #                facecolor='black',
    #                lw=1))
    plt.tight_layout()
    plt.savefig("./Vector/sAv_nel_T_{:.0e}.{}".format(ids[i],params['savefig.format']), bbox_inches="tight", dpi=150)
    # linearity as gm over vgs
    db_180 = db.where((db['M0.l'] == 180e-9) & (db['VDS'] == 0.4) & (db['IDS'] < 1e-4))
    db_360 = db.where((db['M0.l'] == 360e-9) & (db['VDS'] == 0.4) & (db['IDS'] < 1e-4))
    db_540 = db.where((db['M0.l'] == 540e-9) & (db['VDS'] == 0.4) & (db['IDS'] < 1e-4))
    db_720 = db.where((db['M0.l'] == 720e-9) & (db['VDS'] == 0.4) & (db['IDS'] < 1e-4))
    db_1000 = db.where((db['M0.l'] == 1000e-9) & (db['VDS'] == 0.4) & (db['IDS'] < 1e-4))
    vgsn_180  = pd.to_numeric(db_180['vgsn_raw'], errors='coerce').dropna().values.reshape(len(temp), -1)
    idn_180   = pd.to_numeric(db_180['idn'], errors='coerce').dropna().values.reshape(len(temp), -1)
    gmn_180   = pd.to_numeric(db_180['gmn_raw'], errors='coerce').dropna().values.reshape(len(temp), -1)
    vgsn_360  = pd.to_numeric(db_360['vgsn_raw'], errors='coerce').dropna().values.reshape(len(temp), -1)
    idn_360   = pd.to_numeric(db_360['idn'], errors='coerce').dropna().values.reshape(len(temp), -1)
    gmn_360   = pd.to_numeric(db_360['gmn_raw'], errors='coerce').dropna().values.reshape(len(temp), -1)
    vgsn_540  = pd.to_numeric(db_540['vgsn_raw'], errors='coerce').dropna().values.reshape(len(temp), -1)
    idn_540   = pd.to_numeric(db_540['idn'], errors='coerce').dropna().values.reshape(len(temp), -1)
    gmn_540   = pd.to_numeric(db_540['gmn_raw'], errors='coerce').dropna().values.reshape(len(temp), -1)
    vgsn_720  = pd.to_numeric(db_720['vgsn_raw'], errors='coerce').dropna().values.reshape(len(temp), -1)
    idn_720   = pd.to_numeric(db_720['idn'], errors='coerce').dropna().values.reshape(len(temp), -1)
    gmn_720   = pd.to_numeric(db_720['gmn_raw'], errors='coerce').dropna().values.reshape(len(temp), -1)
    vgsn_1000  = pd.to_numeric(db_1000['vgsn_raw'], errors='coerce').dropna().values.reshape(len(temp), -1)
    idn_1000   = pd.to_numeric(db_1000['idn'], errors='coerce').dropna().values.reshape(len(temp), -1)
    gmn_1000   = pd.to_numeric(db_1000['gmn_raw'], errors='coerce').dropna().values.reshape(len(temp), -1)
#
    plt.figure()
    plt.plot(vgsn_180[:][1], gmn_180[:][1]/idn_180[:][1], '-',alpha=1, label="T = {:.0f}$\degree$C".format(temp[1]))
    plt.plot(vgsn_180[:][3], gmn_180[:][3]/idn_180[:][3], '-.',alpha=0.85, label="T = {:.0f}$\degree$C".format(temp[3]))
    plt.plot(vgsn_180[:][6], gmn_180[:][6]/idn_180[:][6], '--',alpha=0.7, label="T = {:.0f}$\degree$C".format(temp[6]))
    plt.plot(vgsn_180[:][12], gmn_180[:][12]/idn_180[:][12], '--.',alpha=0.55, label="T = {:.0f}$\degree$C".format(temp[12]))
    plt.plot(vgsn_180[:][-1], gmn_180[:][-1]/idn_180[:][-1], '.-',alpha=0.4, label="T = {:.0f}$\degree$C".format(temp[-1]))
    plt.xlabel("Gate-Source Voltage [V]")
    plt.ylabel("$g_m/I_D$ [$\mu S/\mu A$]")   
    plt.legend()
    plt.tight_layout() 
    plt.savefig("./Vector/gm_nel_T.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
#
    rcParams['legend.fontsize'] = 8
    plt.figure()
    plt.plot(vgsn_180[:][6], gmn_180[:][6]/idn_180[:][6], '-', alpha=1, label="L = 180 nm")
    plt.plot(vgsn_360[:][6], gmn_360[:][6]/idn_360[:][6], '-.', alpha=0.85, label="L = 360 nm")
    plt.plot(vgsn_540[:][6], gmn_540[:][6]/idn_540[:][6], '--', alpha=0.7, label="L = 540 nm")
    plt.plot(vgsn_720[:][6], gmn_720[:][6]/idn_720[:][6], '--.', alpha=0.55, label="L = 720 nm")
    plt.plot(vgsn_1000[:][6], gmn_1000[:][6]/idn_1000[:][6], '.-', alpha=0.4, label="L = 1 $\mu$m")
    plt.plot(vgsn_180[:][-1], gmn_180[:][-1]/idn_180[:][-1], '-', alpha=1, label="L = 180 nm")
    plt.plot(vgsn_360[:][-1], gmn_360[:][-1]/idn_360[:][-1], '-.', alpha=0.85, label="L = 360 nm")
    plt.plot(vgsn_540[:][-1], gmn_540[:][-1]/idn_540[:][-1], '--', alpha=0.7, label="L = 540 nm")
    plt.plot(vgsn_720[:][-1], gmn_720[:][-1]/idn_720[:][-1], '--.', alpha=0.55, label="L = 720 nm")
    plt.plot(vgsn_1000[:][-1], gmn_1000[:][-1]/idn_1000[:][-1], '.-', alpha=0.4, label="L = 1 $\mu$m")
    plt.xlabel("Gate-Source Voltage [V]")
    plt.ylabel("$g_m/I_D$ [$\mu S/\mu A$]")   
    plt.axis([0, 2, 0, 25])
    ax = plt.gca()
    ax.annotate(r'T = 27$\degree $C', xy=(0.7, 17), xycoords='data',
                xytext=(1, 20), textcoords='data', color='k',
                va="center", ha="center", size=12,
                arrowprops=dict(
                    arrowstyle="->",
                    facecolor='black',
                    lw=1))
    ax.annotate(r'T = 175$\degree $C', xy=(0.5, 12), xycoords='data',
                xytext=(0.3, 8), textcoords='data', color='k',
                va="center", ha="center", size=12,
                arrowprops=dict(
                    arrowstyle="->",
                    facecolor='black',
                    lw=1))
    plt.legend()#loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout() 
    plt.savefig("./Vector/gm_nel_L.{}".format(params['savefig.format']), bbox_inches="tight", dpi=150)
    # display Ron over temp for L
    gdsn_180  = pd.to_numeric(db_180['gdsn_raw'], errors='coerce').dropna().values.reshape(len(temp), -1)
    gdsn_360  = pd.to_numeric(db_360['gdsn_raw'], errors='coerce').dropna().values.reshape(len(temp), -1)
    gdsn_540  = pd.to_numeric(db_540['gdsn_raw'], errors='coerce').dropna().values.reshape(len(temp), -1)
    gdsn_720  = pd.to_numeric(db_720['gdsn_raw'], errors='coerce').dropna().values.reshape(len(temp), -1)
    plt.figure()
    plt.semilogx(idn_180[:][1], 1/gdsn_180[:][1], '-',alpha=1, label="T = {:.0f}$\degree$C".format(temp[1]))
    plt.semilogx(idn_180[:][3], 1/gdsn_180[:][3], '-.',alpha=0.85, label="T = {:.0f}$\degree$C".format(temp[3]))
    plt.semilogx(idn_180[:][6], 1/gdsn_180[:][6], '--',alpha=0.7, label="T = {:.0f}$\degree$C".format(temp[6]))
    plt.semilogx(idn_180[:][12], 1/gdsn_180[:][12], '--.',alpha=0.55, label="T = {:.0f}$\degree$C".format(temp[12]))
    plt.semilogx(idn_180[:][-1], 1/gdsn_180[:][-1], '.-',alpha=0.4, label="T = {:.0f}$\degree$C".format(temp[-1]))
    plt.tight_layout()
    plt.show()