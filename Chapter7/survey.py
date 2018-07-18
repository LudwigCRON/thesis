#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import gamma
from matplotlib import rcParams

params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
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

def only_numerics(seq):
    seq_type= type(str(seq))
    ans = seq_type().join(filter(lambda e: seq_type.isdigit(e) or e=='.', str(seq)))
    return float(ans) if len(ans) > 0 else float('nan')

if __name__ == "__main__":
    dbISSCC = pd.read_excel("./ADCSurvey.xls", sheet_name="ISSCC")
    dbISSCC = dbISSCC.where(dbISSCC['TECHNOLOGY'].apply(only_numerics) >= 0.065)
    dbVLSI = pd.read_excel("./ADCSurvey.xls", sheet_name="VLSI")
    dbVLSI = dbVLSI.where(dbISSCC['TECHNOLOGY'].apply(only_numerics) >= 0.065)
    techno  = sorted(list(set(dbISSCC['TECHNOLOGY'].apply(only_numerics).dropna())))
    ampISSCC= dbISSCC.dropna(thresh=12).apply(
        lambda r: [r['YEAR'], r['TECHNOLOGY'], r['TYPE'], [] if str(r['Amplifier Type']).lower() == "nan" else str(r['Amplifier Type']).replace('/', ' ').lower().split()]
    , axis=1)
    ampVLSI = dbVLSI.dropna(thresh=12).apply(
        lambda r: [r['YEAR'], r['TECHNOLOGY'], r['TYPE'], [] if str(r['Amplifier Type']).lower() == "nan" else str(r['Amplifier Type']).replace('/', ' ').lower().split()]
    , axis=1)
    print([h for h in dbISSCC])
    print([h for h in dbVLSI])
    tmp   = []
    years = []
    for a in ampISSCC:
        years.append(a[0])
        tmp.extend(a[-1])
    for a in ampVLSI:
        years.append(a[0])
        tmp.extend(a[-1])
    years = sorted(list(set(years)))
    AmpType = list(set(tmp))
    NbBins = len(AmpType)
    print(NbBins, AmpType)

    hist = np.zeros((NbBins, len(years)))
    for a in ampISSCC:
        year = years.index(a[0])
        for w in a[-1]:
            idx = AmpType.index(w)
            hist[idx][year] += 1
    for a in ampVLSI:
        year = years.index(a[0])
        for w in a[-1]:
            idx = AmpType.index(w)
            hist[idx][year] += 1

    FCAIdx  = AmpType.index('folded-cascode')
    GBIdx   = AmpType.index('gain-boosted')
    TSIdx   = AmpType.index('two-stage')
    TeleIdx = AmpType.index('telescopic')

    plt.figure()
    ax = plt.gca()
    #plt.stackplot(years, [hist[GBIdx], hist[FCAIdx], hist[TeleIdx], hist[TSIdx]])
    #for k in range(NbBins):
    #    plt.plot(years, Repartition[k], label=AmpType[k])
    ax.fill_between(years, 0, hist[GBIdx]  , label=AmpType[GBIdx], alpha=0.75)#  , ls='-')
    ax.fill_between(years, hist[GBIdx], hist[GBIdx]+hist[FCAIdx] , label=AmpType[FCAIdx], alpha=0.75)# , ls='--')
    ax.fill_between(years, hist[GBIdx]+hist[FCAIdx], hist[GBIdx]+hist[FCAIdx]+hist[TeleIdx], label=AmpType[TeleIdx], alpha=0.75)#, ls='-.')
    ax.fill_between(years, hist[GBIdx]+hist[FCAIdx]+hist[TeleIdx], hist[GBIdx]+hist[FCAIdx]+hist[TeleIdx]+hist[TSIdx]  , label=AmpType[TSIdx], alpha=0.75)#  , ls=':')
    plt.xlabel('Year')
    plt.ylabel('Stacked Count')
    plt.xticks(range(1997, 2018, 5))
    plt.yticks(range(0, 20, 4))
    plt.axis([1997, 2017, 0, 18])
    l = plt.legend(loc='upper right')#, bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("./Figs/amplifier_repartition."+params['savefig.format'], format=params['savefig.format'], dpi=params['figure.dpi'], bbox_inches="tight")
    plt.show()
