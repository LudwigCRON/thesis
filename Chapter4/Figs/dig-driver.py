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
   'text.usetex': False,
   'savefig.format': 'pdf',
   'figure.dpi': 200,
   'figure.figsize': [4, 3],
   'mathtext.fontset': 'stix',
   'font.family':'STIXGeneral',
   #'axes.color_cycle': ['#AA3939', '#AA9739', '#403075', '#2D882D', '#246B61', '#AA6839', '#333333'],
   "pgf.texsystem": "pdflatex", 
   "pgf.preamble": [
     r"\usepackage{gensymb}",
     r"\usepackage[utf8x]{inputenc}", # use utf8 fonts because your computer can handle it :)
     r"\usepackage[T1]{fontenc}", # plots will be generated using this preamble
   ]
}
rcParams.update(params)

import matplotlib.pyplot as plt

DEBUG = True
k = 1
PATH = ["~/Documents/GoogleDrive/Thesis/Manuscrit_v1/Chapter4/Figs/dig-driver-tm-temp.csv", "~/Documents/GoogleDrive/Thesis/Manuscrit_v1/Chapter4/Figs/dig-driver-tm-temp-with-tgate.csv"]

def format_corner(s):
    t = s.lower().strip()
    if t == "nom" or t == "tm":
        return "TT"
    elif t == "wp":
        return "FF"
    elif t == "ws":
        return "SS"
    elif t == "wo":
        return "FS"
    else:
        return "SF"

def format_header(h):
    t = h.split("=")
    return r"{:.0f} $\degree$C".format(float(t[1].split(")")[0]))

def isFloat(f):
    try:
        float(f)
        return True
    except ValueError:
        return False

def findCrossing(x1, y1, x2, y2):
    N = min(len(x1), len(x2), len(y1), len(y2))
    for i in range(N):
        if x1[i] > 0.4 and x1[i] < 0.8:
            if (y1[i-1]<y2[i] and y1[i+1]>=y2[i]) or (y2[i-1]<y1[i] and y2[i+1]>=y1[i]):
                return (i, x1[i], y1[i], x2[i], y2[i])
    return None

if __name__ == "__main__":
    # db load
    db = pd.read_csv(PATH[k])
    db.convert_objects(convert_numeric=True)
    headers = [header for header in db]
    print(headers)
    
    # display all curves
    plt.figure()
    Lmid = int(len(headers)/2)
    intersect = []
    for i in range(0, Lmid, 2):
        F1p_db = np.array(db[headers[i:i+2]])
        F1p_db = list(map(lambda r: [float(a) for a in r if isFloat(a)], F1p_db))
        F1m_db = np.array(db[headers[Lmid+i:Lmid+i+2]])
        F1m_db = list(map(lambda r: [float(a) for a in r if isFloat(a)], F1m_db))
        xp = [r[0]*1e9 for r in F1p_db if len(r) > 0]
        yp = [r[1] for r in F1p_db if len(r) > 0]
        p = plt.plot(xp, yp, label=format_header(headers[i+1]))
        xm = [r[0]*1e9 for r in F1m_db if len(r) > 0]
        ym = [r[1] for r in F1m_db if len(r) > 0]
        plt.plot(xm, ym, label=format_header(headers[Lmid+i+1]), color=p[0].get_color())
        # get intersection
        intersect.append(findCrossing(xp, yp, xm, ym))
    
    xi = [(i[1]+i[3])/2 for i in intersect if i is not None]
    yi = [(i[2]+i[4])/2 for i in intersect if i is not None]
    xiMinMax = [min(xi), max(xi)]
    yiMinMax = [min(yi), max(yi)]
    plt.plot(xi, yi, 'k-', lw=4)
    
    ax = plt.gca()
    ax.annotate("{:.0f} mV".format(yiMinMax[0]*1000), xy=(xiMinMax[0], yiMinMax[0]), xytext=(xiMinMax[0]+0.105, yiMinMax[0]),
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=4),
            horizontalalignment='center',
            verticalalignment='center',
            )
    ax.annotate("{:.0f} mV".format(yiMinMax[1]*1000), xy=(xiMinMax[1], yiMinMax[1]), xytext=(xiMinMax[1]+0.105, yiMinMax[1]),
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=4),
            horizontalalignment='center',
            verticalalignment='center',
            )
    
    plt.yticks([0, 0.45, 0.9, 1.35, 1.8])
    plt.axis([0.35, 0.85, -0.1, 2])
    plt.xlabel("Time [ns]")
    plt.ylabel("Voltage [V]")
    plt.tight_layout()
    
    plt.savefig("./crossing-driver-with-tgate."+params['savefig.format'], format=params['savefig.format'], dpi=params['figure.dpi'], bbox_inches="tight")

    plt.show()
