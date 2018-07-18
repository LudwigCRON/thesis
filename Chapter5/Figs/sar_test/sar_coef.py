#!/usr/bin/env python3

import re
import numpy as np
from matplotlib import cm
from matplotlib import rcParams

params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 8,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': True,
   'savefig.format': 'pgf',
   'figure.dpi': 200,
   'figure.figsize': [4, 3],
   'mathtext.fontset': 'stix',
   'font.family':'STIXGeneral',
   "pgf.texsystem": "pdflatex", 
   "pgf.preamble": [
     r"\usepackage{gensymb}",
     r"\usepackage[utf8x]{inputenc}", # use utf8 fonts because your computer can handle it :)
     r"\usepackage[T1]{fontenc}", # plots will be generated using this preamble
   ]
}

rcParams.update(params)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == "__main__":
  data = [re.sub(' +', ' ', line.rstrip('\n')).strip().split(' ') for line in open("sar_coef.txt")]
  ref = list(map(float, data[0][1:]))
  db_doe05_13   = []
  temp_doe05_13 = []
  db_doe05_11   = []
  temp_doe05_11 = []
  off_doe05_11  = []
  off_doe05_13  = []
  for i in range(2, 13):
    db_doe05_13.append(
      list(map(float, data[i][1:-1]))
    )
    temp_doe05_13.append(float(data[i][0]))
    off_doe05_13.append(data[i][-1])
  for i in range(15, 26):
    db_doe05_11.append(
      list(map(float, data[i][1:-1]))
    )
    temp_doe05_11.append(float(data[i][0]))
    off_doe05_11.append(data[i][-1])

  print(ref)
  print(temp_doe05_11)
  print(db_doe05_11)
  print(temp_doe05_13)
  print(db_doe05_13)
  
  var_db_doe05_11 = []
  for coef in db_doe05_11:
    var_db_doe05_11.append(np.divide(np.subtract(coef, ref), ref[-1]))
  var_db_doe05_11 = np.transpose(var_db_doe05_11)
  
  var_db_doe05_13 = []
  for coef in db_doe05_13:
    var_db_doe05_13.append(np.divide(np.subtract(coef, ref), ref[-1]))
  var_db_doe05_13 = np.transpose(var_db_doe05_13)
  
  plt.figure()
  colors = plt.cm.Purples(np.linspace(0,1,len(var_db_doe05_11[0])+5))
  for i in range(len(var_db_doe05_11)):
    plt.plot(temp_doe05_11, var_db_doe05_11[i], marker=".", color=colors[i+5], label="code[{}]".format(i))
  plt.xlabel("Temperature [$\degree$C]")
  plt.ylabel("Weight Error/ideal LSB Weight")
  plt.legend(ncol=2)
  plt.tight_layout()
  plt.savefig("./sar_coef_doe05_11.{}".format(params["savefig.format"]), dpi=params["figure.dpi"])
  
  plt.figure()
  ax = plt.gca()
  colors = plt.cm.Purples(np.linspace(0,1,len(var_db_doe05_13[0])+5))
  for i in range(len(var_db_doe05_13)):
    plt.plot(temp_doe05_13, var_db_doe05_13[i], marker=".", color=colors[i+5], label="code[{}]".format(i))
  plt.xlabel("Temperature [$\degree$C]")
  plt.ylabel("Weight Error/ideal LSB Weight")
  plt.legend(ncol=2)
  plt.tight_layout()
  plt.axvline(x=185, linestyle='--', color='r')
  ax.annotate("bonding shearing", xy=(90, -0.5), xytext=(90, -0.5))
  ax.annotate("for $T > 180\degree$C", xy=(110, -0.55), xytext=(110, -0.55))
  style="Simple,tail_width=0.5,head_width=4,head_length=8"
  kw = dict(arrowstyle=style, color="k")
  ax.add_patch(patches.FancyArrowPatch((120,-0.57), (180,-0.57),connectionstyle="arc3,rad=.5", **kw))
  plt.savefig("./sar_coef_doe05_13.{}".format(params["savefig.format"]), dpi=params["figure.dpi"])
 
  plt.show()
