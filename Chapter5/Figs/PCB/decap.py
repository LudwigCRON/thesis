#!/usr/bin/env python3

from matplotlib import cm
from matplotlib import rcParams

params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 10,
   'axes.titlesize': 8,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'savefig.format': 'eps',
   'figure.dpi': 200,
   'figure.figsize': [4, 3],
   'mathtext.fontset': 'stix',
   'font.family':'STIXGeneral',
   'axes.formatter.useoffset': False,
   'axes.color_cycle': ['#AA3939', '#AA9739', '#403075', '#2D882D', '#246B61', '#AA6839', '#333333']
}
rcParams.update(params)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# zoom effect
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

if __name__ == "__main__":
  db = pd.read_csv("./decap.txt", delimiter="\t")
  # display header
  for i in db:
    print(i)
  
  # load series
  time = db["time"]*1e9
  Iload = db["I(I1)"]*1e3
  ErrorIC = db["V(Vrefp,Vref)"]*1e3
  ErrorDecap = (db["V(vdecap)"]-db["V(vref)"])*1e6
  
  # print the voltage error and the load
  plt.figure()
  h1 = plt.subplot(3,1,1)
  plt.plot(time, Iload, color=params["axes.color_cycle"][0])
  #plt.xlabel("Time [ns]")
  plt.ylabel("Current [mA]")
  h1.get_yaxis().set_label_coords(-0.1,0.5)
  plt.title("Current Load to charge the DAC",fontweight="bold")
  h2 = plt.subplot(3,1,2, sharex=h1)
  plt.plot(time, ErrorIC, color=params["axes.color_cycle"][2])
  #plt.xlabel("Time [ns]")
  plt.ylabel("Voltage [mV]")
  h2.get_yaxis().set_label_coords(-0.1,0.5)
  plt.title("Error on the reference voltage inside the IC",fontweight="bold")
  # zoomed plot
  axins = zoomed_inset_axes(h2, 1000, loc=4)
  axins.plot([0, 20], [0, 0], 'k--')
  axins.plot(time, ErrorIC, color=params["axes.color_cycle"][2])
  x1, x2, y1, y2 = 15.995, 16, -0.25, 0.01 # specify the limits
  axins.set_xlim(x1, x2) # apply the x-limits
  axins.set_ylim(y1, y2) # apply the y-limits
  plt.xticks(visible=False)
  plt.yticks(visible=False)
  mark_inset(h2, axins, loc1=2, loc2=1, fc="none", ec="0.5")
  plt.annotate(
    s="",
    xy=(0.2, 1.1),
    xytext=(0, -15),
    xycoords="axes fraction",
    textcoords="offset pixels",
    arrowprops= dict(
      arrowstyle= '<->'
    )
  )
  plt.annotate(
    s=r"250 $\mu$V",
    xy=(0.25, 0.3),
    xytext=(0, 0),
    xycoords="axes fraction",
    textcoords="offset pixels"
  )
  h3 = plt.subplot(3,1,3, sharex=h1)
  plt.plot(time, ErrorDecap, color=params["axes.color_cycle"][3])
  plt.xlabel("Time [ns]")
  plt.ylabel("Voltage [$\mu$V]")
  h3.get_yaxis().set_label_coords(-0.1,0.5)
  plt.title("Error on the reference voltage after the X7R decoupling capacitor",fontweight="bold")
  plt.xlim(0, 20)
  plt.tight_layout()
  plt.savefig(r"./decap-reference-sar.{}".format(params["savefig.format"]), dpi=params["figure.dpi"])
  plt.show()

  
