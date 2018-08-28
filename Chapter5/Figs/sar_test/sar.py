#!/usr/bin/env python3

import numpy as np

def sar_split_ratio(cap, split_pos):
  ptmp = np.sum(cap[split_pos+1:])*np.sum(cap[:split_pos+1])
  r = 1-cap[split_pos]**2/ptmp
  return np.concatenate((
    np.divide(
      cap[:split_pos],
      np.sum(cap[:split_pos+1])*r
    ),
      np.multiply(cap[split_pos], cap[split_pos+1:-1])/ptmp/r
    ), 0)

def sar_dac(bit, ratio, vref):
  return vref * ratio * (2 * bit - 1)

def sar_est(bits, ratio, vref):
  return vref * (np.dot(bits, ratio) * 2 - 1)

def sar(vin, ratio, vref, quad):
  """
  @param osr  : the number of oversampling ratio
  @param ratio: ratio used for the comparison
  @param vref : the reference voltage
  @return (vres, bits)
  """
  bits = []
  vres = 0
  for i, r in enumerate(ratio):
    if i < 2:
      bits.append(quad[i])
    else:
      bits.append(1. if vin > vres else 0.)
    vres = vres + sar_dac(bits[-1], r, vref)
  return (0, bits)