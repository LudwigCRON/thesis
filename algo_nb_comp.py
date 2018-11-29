#!/usr/bin/env python3

import numpy as np

N = 2**np.linspace(3, 8, 6, dtype=int)
OSR = np.linspace(1, 8, 8, dtype=int)

print(" ",N)

def f(M, levels):
	return [max(1, int(c)) for c in np.floor(np.divide(levels, 2**(M-1)))]

ans = []
for M in OSR:
	ans.append(f(M, N))
	print(M, f(M, N))
