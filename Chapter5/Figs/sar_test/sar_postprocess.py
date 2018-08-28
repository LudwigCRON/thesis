#!/usr/bin/env python3

import os
import sys
import mmap
from collections import Counter
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sar

def get_opt(keys, num, default):
    """
    extract arguments from the command line and give the expected value
    in an array if the num > 1, or the element otherwise.
    If none of the keys if found, return the default value
    """
    for key in keys:
        if key in sys.argv:
            idx = sys.argv.index(key)
            if idx+num < len(sys.argv) and num > 1:
                return sys.argv[idx+1:idx+num]
            elif num == 1:
                return sys.argv[idx+1]
    return default

def parse(filepath, version, algoDec):
    """
    parse the output file from test
    if version is 1 (test1.exe/test1b.exe) the quadrant is changed based on
    thresholds estimation. While for version == 2, a ramp is applied for each
    quadrant. Therefore, in version 1 data return are a simple 2d array whereas
    in version 2 data is a 3d array (2d array for each quadrant)
    """
    # read the file line by line
    f = open(filepath, 'r+')
    buf = mmap.mmap(f.fileno(), 0)
    readline = buf.readline
    # variable allocation
    data = []
    sepOnce = False
    quad = 0
    Nsample = 0
    counter = 0
    r = [0]*14
    db = []
    bits = []
    l = " "
    # if the strategy is select only one
    pos = int(get_opt(["-pos"],1,-1))
    while not l == "":
        l = str(readline(), 'utf-8')
        if not l[0:2] == '--':
            sepOnce = False
            if '=' in l:
                key, val = l.split('=', 2)
                if key.strip() == "Voltage":
                    r[0] = float(val)
                elif key.strip() == "Resetn":
                    r[1] = int(val)
                elif key.strip() == "Start":
                    r[2] = int(val)
                elif key.strip() == "Quadrant":
                    r[3] = int(val.strip()[0])
                elif key.strip() == "Cycle6":
                    r[4] = int(val)
                elif key.strip() == "Error":
                    r[5] = int(val)
                elif key.strip()[:7] == "Results" and len(key.strip()) > 7:
                    bits.append([float(b) for b in val.strip()])
                elif key.strip() == "Warning":
                    pass
                else:
                    r[6:13] = [float(b) for b in val.strip()]
        else:
            # new sample
            if not sepOnce:
                Nsample += 1
                counter += 1
                db.append(r)
                r = [0]*14
                sepOnce = True
                # multisample per voltage strategy
                if len(bits) > 0:
                    bi = np.sum(bits, axis=0)
                    #  average
                    if algoDec == 2:
                        r[6:13] = np.divide(bi, len(bits))
                    # majority vote
                    elif algoDec == 1:
                        c = Counter(list(map(bits2int, bits)))
                        value, count = c.most_common()[0]
                        r[6:13] = int2bits(value, 7)
                    # keep only one results
                    else:
                        r[6:13] = bits[pos]
                bits = []
            # new quad test
            else:
                quad += 1
                counter = 0
                data.append(db)
                db = []
    data.append(db)
    f.close()
    return (data, Nsample)

# get weight pondering bits given by a conversion performed
def get_weight(cycle):
    if cycle == 5:
            cap = [6, 3, 2, 3, 4, 2, 1, 1] # capacitor weight
    elif cycle == 6:
            cap = [6, 3, 2, 3, 4, 2, 1, 0.5, 0.5] # capacitor weight
    idx_att = 3                        # index of the split capacitor
    scale   = (sum(cap[idx_att:])*sum(cap[0:idx_att+1])-cap[idx_att]**2)/(sum(cap[0:idx_att+1])*sum(cap[idx_att:])) # scale for the ratio of each bit
    ratio   = [cap[i]/sum(cap[0:idx_att+1]) if i <= idx_att else cap[i]*cap[idx_att]/sum(cap[0:idx_att+1])/sum(cap[idx_att:]) for i in range(len(cap))]
    ratio   = [r/scale for r in ratio]
    del ratio[idx_att]
    if cycle == 6:
        del ratio[-1]
    else:
        ratio[-1] = 0
    return np.multiply(ratio, 2/sum(ratio))

# vout estimation from binary output
def estimate(db, weight):
    return [np.dot(weight, r[6:13]) for r in db]

# histogram based dnl calculation
def DNL(vin, vout, Nbits):
    # generate bins limits
    bins = np.linspace(min(vin),max(vin),2**Nbits)
    # fill an histogram
    hist, bins_edges = np.histogram(vout, bins=bins)
    # calculate the ideal number per step
    VLSB = len(vin)/2**Nbits
    return (bins_edges[1:-1], np.divide(hist[1:-1],VLSB)-1)

def DNL2(vin, vout, weight):
    # ideally convert the vin
    bits = []
    w = list(weight)
    for v in vin:
        # generate the quadrant
        quad = [0, 0] if v <= -0.5 else \
               [0, 1] if v > -0.5 and v <= 0 else \
               [1, 0] if v > 0 and v <= 0.5 else \
               [1, 1]
        # convert
        vres, bi = sar.sar(v, w, 0.5, quad)
        bits.append(bi)
    # estimate
    videal = sar.sar_est(bits, w, 1)
    videal = [(v-1)/2 for v in videal]
    videal = signal.medfilt(videal)
    bins = list(sorted(set(videal)))
    h_ideal, codes_ideal = np.histogram(videal, bins)
    # extract gain and offset error from vout
    a, b = gain_offset(vin, videal)
    g, o = gain_offset(vin, vout)
    vo = [(v-o)*a/g+b for v in vout]
    h, codes = np.histogram(vo, bins)
    dnl = np.divide(h, h_ideal)-1
    # nice plot of enhance dnl
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3, 9), dpi=100)
    # transfer function
    ax = plt.subplot(311)
    plt.plot(vin, videal, 'b')
    plt.plot(vin, vo, 'r')
    plt.xlabel("$\Delta V_{in}$ [V]")
    plt.ylabel("Voltage [V]")
    plt.title("Transfer Functions (blue) ideal/ (red) circuit", \
     fontsize=6, fontweight="bold")
    # histogram
    plt.subplot(312, sharex=ax)
    plt.plot(codes_ideal[2:-1], h_ideal[1:-1], 'b')
    plt.plot(codes[2:-1], h[1:-1], 'r')
    plt.xlabel("$\Delta V_{in}$ [V]")
    plt.ylabel("Count []")
    plt.title("Histogram of Codes (blue) ideal/ (red) circuit", \
     fontsize=6, fontweight="bold")
    # dnl
    plt.subplot(313, sharex=ax)
    plt.plot(codes[2:], dnl[1:])
    plt.xlabel("$\Delta V_{in}$ [V]")
    plt.ylabel("[Ideal LSB]")
    plt.title("DNL", \
     fontsize=6, fontweight="bold")
    fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
    plt.savefig('{}-fsr.eps'.format(outputName), bbox_inches='tight')
    return (bins[1:-1], dnl[1:-1])

# basic INL calculation
# cumulative sum of the DNL
def INL(vin, vout, Nbits):
    bins = np.linspace(min(vin),max(vin),2**Nbits)
    hist, bins_edges = np.histogram(vout, bins=bins)
    VLSB = len(vin)/2**Nbits
    return (bins_edges[1:-1], np.cumsum(np.divide(hist[1:-1],VLSB)-1))

# convert a binary word into an integer
def bits2int(bits):
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out

# convert int to binary word
def int2bits(i, length):
    fmtStr = "{0:0%db}" % (length)
    bits = fmtStr.format(i)
    return [int(k) for k in bits]

# calculate the limits of each strips
def stripDelimiter(db, weight, version):
    maxi = [-1]*len(db)
    mini = [-1]*len(db)
    # function to apply for min overlap
    def firstValueAbove(arr, m):
        for i, val in enumerate(arr):
            if i > 1 and val > m:
                return i
        return -1
    # sweep all quad tests
    prevStep = 0
    lastMax = 0
    sw = sum(weight)
    for quad in range(len(db)):
        # if there is data detect overlap
        if len(db[quad]) > 0:
            # filter any spikes to be sure of range
            codes = list(signal.medfilt(estimate(db[quad], weight)))
            step = np.mean(np.diff(codes))
            # detect which quad
            i, a = min(codes), max(codes)
            q = 0 if i < -0.75 else \
                1 if i < -0.25 else \
                2 if i < 0.25 else 3
            # extract limits
            if step > 0:
                maxi[quad] = codes.index(a)
                mini[quad] = firstValueAbove(codes, \
                   min(codes) if (version == 2 and prevStep <= 0 and step > 0) \
                              or version > 2 \
                              else lastMax)
                lastMax = a
            else:
                negcodes = list(np.multiply(codes, -1))
                maxi[quad] = negcodes.index(max(negcodes))
                mini[quad] = firstValueAbove(negcodes, -lastMax if version == 2 else min(negcodes))
                lastMax = i
            prevStep = step
    return (mini, maxi)

# define best-fit INL
def best_fit_INL(vin, vout, Nbits):
    # gain and offset for the end point
    a = (max(vout)-min(vout))/(max(vin)-min(vin))
    b = min(vout)-a*min(vin)
    # get middle of steps
    steps = [vout[i]-vout[i-1]>(max(vout)-min(vout))/2**(Nbits+1) if i > 1 else 0 for i in range(len(vout))]
    steps = [i-1 for i in range(len(steps)) if steps[i]]
    support = [(steps[i-1]+steps[i])/2 for i in range(len(steps))]
    support = [steps[0]/2]+support
    support = support + [int((len(vout)+steps[-1])/2)]
    vout_middle = [vout[int(s)-1] if s%1==0 else (vout[int(s)]+vout[int(s)-1])/2 for s in support]
    vin_middle = [b+a*vin[int(s)] if s%1==0 else b+a*(vin[int(s)]+vin[int(s)-1])/2 for s in support]
    error = [(vout_middle[i]-vin_middle[i])*2**Nbits/(max(vout)-min(vout)) for i in range(len(vout_middle))]
    return (vin_middle[1:-1], error[2:-1])

# return the code to which the output voltage voltage is 0
def offset(db, est, w, version):
    idx = 0
    quad = 0
    if len(db) == 0:
        return (-1, -1, None)
    for quad in range(len(db)):
        vout = est(db[quad], w)
        for v in vout:
            if v > 0:
                return (quad, idx, db[quad][idx])
            idx += 1
        idx = 0
    return (-1, -1, None)

# linear fit of the reconstructed output voltage
def gain_offset(vin, vout):
    A = np.vstack([vin, np.ones(len(vin))]).T
    g, o = np.linalg.lstsq(A, vout)[0]
    return (g, o)

def adjustOverlap(db, maxi, mini, version):
    ans = []
    # no overlap for test of version 1
    if version == 1:
        return (db, maxi, mini)
    # otherwise there is overlap
    if version == 2:
        # keep the part of the previous quad
        ans = [db[0][mini[0]:maxi[0]]]
        for quad in range(1, len(db)):
            if len(db[quad][mini[quad]:maxi[quad]]) > 0:
                codes = [r[0] for r in db[quad][mini[quad]:maxi[quad]]]
                step = np.mean(np.diff(codes))
                prevV = db[quad-1][maxi[quad-1]][0]
                ans.append(
                    list(
                        filter(lambda r: r[0] > prevV and r[0] < 1 \
                        if step > 0 else \
                        r[0] <= prevV and r[0] > -1 , db[quad][mini[quad]:maxi[quad]])
                    )
                )
            else:
                ans.append([])
    else:
        # split based on ideal quadrant change
        for quad in range(len(db)):
            if len(db[quad][mini[quad]:maxi[quad]]) > 0:
                codes = [r[0] for r in db[quad][mini[quad]:maxi[quad]]]
                step = max(np.diff(codes))
                # detect which quad it is
                i, a = min(codes), max(codes)
                q = 0 if i < -0.75 else \
                    1 if i < -0.25 else \
                    2 if i < 0.25 else 3
                # find limits
                limit_min = q*0.5-1
                limit_max = q*0.5-0.5
                ans.append(
                    list(
                        filter(lambda r: r[0] > limit_min and r[0] <= limit_max, \
                        db[quad][mini[quad]:maxi[quad]])
                    )
                )
            else:
                ans.append([])
        pass 
    return ans


if __name__ == "__main__":
    version = 2 if "-v2" in sys.argv else 3 if "-v3" in sys.argv else 1
    algoDec = 1 if "-mj" in sys.argv else 2 if "-avg" in sys.argv else \
              3 if "-med" in sys.argv else 0
    InputFile = get_opt(["-i", "--input"], 1, None)
    if not InputFile is None:
        outputName = "./graph/{}".format(os.path.basename(InputFile).split('.')[0])
        db, Nsamples = parse(InputFile, version, algoDec)
        cycle = 6 if db[0][10][4] > 0 else 5
        Nbits = cycle
        print("{} samples detected".format(Nsamples))
        print("{} clock cycles per sample".format(cycle))
        # calculate the ideal weight
        weight = get_weight(cycle)
        print(weight, sum(weight))
        vin = []
        for i in range(len(db)):
            vin.extend([r[0] for r in db[i]])
        plt.figure(figsize=(4,3))
        plt.plot(vin)
        plt.xlabel("Sample index")
        plt.ylabel("Measured Input Voltage [V]")
        plt.tight_layout()
        plt.savefig('{}-vin.eps'.format(outputName), bbox_inches='tight')
        plt.show()
        # estimate the transfert function
        # first compute strips limits
        mini, maxi = stripDelimiter(db, weight, version)
        dba = adjustOverlap(db, maxi, mini, version)
        if version == 1:
            vin  = [r[0] for r in dba[0]]
            vout = estimate(dba[0], weight)
        else:
            vin = []
            vout = []
            for i in range(len(dba)):
                vin.extend([r[0] for r in dba[i]])
                vout.extend(estimate(dba[i], weight))
            vout = [v-1 for v in vout]
            if algoDec == 3:
                vout = signal.medfilt(vout)
        #vout = signal.medfilt(vout)
        if version == 1:
            plt.figure(figsize=(6,3), dpi=150)
            plt.step(np.arange(len(vout)), vout, linewidth=2.)
            plt.tight_layout()
            plt.show()
        else:
            plt.figure(figsize=(4,8), dpi=100)
            for i in range(len(db)):
                if len(db[i]) > 0:
                    plt.subplot(len(db),1,i+1)
                    plt.step([r[0] for r in db[i]], \
                        np.subtract(estimate(db[i], weight), 1), linewidth=2.)
            plt.tight_layout()
            plt.savefig('{}-range.eps'.format(outputName), bbox_inches='tight')
            plt.show()
        plt.figure(figsize=(4,3))
        plt.plot(vin, vout)
        plt.xlabel("Input Voltage [V]")
        plt.ylabel("Estimated Output Voltage [V]")
        plt.tight_layout()
        plt.savefig('{}-tf.eps'.format(outputName), bbox_inches='tight')
        plt.show()
        # estimate the DNL
        codes, dnl = DNL(vin, vout, Nbits)
        c,d = DNL2(vin, vout, weight)
        if not "--stop" in sys.argv:
            # estimate the INL
            #inl = INL(dnl)
            #support = codes
            support, inl = best_fit_INL(vin, vout, Nbits)
            # calculated correction of bits weight
            bits_offset = []
            for i in range(len(dba)):
                bits_offset.extend([np.concatenate((line[6:13], [1])) for line in dba[i]])
            corr_weight = np.linalg.lstsq(bits_offset, vin)[0]
            corr_weight = corr_weight/sum(corr_weight)
            print(corr_weight, sum(corr_weight))        
            corr_vout = []
            for i in range(len(dba)):
                corr_vout.extend(estimate(dba[i], corr_weight[:-1]))
            corr_vout = [c+corr_weight[-1] for c in corr_vout]
            if algoDec == 3:
                corr_vout = signal.medfilt(corr_vout)
            #corr_vout = signal.medfilt(corr_vout)
            corr_codes, corr_dnl = DNL(vin, corr_vout, Nbits)
            #corr_support = corr_codes
            #corr_inl = INL(corr_dnl)
            corr_support, corr_inl = best_fit_INL(vin, corr_vout, Nbits)
            # display
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), dpi=150)
            plt.subplot(131)
            #plt.hold('on')
            plt.step(vin, vout, 'k-', linewidth=2.)
            plt.step(vin, corr_vout, 'b-', linewidth=2.)
            plt.xlabel('$\Delta V_{in}$ [V]')
            plt.ylabel('$V_{out}$ [V]')
            # plot the INL and the DNL
            plt.subplot(132)
            #plt.hold('on')
            plt.plot(codes[1:], dnl, 'k-', linewidth=2.)
            plt.plot(corr_codes[1:], corr_dnl, 'b-', linewidth=2.)
            plt.xlabel('$\Delta V_{in}$ [V]')
            plt.ylabel('DNL [LSB]')
            plt.subplot(133)
            #plt.hold('on')
            plt.plot(support[1:], inl, 'k-', linewidth=2.)
            plt.plot(corr_support[1:], corr_inl, 'b-', linewidth=2.)
            plt.xlabel('$\Delta V_{in}$ [V]')
            plt.ylabel('INL [LSB]')
            fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
            plt.savefig('{}.eps'.format(outputName), bbox_inches='tight')

            gain, off = gain_offset(vin, vout)
            print("Before Calib:\n\tGain\t: {:.3f}\n\tOffset\t: {:.3f}".format(gain, off))
            gain, off = gain_offset(vin, corr_vout)
            print("After Calib:\n\tGain\t: {:.3f}\n\tOffset\t: {:.3f}".format(gain, off))
            quad, off_idx, r_off = offset(dba, estimate, weight, version)
            if off_idx >= 0:
                print("Offset Before Calib\t: {:.3f} V --> {}".format(vin[off_idx], r_off[6:13]))
            quad, off_idx, r_off = offset(dba, estimate, corr_weight[:-1], version)
            if off_idx >= 0:
                print("Offset After Calib\t: {:.3f} V --> {}".format(vin[off_idx], r_off[6:13]))
            print("DNL Before Calib\t:{:.2f}/{:.2f} LSB".format(min(dnl), max(dnl)))
            print("DNL After Calib\t\t:{:.2f}/{:.2f} LSB".format(min(corr_dnl), max(corr_dnl)))
            print("INL Before Calib\t:{:.2f}/{:.2f} LSB".format(min(inl), max(inl)))
            print("INL After Calib\t\t:{:.2f}/{:.2f} LSB".format(min(corr_inl), max(corr_inl)))
            print("Estimated Resolution Before Calib:\n\t{}".format(Nbits-np.log(max(dnl)-min(dnl))/np.log(2)))
            print("Estimated Resolution After Calib:\n\t{}".format(Nbits-np.log(max(corr_dnl)-min(corr_dnl))/np.log(2)))

        plt.show()
