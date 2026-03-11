import numpy as np
import numpy.ma as ma
import os
import random
import matplotlib.pyplot as plt
import matplotlib
import re
from network import SpinNetwork, IncNetwork
import torch


ROOT = "C:/Users/yanic/Documents/GitHub/CAP/"
functions = {
    "a": (lambda x: float(re.findall("a-?\d+.?\d*", x)[0].replace("a", ""))),
    "inc": (lambda x: float(re.findall("inc\d+", x)[0].replace("inc", "")))
}


print("Create Networks")
spin_net = SpinNetwork()
inc_net = IncNetwork()

print("Load Networks")
device = torch.device('cpu')
spin_net.model.load_state_dict(torch.load(f"{ROOT}/Spin_network.net", map_location=device))
inc_net.model.load_state_dict(torch.load(f"{ROOT}/Inc_network.net", map_location=device))

def plotriafquick(infile):
    # load file
    data = np.load(infile)

    a = functions["a"](infile)
    inc = functions["inc"](infile)

    a_guess = spin_net.predict(torch.from_numpy(data["image"].flatten()).float())
    inc_guess = inc_net.predict(torch.from_numpy(data["image"].flatten()).float())

    # get pixelscale
    muascorr = muscale(data['mbh'], data['dbh'], data['width'])

    fig = plt.figure()
    mycmap = 'cubehelix'

    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

    ##set boundaries of subplots
    # plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.88, wspace=0.15, hspace=0.09)

    ##scale flux
    minflux = 0.0
    maxflux = 1.0

    levs11 = np.linspace((minflux), (maxflux), 10)

    ##get min and max axes values
    extent = [-(data['width'] * muascorr), data['width'] * muascorr, -data['width'] * muascorr,
              data['width'] * muascorr]

    ax1 = fig.add_subplot(1, 1, 1, frame_on='True', aspect='equal', facecolor='k')

    i1 = plt.imshow(((ma.fix_invalid(data['image'].T)) / ma.amax(data['image'].T)), vmin=0, vmax=1, extent=extent,
                    cmap=mycmap, rasterized=True, origin='lower')
    print('total flux', ma.sum(ma.fix_invalid(data['image'])))

    ax1.annotate(r'$\mathrm{\nu=230\,GHz}$', xy=(0.98, 0.91),
                 xycoords='axes fraction',  horizontalalignment='right',
                 verticalalignment='bottom', color='white')
    ax1.annotate(r'$\mathrm{S_{tot}=%1.1f\,Jy}$' % (ma.sum(ma.fix_invalid(data['image']))), xy=(0.1, 0.1),
                 xycoords='axes fraction',  horizontalalignment='left',
                 verticalalignment='bottom', color='white')

    ax1.set_ylabel(r'${\rm Relative\,Declination\,[\mu as]}$')
    ax1.set_xlabel(r'${\rm Relative\,R.A.\,[\mu as]}$')

    if np.abs(extent[0]) >= 50:
        ax1.yaxis.set_major_locator(plt.MultipleLocator(50))
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(10))
        ax1.xaxis.set_major_locator(plt.MultipleLocator(50))
        ax1.xaxis.set_minor_locator(plt.MultipleLocator(10))

    else:
        ax1.yaxis.set_major_locator(plt.MultipleLocator(20))
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(5))
        ax1.xaxis.set_major_locator(plt.MultipleLocator(20))
        ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))

    ax1.tick_params(which='major', direction='out', pad=5, right='on', top='off', length=8.0, width=0.75)
    ax1.tick_params(which='minor', direction='out', pad=5, right='on', top='off', length=4.0, width=0.5)

    for spine in ax1.spines.values():
        spine.set_edgecolor('k')

    # set colorbar
    cbar1 = plt.colorbar(i1, ticks=levs11, format='%1.1f')
    cbar1.set_label(r'${S}/{S}_{\mathrm{max}}$', rotation=0)
    cbar1.ax.tick_params(direction='out', width=1, length=8, pad=5)
    cbar1.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1], update_ticks=True)
    cbar1.set_ticklabels([r'$0$', r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1$'], update_ticks=True)

    plt.title(f"(Is|Guess)   a = {a: .2f}|{a_guess: .2f}   i = {int(inc)}|{int(inc_guess)}")
    plt.show()


def muscale(mbh, dbh, width):
    pc = 3.08e18
    rgeo = 1.4774e5  # G*m_sun/c^2
    jansky = 1.0e-23

    # jansky correction
    distance = dbh * pc
    theta = (width * mbh * rgeo / distance)

    muasscale = theta / width * (6.48e5 / np.pi) * 1.0e6

    return muasscale


path = f"{ROOT}/test"

test_files = [file for file in os.listdir(path) if file[-4:] == ".npz"]
random.shuffle(test_files)
for file in test_files:
    plotriafquick(f"{ROOT}/test/{file}")
