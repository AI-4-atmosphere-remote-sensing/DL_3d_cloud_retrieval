'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu


This script is used to generate visualization for the experiments.

'''

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
np.random.seed(0)

def plot_cot(cot,title,fname,use_log,limit):
    fig,axs = plt.subplots(1,1)
    Z = cot
    if use_log:
        pcm = axs.pcolor(Z, norm=colors.LogNorm(vmin=limit[0], vmax=limit[1]),cmap='jet', shading='auto')
    else:
        pcm = axs.pcolor(Z, cmap='jet', shading='auto')
        pcm.set_clim(limit[0],limit[1])
    # pcm.set_clim(-2.5,13) # normalized range
    # pcm.set_clim(0,360) # denormalized regular range
    axs.set(title=title)
    axs.axes.yaxis.set_visible(True)
    axs.axes.xaxis.set_visible(True)
    fig.colorbar(pcm, ax=axs)
    plt.savefig(fname)
    plt.close()

def plot_cmask(cmask,title,fname):
    fig,axs = plt.subplots(1,1)
    Z = cmask
    # pcm = axs.pcolor(Z, norm=colors.LogNorm(vmin=eps, vmax=723),cmap='jet', shading='auto')
    pcm = axs.pcolor(Z, cmap='jet', shading='auto')
    pcm.set_clim(0,1)
    axs.set(title=title)
    axs.axes.yaxis.set_visible(True)
    axs.axes.xaxis.set_visible(True)
    fig.colorbar(pcm, ax=axs)
    plt.savefig(fname)
    plt.close()


if __name__=="__main__":
    gt1 = np.random.randint(1,440,(5,10,10,1))
    gt2 = np.random.randint(0,2,(5,10,10,1))
    gt = np.concatenate((gt1,gt2),axis=3)