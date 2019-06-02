import os
import numpy as np

import matplotlib.pyplot as plt

import argparse
import pdb


parser = argparse.ArgumentParser()
parser.add_argument("logfile")
args = parser.parse_args()


data_type = {'names': ('env', 'episode','step','loss0','loss1','dist','wasslow','wasshigh','reward','pol_loss','val_loss','p_left','p_right','p_fwd','lr','lp','eucl'),
             'formats': ('i4','i4','i4','float','float','float','float','float','float','float','float','float','float','float','float','float','float')}

raw=np.genfromtxt(args.logfile, dtype=data_type, delimiter=' ', invalid_raise=False, usecols=range(14))

n_epi = np.max(raw['episode'])+1
len_epi = np.max(raw['step'])+1
n_total = n_epi * len_epi

print (n_epi, len_epi, n_total)
LOW = raw['wasslow'][:n_total].reshape((n_epi, len_epi))
HIGH = raw['wasshigh'][:n_total].reshape((n_epi, len_epi))

fig,ax=plt.subplots(1,2)
ax[0].plot(np.mean(LOW,axis=0),'-')
ax[1].plot(np.mean(HIGH,axis=0),'-')
ax[0].set_title('Wasser-Low')
ax[1].set_title('Wasser-High')

plt.savefig('wasser.png')



