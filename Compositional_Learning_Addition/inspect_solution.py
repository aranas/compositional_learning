#imports
import torch

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import textwrap

from cryptic_rnn import *
from run_models_functions import *

# read in models & losses
d_models = torch.load('results/2seqs_res_20_modelonly.pt')
with open('results/2seqs_res_20_losses.pkl', 'rb') as f:
    data = pickle.load(f)

# get dimensions
n_loss, n_epochs, n_sim = data.shape
# select those with train loss of epoch lower than 1
sel = data.sel(loss_type=['train_loss_p','train_loss_b'], epoch=n_epochs-1) < 0.001
# get indices of those that are true for both losses
sel = sel.all(dim='loss_type')


mod_names = ['init_mod_b', 'best_mod_b', 'final_mod_b', 'init_mod_p', 'best_mod_p', 'final_mod_p'] #'init_mod_b', 'best_mod_b', 'final_mod_b', 'init_mod_p', 'best_mod_p', 'final_mod_p'
fig, axs = plt.subplots(len(mod_names), len(sel), figsize=(15, 5))

#loop through x and mod_names
vmin = -5
vmax = 5
for idx, mod_name in enumerate(mod_names):
    print('plotting weights for '+mod_name)
    for ix in range(n_sim):
        tensor_weights = d_models[mod_name][ix]['input2hidden.weight']#input2hidden or fc1tooutput
        # plot weights
        axs[idx,ix].imshow(tensor_weights, aspect='auto',vmin=vmin, vmax=vmax)
        axs[idx, ix].set_yticklabels([])
        #loss to string
        loss_test = data.sel(loss_type='test_loss_b', epoch=n_epochs-1)[ix].item()
        loss_train = data.sel(loss_type='train_loss_b', epoch=n_epochs-1)[ix].item()
        title = 'loss_test:' + str(round(loss_test,3)) + '\n loss_train:' + str(round(loss_train,3)) + '\n'

        if ix == 0:
            axs[idx, ix].set_ylabel(mod_name, rotation=0, ha='right')
        
        if idx == 0:
            title = title + str(d_models['cue_dict'][0].values())
        
            wrapped_title = '\n'.join(textwrap.wrap(title, width=15))
            axs[idx,ix].set_title(wrapped_title,fontsize=8)

        if idx == len(mod_names)-1:
            axs[idx,ix].set_xlabel('input units')
    #plt.show()
plt.tight_layout()
plt.savefig('figures/2seqs_res_20_weigths.png')
    