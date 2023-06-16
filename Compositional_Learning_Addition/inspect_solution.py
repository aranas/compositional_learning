#imports
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import textwrap

from cryptic_rnn import *
from run_models_functions import *
from inspect_weights import *

arch = '_1' #how many hidden units
affix = '_p' #plot primitive or balanced
mod_names = 'final_mod' + affix #'init_mod_b', 'best_mod_b', 'final_mod_b', 'init_mod_p', 'best_mod_p', 'final_mod_p'

# read in models & losses
fname = '2seqs_1_500'
d_models = torch.load(f'results/{fname}_modelonly.pt')
with open(f'results/{fname}_losses.pkl', 'rb') as f:
    data = pickle.load(f)
# get dimensions
n_loss, n_epochs, n_sim = data.shape
mod_config = d_models['config_model']


# select best performing models (arbitrary threshold)
loss_thresh = 100
# find loss threshold that captures 5 highest losses
thresh_max = np.sort(data.sel(loss_type=[f'train_loss{affix}'], epoch=n_epochs-1).values.flatten())[-5]
# find loss threshold that captures 5 lowest losses
thresh_min = np.sort(data.sel(loss_type=[f'train_loss{affix}'], epoch=n_epochs-1).values.flatten())[5]

#select data that is above max threshold or below min threshold
sel = data.sel(loss_type=[f'train_loss{affix}'], epoch=n_epochs-1) > thresh_max
sel = sel + (data.sel(loss_type=[f'train_loss{affix}'], epoch=n_epochs-1) < thresh_min)

sel = np.where(sel.all(dim='loss_type'))[0]
print(f'based on threshold of {loss_thresh}, we select {len(sel)} models')



# correlate weights with input values
corr_p, corr_b = corr_weights2hidden_inputs(d_models, verbose=False)
#select correlation based on affix
corr = corr_p if affix == '_p' else corr_b
#scale correlation values to be between 0 and 1 
corr = (corr - np.min(corr)) / (np.max(corr) - np.min(corr))
#sort sel indices according to correlation strength
sel = sel[np.argsort(corr[sel])][::-1]

#Plot up to 10 models
n_plot = int(min(sum(sel), 10)) #only plot up to 10 models
print(f'plotting {n_plot} models')
fig, axs = plt.subplots(2, n_plot, figsize=(15, 5)) 

for i,ix in enumerate(sel[:n_plot]):
    tensor_weights = d_models[mod_names][ix]['input2hidden.weight']#input2hidden or fc1tooutput
    state_dict = d_models[mod_names][ix]
    tensor_weights = torch.cat((tensor_weights[:, 2:6], tensor_weights[:, -5:]), dim=1)

    # plot weights
    axs[0,i].imshow(tensor_weights, aspect='auto')
    axs[0,i].set_yticklabels([])
    axs[0,i].set_xticks(range(tensor_weights.shape[1]))  # Set the x tick positions
    axs[0,i].set_xticklabels(['a','b','c','d','+','*','-','=','recurrent'], rotation='vertical')  # Set the custom x tick labels

    #loss to string
    loss_test = data.sel(loss_type='test_loss' + affix, epoch=n_epochs-1)[ix].item()
    loss_train = data.sel(loss_type='train_loss' + affix, epoch=n_epochs-1)[ix].item()
    title = 'loss_test:' + str(round(loss_test,3)) + '\n loss_train:' + str(round(loss_train,3)) + '\n'
    title = title + str(d_models['cue_dict'][ix].values()) + '\n' + str(round(corr[ix],3))

    if ix == 0:
        axs[0, i].set_ylabel(mod_names, rotation=0, ha='right')
    wrapped_title = '\n'.join(textwrap.wrap(title, width=15))
    axs[0,i].set_title(wrapped_title,fontsize=8)
    # add colorbar to each subplot
    fig.colorbar(axs[0,i].imshow(tensor_weights, aspect='auto'), ax=axs[0,i])


    #reinstantiating model
    model = OneStepRNN(mod_config['input_size'], mod_config['output_size'], 
                    mod_config['hidden_size'], mod_config['num_layers'], mod_config['xavier_gain'])
    model.load_state_dict(state_dict)

        
    df = test_preds(model, [d_models['test'][ix]], mod_config['hidden_size'])
    df_train = test_preds(model, [d_models['train'+affix][ix]], mod_config['hidden_size'])
    preds, labs = df['pred'], df['label']
    #round preds
    print(pd.DataFrame({'pred':np.round(preds), 'label':labs}))

    xy = np.arange(np.min(preds)-1, np.max(labs)+1, 0.1)
    r2_val = r2_score(df['pred'],df['label'])
    df_fin = df.groupby(['trial']).mean().sort_values(by = 'acc' , ascending=False)
    for d in df:
        axs[1,i].scatter(df['label'], df['pred'], color = 'red')
        axs[1,i].scatter(df_train['label'], df_train['pred'], color = 'blue')
    axs[1,i].plot(xy,xy)


    #plt.show()
plt.tight_layout()
plt.savefig(f'figures/weights_{fname}{affix}.png')

'''
#TEST, UNDERSTAND SOLUTION
model = OneStepRNN(mod_config['input_size'], mod_config['output_size'], 
                    mod_config['hidden_size'], mod_config['num_layers'], mod_config['xavier_gain'])
model.load_state_dict(d_models[mod_names][8])
seq_dat = convert_seq2inputs([[('+', 'A'), ('+', 'A'), '=', 4],[('+', 'A'), ('+', 'A'), '=', 4]], num_classes=22, seq_len=5)
seq_dat = DataLoader(seq_dat, batch_size=1, shuffle=True, collate_fn=collate_fn)

df = test_preds(model, [seq_dat], mod_config['hidden_size'])
for ix in range(1,len(d_models[mod_names])):
    print(ix)
    print(d_models[mod_names][ix]['fc1tooutput.weight'])
    print(d_models[mod_names][ix]['fc1tooutput.bias'])
'''