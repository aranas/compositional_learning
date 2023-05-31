#imports
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import textwrap

from cryptic_rnn import *
from run_models_functions import *

arch = '_1' #how many hidden units
affix = '_b' #plot primitive or balanced
mod_names = 'final_mod' + affix #'init_mod_b', 'best_mod_b', 'final_mod_b', 'init_mod_p', 'best_mod_p', 'final_mod_p'

# read in models & losses
d_models = torch.load('results/2seqs_res'+ arch +'_2000_trainlarge_modelonly.pt')
with open('results/2seqs_res'+ arch +'_2000_trainlarge_losses.pkl', 'rb') as f:
    data = pickle.load(f)

mod_config = d_models['config_model']

# get dimensions
n_loss, n_epochs, n_sim = data.shape
# select those with train loss of epoch lower than 1
sel = data.sel(loss_type=['train_loss_b','train_loss_p'], epoch=n_epochs-1) < 0.001
# get indices of those that are true for both losses
sel = sel.all(dim='loss_type')

d_models[mod_names].append(d_models[mod_names][0].copy())
tensor_weights = torch.from_numpy(np.array([[0,0,8,11,14,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]]))
d_models[mod_names][-1]['input2hidden.weight'] = tensor_weights
d_models[mod_names][-1]['input2hidden.bias'] = torch.from_numpy(np.array([0]))
d_models[mod_names][-1]['fc1tooutput.bias'] = torch.from_numpy(np.array([0]))
d_models[mod_names][-1]['fc1tooutput.weight'] = torch.from_numpy(np.array([[1]]))
d_models['test'].append(d_models['test'][0])
d_models['train'+affix].append(d_models['train'+affix][0])

fig, axs = plt.subplots(2, 10, figsize=(15, 5)) #sum(sel).item()
models_plot = np.where(sel)[0]
for i,ix in enumerate(models_plot):
    if i > 9:
        break
    tensor_weights = d_models[mod_names][ix]['input2hidden.weight']#input2hidden or fc1tooutput
    state_dict = d_models[mod_names][ix]
    tensor_weights = torch.cat((tensor_weights[:, 2:6], tensor_weights[:, -5:]), dim=1)

    # plot weights
    axs[0,i].imshow(tensor_weights, aspect='auto')
    axs[0,i].set_yticklabels([])
    axs[0,i].set_xticks(range(tensor_weights.shape[1]))  # Set the x tick positions
    axs[0,i].set_xticklabels(['a','b','c','d','+','*','-','=','recurrent'], rotation='vertical')  # Set the custom x tick labels

    #loss to string
    if ix != 10:
        loss_test = data.sel(loss_type='test_loss' + affix, epoch=n_epochs-1)[ix].item()
        loss_train = data.sel(loss_type='train_loss' + affix, epoch=n_epochs-1)[ix].item()
        title = 'loss_test:' + str(round(loss_test,3)) + '\n loss_train:' + str(round(loss_train,3)) + '\n'
        title = title + str(d_models['cue_dict'][ix].values())
    else:
        title = 'model solution: weights correspond to value'


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
plt.savefig('figures/res'+ arch + affix +'_trainlarge.png')

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