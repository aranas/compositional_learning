# %%
import random
import time

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from cryptic_rnn import *
from run_models_functions import *

def plot_predcorr(config, model,test_data, model_config,title):
    plt.figure()
    r2, _, _ = predcorr(config, model, test_data, model_config['hidden_size'], plot_corr =True)
    plt.title(title + '; r^2 = '+ str(round(r2, 3)))
    plt.savefig('figures/predcorr_'+title+'.png')

def plot_loss(data, title='',  colors=['green', 'yellow', 'red']):

    fig, ax = plt.subplots()
    for i, level in enumerate(data.coords['loss_type'].values):
        level_data = data.sel(loss_type=level)
        mean = level_data.mean(dim='sim')
        std = level_data.std(dim='sim')
        mean.plot.line(ax=ax, label=level, color = colors[i])
        ax.fill_between(mean['epoch'], mean-std, mean+std, alpha=0.3,  facecolor = colors[i])
    plt.legend()
    
    fig.suptitle(title, fontsize=10)
    plt.savefig('figures/loss_'+title+'.png')

def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("GPU is not enabled")
        num_processes = mp.cpu_count()
        print("Number of CPUs: ", num_processes)
    else:
        print("GPU is enabled")

    return device

def main():
    
    # Set Device function (to GPU)
    device = set_device()

    ## Generate input
    num_classes = 22
    num_inputs  = 4
    batchsize   = 1

    ## Initiate model
    # model parameters
    config_model = {}
    config_model['input_size']      = num_classes
    config_model['output_size']     = 1
    config_model['num_layers']      = 1
    config_model['hidden_size']     = 1
    config_model['xavier_gain']     = 0.0001

    # Run model in parallel
    ## training parameters:
    config_train = {}
    config_train['batchsize']   = batchsize
    config_train['learningRate']= 0.005
    config_train['epochs']      = 2000
    config_train['num_sims']    = 10

    random.seed(1234)
    random_seeds = random.sample([i for i in range(config_train['num_sims'])], config_train['num_sims'])
    t1 = time.time()
    model_list = []
    ctx = mp.get_context('spawn')

    with ctx.Pool() as pool:
        multiple_results = [pool.apply_async(run_exp, args=(config_model, config_train, seed, device))
                                            for seed in tqdm(random_seeds)]
        results =[res.get() for res in multiple_results]
    t2 = time.time()
    print('run time: ', (t2-t1)/60)

    res = results
    # turn list of dicts into dict of lists
    res = {k: [dic[k] for dic in res] for k in res[0]}
    for key in res:
        if isinstance(res[key][0],np.ndarray):
            res[key] = np.stack(res[key])
            #reshape
            res[key] = res[key].squeeze().T
    
    print('balanced loss', res['loss_b'][-1,:].mean())
    print('primitives loss', res['loss_p'][-1,:].mean())

    keys_model_param = ['best_mod_p', 'best_mod_b', 'final_mod_b', 'final_mod_p', 'init_mod_b', 'init_mod_p', 'cue_dict','test', 'train_b','train_p']
    keys_loss = ['loss_b', 'loss_p', 'train_loss_b', 'train_loss_p', 'test_loss_b', 'test_loss_p']
    d_models = {k: v for k, v in res.items() if k in keys_model_param}
    d_models['config_model'] = config_model
    d_losses = {k: v for k, v in res.items() if k in keys_loss}

    data = xr.DataArray(np.stack([d_losses[k] for k in d_losses.keys()]), dims=('loss_type','epoch','sim'), coords={'loss_type': list(d_losses.keys())})

    ##Plot loss
    plot_loss(data.sel(loss_type=data['loss_type'].str.endswith('b')), colors = ['green', 'orange', 'red'], title = 'balanced -no primitives')
    plot_loss(data.sel(loss_type=data['loss_type'].str.endswith('p')), colors = ['green', 'orange', 'red'], title = 'with primitives')

    plot_predcorr(config_model ,d_models['final_mod_b'], d_models['test'], config_model, title = 'final: balanced -no primitives')
    plot_predcorr(config_model, d_models['final_mod_p'], d_models['test'], config_model, title = 'final: with primitives')
   
    plot_predcorr(config_model ,d_models['best_mod_b'], d_models['test'], config_model, title = 'best: balanced -no primitives')
    plot_predcorr(config_model, d_models['best_mod_p'], d_models['test'], config_model, title = 'best: with primitives')
    
    ## Save models & loss
    torch.save(d_models, 'results/2seqs_res_1_modelonly.pt')
    with open('results/2seqs_res_1_losses.pkl', 'wb') as f:
        pickle.dump(data, f)
if __name__ == "__main__":
    main()

