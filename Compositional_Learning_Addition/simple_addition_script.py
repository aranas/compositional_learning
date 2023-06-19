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
    batchsize   = 2

    ## Initiate model
    # model parameters
    config_model = {
        'input_size': num_classes,
        'output_size': 1,
        'num_layers': 1,
        'hidden_size': 1,
        'xavier_gain': 0.0001,
    }
    # Run model in parallel
    ## training parameters:
    config_train = {
        'batchsize': batchsize,
        'learningRate': 0.005,
        'epochs': 250,
        'num_sims': 100,
        'n_train_seq': 2,
        'valsorted': True,
    }
    random.seed(1234)
    random_seeds = random.sample(
        list(range(config_train['num_sims'])), config_train['num_sims']
    )
    t1 = time.time()
    model_list = []
    ctx = mp.get_context('spawn')

    with ctx.Pool() as pool:
        multiple_results = [pool.apply_async(run_exp, args=(config_model, config_train, config_train['n_train_seq'], seed, device))
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

    data = xr.DataArray(
        np.stack([d_losses[k] for k in d_losses]),
        dims=('loss_type', 'epoch', 'sim'),
        coords={'loss_type': list(d_losses.keys())},
    )

    ## Save models & loss
    if config_train['valsorted']==True:
        affix = '_trainsorted'
    else:
        affix = ''
    torch.save(d_models, f"results/{config_train['n_train_seq']}seqs_{config_model['hidden_size']}_{config_train['epochs']}{affix}_batchsize{config_train['batchsize']}_modelonly.pt")
    with open(f"results/{config_train['n_train_seq']}seqs_{config_model['hidden_size']}_{config_train['epochs']}{affix}_batchsize{config_train['batchsize']}_losses.pkl", 'wb') as f:
        pickle.dump(data, f)
if __name__ == "__main__":
    main()


# %%
