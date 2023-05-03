# %%
import random
import time

import numpy as np
import pandas as pd
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

def plot_loss(loss_arrays, title='', labels=['train', 'test1', 'test2'],  colors=['green', 'yellow', 'red']):

    fig, axs = plt.subplots()
    for i, arr in enumerate(loss_arrays):
        x = np.arange(0,arr.shape[0],1)
        mn = arr.mean(axis=1)
        errs = arr.std(axis=1)
        
        axs.plot(x, mn, label = labels[i], color = colors[i])
        axs.fill_between(x, mn - errs, mn + errs, alpha = 0.3, facecolor = colors[i])
    
    axs.set_xlabel('epoch')
    axs.set_ylabel('loss')
    axs.legend()
    
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
    trainseqs_b, trainseqs_p, testseqs, cue_dict = generate_sequence_data(num_inputs,num_classes,batchsize)

    ## Initiate model
    # model parameters
    config_model = {}
    config_model['input_size']      = num_classes
    config_model['output_size']     = 1
    config_model['num_layers']      = 1
    config_model['hidden_size']     = 20
    config_model['xavier_gain']     = 0.0001

    # Run model in parallel
    ## training parameters:
    config_train = {}
    config_train['batchsize']   = batchsize
    config_train['learningRate']= 0.005
    config_train['epochs']      = 500
    config_train['num_sims']    = 8

    random.seed(1234)
    random_seeds = random.sample([i for i in range(config_train['num_sims'])], config_train['num_sims'])
    t1 = time.time()
    res = [run_exp(config_model, config_train,1,device)]     
    t2 = time.time()
    print('run time: ', (t2-t1)/60)
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

    plot_loss([res['loss_b'], res['train_loss_b'], res['test_loss_b']],labels=['train', 'test-train', 'test'], colors = ['green', 'orange', 'red'], title = 'balanced -no primitives')
    plot_loss([res['loss_p'], res['train_loss_p'], res['test_loss_p']],labels=['train', 'test-train',  'test'], colors = ['green', 'orange', 'red'], title = 'with primitives')

    plot_predcorr(config_model ,res['final_mod_b'], res['test'], config_model, title = 'final: balanced -no primitives')
    plot_predcorr(config_model, res['final_mod_p'], res['test'], config_model, title = 'final: with primitives')
   
    plot_predcorr(config_model ,res['final_mod_b'], res['test'], config_model, title = 'best: balanced -no primitives')
    plot_predcorr(config_model, res['final_mod_p'], res['test'], config_model, title = 'best: with primitives')

    # select fully trained ones
    acc_df = pd.DataFrame({'train_b': res['train_loss_b'][-1,:],'train_p': res['train_loss_p'][-1,:].reshape(-1),\
                           'test_b': res['test_loss_b'][-1,:],'test_p': res['test_loss_p'][-1,:]})
    res_2input_20 = {'mods_b':res['final_mod_b'], 'mods_p':res['final_mod_p'], 'losses_b_final': res['loss_b'][-1,:], 'losses_p_final':res['loss_p'][-1,:],\
            'res':res, 'cue_dicts': res['cue_dict'], 'acc_df':acc_df }
    accres2_20 = extract_ft(res_2input_20)
    
    ## Save models
    torch.save(accres2_20, 'results/2seqs_res_20_dictonly.pt')

if __name__ == "__main__":
    main()

