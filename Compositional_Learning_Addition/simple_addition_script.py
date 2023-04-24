# %%
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
import random
import time
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from cryptic_rnn import *

## %%
def collate_fn(batch):
    # batch is a list of tuples (sequence, out_state)
    sequences, out_states = zip(*batch)
    seq_len = max([len(t[0]) for t in batch])
    # pad sequences to seq_len
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.)
    
    return padded_sequences, out_states

def generate_sequence_data(num_inputs,num_classes,batchsize,verbose=False):
        ops = '+'
        total_syms = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
        
        all_syms = total_syms[:num_inputs]
        all_input_vals = list(np.arange(2,18))
        input_vals = random.sample(all_input_vals,num_inputs)
        
        # randomly select values for each input
        cue_dict = {}
        for i, s in enumerate(all_syms):
            cue_dict[s] = input_vals[i]
        
        primitives = generate_pos_primitives(all_syms, cue_dict)
        trainseqs = generate_pos_other(ops, all_syms, cue_dict)[:2]
        trainseqs_b = trainseqs + generate_balanced_primitives(ops, all_syms, cue_dict)
        trainseqs_p = trainseqs + primitives

        testseqs_all = generate_pos_trials(ops, all_syms, all_syms, cue_dict)
        testseqs = [seq for seq in testseqs_all if seq not in trainseqs_b]

        if verbose:
            print('cue_dict ',cue_dict)
            print('primitives ',primitives)
            print('trainseqs ',trainseqs)
            print('trainseqs_b ',trainseqs_b)
            print('trainseqs_p ',trainseqs_p)
            print('testseqs ', testseqs )

        # load data for primitive training
        train_inputs = convert_seq2inputs(trainseqs_b, num_classes=num_classes, seq_len=5)
        trainset_b = DataLoader(train_inputs, batch_size=batchsize, shuffle=True, collate_fn=collate_fn)

        train_inputs = convert_seq2inputs(trainseqs_p, num_classes=num_classes, seq_len=5)
        trainset_p = DataLoader(train_inputs, batch_size=batchsize, shuffle=True, collate_fn=collate_fn)

        test_inputs = convert_seq2inputs(testseqs, num_classes=num_classes, seq_len=5)
        testset = DataLoader(test_inputs, batch_size=batchsize, shuffle=True, collate_fn=collate_fn)

        return trainset_b, trainset_p, testset, cue_dict

def plot_predcorr(model,test_data, model_config,title):
    plt.figure()
    r2, _, _ = predcorr(model, test_data, model_config['hidden_size'], plot_corr =True)
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

def run_loss(model,optimizer,criterion, train_data, test_data, epochs, hidden_size, verbose = False):
    
    loss_history = np.empty((0,1))
    test_loss_history = np.empty((0,len(test_data)))
    for epoch in range(epochs):
        lossTotal = 0
        for i, (seqs,label) in enumerate(train_data):
            #undo padding by removing all inner lists that contain only zeros
            seqs = [x for x in seqs if x != [0]*len(x[0])][0]
            seqs = seqs.unsqueeze(0)
            if len(label) == 1:
                label = label[0]
            #train
            output, loss = train(seqs,label,model,optimizer,criterion)
            lossTotal += loss # add MSE -> sum of square errors 
        loss_history = np.vstack([loss_history, lossTotal])
        test_loss = test(model, test_data, criterion, hidden_size)
        test_loss_history = np.vstack([test_loss_history, test_loss])

    return loss_history, test_loss_history

def run_exp(trainset_b, trainset_p, testset, cue_dict, config_model, config_train):

    # Initiate RNNs
    model_b = OneStepRNN(config_model['input_size'], config_model['output_size'], 
                        config_model['hidden_size'], config_model['num_layers'], config_model['xavier_gain'])
    model_p = copy.deepcopy(model_b)
    
    criterion = nn.MSELoss()   
    optimizer = torch.optim.Adam(model_b.parameters(), lr=config_train['learningRate'])
    loss_b, test_loss_b = run_loss(model_b,optimizer,criterion, 
                                 trainset_b, [trainset_b, testset], 
                                 config_train['epochs'], config_model['hidden_size'])
    
    optimizer = torch.optim.Adam(model_p.parameters(), lr=config_train['learningRate'])
    loss_p, test_loss_p = run_loss(model_p,optimizer,criterion, 
                                 trainset_p, [trainset_b, testset], 
                                 config_train['epochs'], config_model['hidden_size'])
    
    return {'cue_dict':cue_dict,'test': testset,\
           'loss_b':loss_b, 'test_loss_b':test_loss_b, 'mod_b': model_b,\
           'loss_p':loss_p, 'test_loss_p':test_loss_p, 'mod_p': model_p}

def test(model, testdata, criterion, hidden_size=20):
    model.eval()
    losses_testset = []
    
    for t in testdata:
        loss_set = 0
        for x,y in t:
            for i in range(len(x)):
                hidden = torch.zeros(1, hidden_size)[0]
                for step in x[i]:
                    hidden, y_hat = model.get_activations(step,hidden)
                loss_set += criterion(y_hat, torch.tensor([y[i].item()])).item()
     
        losses_testset.append(loss_set)
        
    return losses_testset

def test_preds(model, testdata, hidden_size, suffix = ''):
    """ takes model and test data and returns a dataframe of:
        trials, ground truth outputs, and model predictions """
    
    model.eval()
    preds = []
    labs = []
    trials = []
    accs = []
    for testset in testdata:
        batch_correct = []
        for x,y in testset:
            for i in range(len(x)):
                hidden = torch.zeros(1, hidden_size)[0]
                for step in x[i]:
                    hidden, y_hat = model.get_activations(step,hidden)
                preds.append(y_hat.detach().item())
                labs.append(y[i].detach().item())
                correct = sum(torch.round(y[i]) == torch.round(y_hat)).item()
                accs.append(correct)
            trials.append(str(onehot2seq(x)))
    df = pd.DataFrame({'trial':trials, 'label'+suffix:labs, 'pred'+suffix: preds, 'acc'+suffix: accs})
    return df 

def predcorr(mods, tests, hidden_size, plot_corr = True):
    dfs1 = []
    for i in range(len(mods)):
        df = test_preds(mods[i], [tests[i]], hidden_size)
        dfs1.append(df)
    all_dfs1 = pd.concat(dfs1) 
    preds, labs = all_dfs1['pred'], all_dfs1['label']
    xy = np.arange(np.min(preds)-1, np.max(labs)+1, 0.1)
    r2_val = r2_score(all_dfs1['pred'],all_dfs1['label'])
    df_fin = all_dfs1.groupby(['trial']).mean().sort_values(by = 'acc' , ascending=False)
    if plot_corr:
        for d in dfs1:
            plt.scatter(d['label'], d['pred'])
        plt.plot(xy,xy)
        plt.xlabel('Ground truth')
        plt.ylabel('Model prediction')
        plt.title('with primitive training, R^2 = ' + str(round(r2_val, 2)) )
             
    return r2_val, df_fin, dfs1 

def extract_ft(res1):
    
    acc_df = res1['acc_df']
    ft_idx = acc_df[(acc_df['train_b'] < 1) & (acc_df['train_p'] < 1)].index
    
    cue_dicts = [res1['cue_dicts'][i] for i in ft_idx]   
    mods_b = [res1['mods_b'][i] for i in ft_idx]
    mods_p = [res1['mods_p'][i] for i in ft_idx]
    
    return {'mods_b':mods_b, 'mods_p':mods_p, 'cue_dicts': cue_dicts}

def main():

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
    config_train['epochs']      = 1000
    config_train['num_sims']    = 5

    run_exp(trainseqs_b, trainseqs_p,testseqs, cue_dict, config_model, config_train)

    t1 = time.time()
    res  = Parallel(n_jobs = -1)(delayed(run_exp)(trainseqs_b, trainseqs_p,testseqs, cue_dict, config_model, config_train) for i in tqdm(range(config_train['num_sims'])))
    t2 = time.time()
    print('run time: ', (t2-t1)/60)
    
    ## unpack trained models
    tests = [r['test'] for r in res]
    cue_dicts = [r['cue_dict'] for r in res]  
    losses_b = np.hstack([r['loss_b'] for r in res])
    train_losses_b = np.array([r['test_loss_b'][:,0] for r in res]).T
    test_losses_b = np.array([r['test_loss_b'][:,1] for r in res]).T
    losses_p = np.hstack([r['loss_p'] for r in res])
    train_losses_p = np.array([r['test_loss_p'][:,0] for r in res]).T
    test_losses_p = np.array([r['test_loss_p'][:,1] for r in res]).T
    print(losses_b[-1,:].shape)
    print(np.array(test_losses_b).shape)
    print('balanced loss', losses_b[-1,:].mean())
    print('primitives loss', losses_p[-1,:].mean())

    plot_loss([losses_b, train_losses_b, test_losses_b],labels=['train', 'test-train', 'test'], colors = ['green', 'orange', 'red'], title = 'balanced -no primitives')
    plot_loss([losses_p, train_losses_p, test_losses_p],labels=['train', 'test-train',  'test'], colors = ['green', 'orange', 'red'], title = 'with primitives')

    mods_b = [r['mod_b'] for r in res]
    plot_predcorr(mods_b, tests, config_model, title = 'balanced -no primitives')
    mods_p = [r['mod_p'] for r in res]
    plot_predcorr(mods_p, tests, config_model, title = 'with primitives')
   
    

    # select fully trained ones
    acc_df = pd.DataFrame({'train_b': train_losses_b[-1,:],'train_p': train_losses_p[-1,:].reshape(-1),\
                           'test_b': test_losses_b[-1,:],'test_p': test_losses_p[-1,:]})
    res_2input_20 = {'mods_b':mods_b, 'mods_p':mods_p, 'losses_b_final': losses_b[-1,:], 'losses_p_final':losses_p[-1,:],\
            'res':res, 'cue_dicts': cue_dicts, 'acc_df':acc_df }

    accres2_20 = extract_ft(res_2input_20)
    
    ## Save models
    savedir = 'results/2seqs_res_20'
    with open(savedir, 'wb') as f:
        pickle.dump(accres2_20, f)


if __name__ == "__main__":
    main()
