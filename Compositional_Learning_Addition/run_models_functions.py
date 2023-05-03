import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

def run_loss(model,optimizer,criterion, train_data, validation_data, epochs, hidden_size, device, verbose = False):
    
    loss_history = np.empty((0,1))
    train_loss = np.empty((0,1))
    test_loss = np.empty((0,1))
    min_loss = 100000
    for epoch in range(epochs):
        lossTotal = 0
        for i, (seqs,label) in enumerate(train_data):
            #undo padding by removing all inner lists that contain only zeros
            seqs = [x for x in seqs if x != [0]*len(x[0])][0]
            seqs = seqs.unsqueeze(0)
            if len(label) == 1:
                label = label[0]
                
            #train
            seqs = seqs.to(device)
            label = label.to(device)
            output, loss = train(seqs,label,model,optimizer,criterion,device)
            lossTotal += loss # add MSE -> sum of square errors 
        loss_history = np.vstack([loss_history, lossTotal])

        lossTrain = test(model, validation_data[0], criterion, device, hidden_size)
        train_loss = np.vstack([train_loss,lossTrain])
        lossTest = test(model, validation_data[1], criterion, device, hidden_size)
        test_loss = np.vstack([test_loss, lossTest])

        if lossTest < min_loss:
            min_loss = lossTest
            best_model = model.state_dict()

    final_model = model.state_dict()

    return best_model, final_model, loss_history, train_loss, test_loss 

def run_exp(config_model, config_train, seed, device):
    ## Generate input
    num_classes = 22
    num_inputs  = 4
    batchsize   = 1
    torch.manual_seed(seed)
    trainset_b, trainset_p, testset, cue_dict = generate_sequence_data(num_inputs,num_classes,batchsize)

    # Initiate RNNs
    model_b = OneStepRNN(config_model['input_size'], config_model['output_size'], 
                        config_model['hidden_size'], config_model['num_layers'], config_model['xavier_gain'])
    model_p = copy.deepcopy(model_b)
    model_p.to(device)
    model_b.to(device)
    
    criterion = nn.MSELoss()   
    optimizer = torch.optim.Adam(model_b.parameters(), lr=config_train['learningRate'])
    best_mod_b, final_mod_b, loss_b, train_loss_b, test_loss_b = run_loss(model_b,optimizer,criterion, 
                                 trainset_b, [trainset_b, testset], 
                                 config_train['epochs'], config_model['hidden_size'], device)
    
    optimizer = torch.optim.Adam(model_p.parameters(), lr=config_train['learningRate'])
    best_mod_p, final_mod_p, loss_p, train_loss_p, test_loss_p = run_loss(model_p,optimizer,criterion, 
                                 trainset_p, [trainset_b, testset], 
                                 config_train['epochs'], config_model['hidden_size'], device)
    
    return {'cue_dict':cue_dict,'test': testset,\
           'loss_b':loss_b, 'train_loss_b':train_loss_b, 'test_loss_b':test_loss_b, 'final_mod_b': final_mod_b, 'best_mod_b': best_mod_b,\
           'loss_p':loss_p, 'train_loss_p':train_loss_p, 'test_loss_p':test_loss_p, 'final_mod_p': final_mod_p, 'best_mod_p': best_mod_p}

def test(model, testdata, criterion, device, hidden_size=20):
    model.eval()
    loss_set = 0
    for x,y in testdata:
        x = x.to(device)
        for i in range(len(x)):
            hidden = torch.zeros(1, hidden_size)[0].to(device)
            for step in x[i]:
                y = torch.tensor([y[i].item()]).to(device)
                hidden = hidden.to(device)
                hidden, y_hat = model.get_activations(step,hidden)
            loss_set += criterion(y_hat, y).item()
            
    return loss_set

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

def predcorr(mod_config, mod_dicts, tests, hidden_size, plot_corr = True):
    dfs1 = []
    for i in range(len(mod_dicts)):
        model = OneStepRNN(mod_config['input_size'], mod_config['output_size'], 
                        mod_config['hidden_size'], mod_config['num_layers'], mod_config['xavier_gain'])
        model.load_state_dict(mod_dicts[i])

        df = test_preds(model, [tests[i]], hidden_size)
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
    #print length ft_idx
    print('Number of models successfully trained: ' + str(len(ft_idx)))
    
    cue_dicts = [res1['cue_dicts'][i] for i in ft_idx]   
    mods_b = [res1['mods_b'][i] for i in ft_idx]
    mods_p = [res1['mods_p'][i] for i in ft_idx]
    
    return {'mods_b':mods_b, 'mods_p':mods_p, 'cue_dicts': cue_dicts}
