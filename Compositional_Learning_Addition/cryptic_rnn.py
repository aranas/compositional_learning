import itertools
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import pickle
import random
from sklearn.model_selection import train_test_split
import time
from joblib import Parallel, delayed
import pandas as pd
import sys
import os
import seaborn as sns 
from sklearn.metrics import r2_score
import math
import matplotlib as mpl
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
import scipy
from matplotlib import cm
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


##########################
# ---------------------
# 1. Generating sequences
# ---------------------
##########################

## Primitive trials 

def generate_primitives(inputs, cue_dict):
    """ generates primitive sequences for ('-',X) operation-symbol pairs
    Args
        inputs: list of input symbol strings
        cue_dict: dictionary of input symbols and their integer values
    Returns
        A list of sequences, where each sequence is a list with the form: [('-','X'), '=']
    """
    seq = []
    for inp in inputs:
        trial = [('-', inp), '=']
        trial.append(-1*cue_dict[inp])
        seq.append(trial)
    return seq

def generate_pos_primitives(inputs, cue_dict):
    """ generates primitive sequences for ('+',X) operation-symbol pairs
    Args
        inputs: list of input symbol strings
        cue_dict: dictionary of input symbols and their integer values
    Returns
        A list of sequences, where each sequence is a list with the form: [('+','X'), '=']
    """
    seq = []
    for inp in inputs:
        trial = [('+', inp), '=']
        trial.append(cue_dict[inp])
        seq.append(trial)
    return seq

## Balancing sequences

def generate_complex_primitives(op, inputs, cue_dict):
    """ generates 2 random simple addition sequences with all symbols appearing once (for balancing)
        op: + or - for addition or subtraction
        inputs: list of input symbol strings
        cue_dict: dictionary of input symbols and their integer values
    Returns
        A list of sequences, where each sequence is a list with the form: [(op,'X'), (op,'X'),'=']
    """
    seq = []
    all_inputs = inputs.copy()
    random.shuffle(all_inputs)
    for q in range(int(len(all_inputs)/2)):
        trial = [(op, all_inputs[2*q]), (op, all_inputs[2*q+1]), '=']
        trial.append(calculate_output(trial, cue_dict))
        seq.append(trial)
    return seq

def generate_balanced_primitives(op, inputs, cue_dict):
    """ generates 2 simple addition sequences with all symbols appearing once (for balancing) - Non random selection.
        op: + or - for addition or subtraction
        inputs: list of input symbol strings
        cue_dict: dictionary of input symbols and their integer values
    Returns
        A list of sequences, where each sequence is a list with the form: [(op,'X'), (op,'X'),'=']
    """
    seq = []
    inputs1 = inputs.copy()
    inputs2 = inputs.copy()
    n = int(len(inputs1)/2)
    for i in range(n):
        trial = [(op, inputs1[i]), (op, inputs2[i+n]), '=']
        trial.append(calculate_output(trial, cue_dict))
        seq.append(trial)
    return seq


## generate addition sequences

def generate_neg_trials(ops, input_ids, init_values, cue_dict, steps = 1):
    """ generates all permutations of addition sequences, with initial negative sign
    Args
        ops: + or - for addition or subtraction, or both
        input_ids: list of input symbol strings for the addition steps
        init_values: list of input symbol strings for the initial symbols (augends)
        cue_dict: dictionary of input symbols and their integer values
        steps: number of addition steps, e.g. steps = 1 for +X+Y=
    Returns
        A list of sequences, where each sequence is a list with the form: [('-','X'), (op,'X'), ... ,'='], where with n*
        (op,symbol) pairs 
        (where n = steps)
    """
    seq = []
    combi_inputcue = list(itertools.product(input_ids, repeat= steps))
    combi_ops = list(itertools.product(ops, repeat= steps))
    for init in input_ids:
        for cue in combi_inputcue:
            seq.append([('-',init),
                        *zip(combi_ops[0], cue), '=']) #group per time point t
    for s in seq:
        s.append(calculate_output(s, cue_dict))
    return seq

def generate_pos_trials(ops, input_ids, init_values, cue_dict, steps = 1):
    """ generates all permutations of addition sequences, with initial positive sign
    Args
        ops: + or - for addition or subtraction, or both
        input_ids: list of input symbol strings for the addition steps
        init_values: list of input symbol strings for the initial symbols (augends)
        cue_dict: dictionary of input symbols and their integer values
        steps: number of addition steps, e.g. steps = 1 for +X+Y=
    Returns
        A list of sequences, where each sequence is a list with the form: [('+','X'), (op,'X'), ... ,'=', output], where with n*
        (op,symbol) pairs 
        (where n = steps)
    """    
    seq = []
    combi_inputcue = list(itertools.product(input_ids, repeat= steps))
    combi_ops = list(itertools.product(ops, repeat= steps))
    for init in input_ids:
        for cue in combi_inputcue:
            seq.append([('+',init),
                        *zip(combi_ops[0], cue), '=']) #group per time point t
    for s in seq:
        s.append(calculate_output(s, cue_dict))
    return seq

def generate_pairs(op, inputs, cue_dict, shift):
    """ generates addition sequences for pairs of symbols, where each pair is formed between symbols seperated by
    the specified shift in rank of the symbol list
    Args
        op: + or - for addition or subtraction
        input: list of input symbol strings
        cue_dict: dictionary of input symbols and their integer values
        shift: shift in rank between symbol pairs. E.g. when rank = 0, +A+A, for rank = 2, +A+C=
    Returns
        A list of sequences, where each sequence is a list with the form: [(op,'X1'), (op,'X2'), '=', output] 
    """
    seq = []
    inputs1 = inputs.copy()
    inputs2 = inputs.copy()
    for s in range(shift):
        inputs2.append(inputs2.pop(0))
    for i in range(len(inputs1)):
        trial = [(op, inputs1[i]), (op, inputs2[i]), '=']
        trial.append(calculate_output(trial, cue_dict))
        seq.append(trial)
    return seq

def generate_neg_other(op, inputs, cue_dict):
    seq = []
    inputs1 = inputs.copy()
    inputs2 = inputs.copy()
    inputs2.append(inputs2.pop(0))
    for i in range(len(inputs1)):
        trial = [('-',inputs1[i]), (op, inputs2[i]), '=']
        trial.append(calculate_output(trial, cue_dict))
        seq.append(trial)
    return seq

def generate_pos_other(op, inputs, cue_dict):
    seq = []
    inputs1 = inputs.copy()
    inputs2 = inputs.copy()
    inputs2.append(inputs2.pop(0))
    for i in range(len(inputs1)):
        trial = [('+',inputs1[i]), (op, inputs2[i]), '=']
        trial.append(calculate_output(trial, cue_dict))
        seq.append(trial)
    return seq

## Calculating output of a trial

def operate_op(currval, step_tuple, cue_dict):
    """ Function takes current value in sequence and applies the next operation
    """
    nextval = cue_dict[step_tuple[1]]
    if step_tuple[0] == '+': # add
        currval = currval + nextval
    elif step_tuple[0] == '*': # multiply
        currval = currval * nextval
    elif step_tuple[0] == '-': # subtract
        currval = currval - nextval
    return currval

def calculate_output(step_tuple_full, cue_dict):
    """ Function calculates the output of the sequence
    Args
      step_tuple_full: full trial sequence excluding output, i.e. in the form [(+, X), ..., =]
      cue_dict: dictionary of input symbols and their integer values
    Returns
      ouput of sequence calculation
    """
    step_tuple = step_tuple_full[:-1]
    step1 = step_tuple[0]
    if len(step1) == 1: 
        curr_val = cue_dict[step1[0]]
    else:
        if step1[0] == '-':
            curr_val = -1*cue_dict[step1[1]]
        elif step1[0] == '+':
            curr_val = cue_dict[step1[1]]
    for i in range(1,len(step_tuple)):
        curr_val = operate_op(curr_val, step_tuple[i], cue_dict)
    return curr_val

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

##############################
# ---------------------------
# 2. Recurrent Neural Network
# ---------------------------
##############################

class SequenceData(Dataset):
    def __init__(self, data, labels, seq_len, stages, cont_out):

        self.data = convert_seq2onehot(data, stages)
        self.seq_len = seq_len
        if cont_out:
            self.labels = labels
        else:
            self.labels = convert_outs2labels(labels)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data[index].astype(np.float32)
        out_state = np.array(self.labels[index]).astype(np.float32)

        #convert everything to tensor for more efficient processing on GPU
        sequence = torch.from_numpy(sequence)
        out_state = torch.from_numpy(out_state)

        return sequence, out_state
    
    
def convert_seq2inputs(sequences, seq_len=5, stages = False, cont_out = True, num_classes=22):
    '''
    Function converts sequences as they are generated by generate_experiment_lists.py
    into input to be fed into RNN (one-hote encoded)
    Parameters:
        sequences: list of trials with format : [initial_value, (operation, input_cue) ... , output_value]
        num_classes: total number of features for onehot encoding
        seq_len: number of time steps per sequence
        stages: if False each unit is a time step, if True each tuple is a time step
        cont_out: if True the output is continuous, if False output is categorical
    ''' 
    seq = [sublist[:-1] for sublist in sequences]
    out = [sublist[-1] for sublist in sequences]
    
    seqdata = SequenceData(seq, out, seq_len, stages, cont_out)

    return seqdata


def convert_seq2onehot(seq, stages, num_classes=22):
    """ Function ...
    Args:
        ...
    Returns:
        ...
    """
    data = []

    for trial in seq:
        trial_data = []
        for i,t in enumerate(trial):
            if len(t)==2:
                op = torch.tensor(convert_operation[t[0]])
                op = torch.nn.functional.one_hot(op, num_classes=num_classes)
                inputcue = torch.tensor(convert_inputcue[t[1]])
                inputcue = torch.nn.functional.one_hot(inputcue, num_classes=num_classes)
                trial_data.append(op)
                trial_data.append(inputcue)
            elif t == "=":
                equals_sign = torch.tensor(convert_operation[t])
                equals_sign = torch.nn.functional.one_hot(equals_sign, num_classes=num_classes)
                trial_data.append(equals_sign)
                continue
            else:
                init = torch.tensor(convert_inputcue[t])
                init = torch.nn.functional.one_hot(init, num_classes=num_classes)
                trial_data.append(init)
                continue
                
        data.append(torch.stack(trial_data).numpy())
    return data

def onehot2seq(seqs):
    curr_trial = []
    for seq in seqs:
        for step in seq:
            curr_trial.append(onehot_dict[np.argmax(step).item()])
    return curr_trial


def convert_outs2labels(outputs, num_outs=1000):
    """ Function ...
    Args:
        ...
    Returns:
        ...
    """
    all_outs = []
    for out in outputs:
        out = torch.tensor(out)
        onehot_out = torch.nn.functional.one_hot(out, num_classes = num_outs)
        all_outs.append(onehot_out)
    return all_outs

######################
### Color schemes
######################

bp_colors = ['#00A7E1', '#F17720']
bp_colors_dark = ['#003D70', '##AE4F0A']
bp_pal = {'Balanced': '#00A7E1', 'Primitive':'#F17720'}

convert_inputcue = {'X': 0,
                    'Y': 1,
                    'A': 2, 
                    'B': 3,
                    'C': 4,
                    'D': 5,
                    'E': 6, 
                    'F': 7,
                    'G': 8,
                    'H': 9,
                    'I': 10, 
                    'J': 11,
                    'K': 12,
                    'L': 13,
                    'M': 14, 
                    'N': 15,
                    'O': 16,
                    'P': 17
                    }

convert_operation = {'+': 18,
                     '*': 19,
                     '-': 20,
                     '=': 21}

onehot_dict = {0:'X',
                1:'Y',
                2:'A', 
                3:'B',
                4:'C',
                5:'D',
                6:'E', 
                7:'F',
                8:'G',
                9:'H',
                10:'I', 
                11:'J',
                12:'K',
                13:'L',
                14:'M', 
                15:'N',
                16:'O',
                17:'P',
                18:'+',
                19:'*',
                20:'-',
                21:'=',
                    }
 


###################
#RNN
###################

class simpleRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, xavier_gain=1.0):
        super(simpleRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.xavier_gain = xavier_gain
        self.input_size = input_size
        
        # recurrent layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='relu')
        # output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        self.initialize_weights()

    def forward(self, inputs):
        output, hidden = self.rnn(inputs)
        # output layer
        output = self.output_layer(output[:, -1, :])[0][0]
        # activation function
        return output, hidden

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, self.xavier_gain)

    def init_hidden(self, batch_size):
        # initialize hidden state
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return self.hidden
    
class OneStepRNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers, xavier_gain):
        super(OneStepRNN, self).__init__()
        # Define parameters
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.xavier_gain = xavier_gain
        # Define the layers
        self.input2hidden = nn.Linear(input_size + self.hidden_size, self.hidden_size)
        self.fc1tooutput = nn.Linear(self.hidden_size, output_size)
        self.initialize_weights()

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), dim=0) ## dim = 1??
        self.hidden = nn.functional.relu(self.input2hidden(combined))
        self.output = self.fc1tooutput(self.hidden)
        return self.output, self.hidden

    def get_activations(self, x, hidden):
        self.forward(x, hidden)  # update the activations with the particular input
        return self.hidden, self.output #, self.fc1_activations

    def get_noise(self):
        return self.hidden_noise

    def init_hidden(self,batch_size=1):
        return torch.zeros(batch_size, self.hidden_size)[0]
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, self.xavier_gain)

def train(sequence, label ,model ,optimizer ,criterion, device):
    model.train()
    optimizer.zero_grad()
    loss = 0
    for seq,l in zip(sequence,[label]):
        #Read each cue in and keep hidden state for next cue
        hidden = model.init_hidden(1)
        hidden = hidden.to(device)
        l = torch.tensor([l]).to(device)
        #loop through sequence
        #to ignore padding, loop until cue '=' is seen (assumed that final input unit marks this event)
        await_cue = True
        step = 0
        while await_cue:
            output, hidden = model.forward(seq[step], hidden)
            if seq[step][-1] == 1:
                await_cue = False
            step +=1
        #Compare final output to target
        if loss == 0:
            loss = criterion(output,l)
        else:
            loss += (criterion(output,l))
    #Back-propagate
    loss.backward()
    optimizer.step()

    return output, loss.item()

def run_acc(model,optimizer,criterion, train_data, test_data, epochs, hidden_size, verbose = False):
    
    loss_history = np.empty((0,1))
    all_accs = np.empty((0,len(test_data)))
    for epoch in range(epochs):
        lossTotal = 0
        for i, (x,y) in enumerate(train_data):
            output, loss = train(x,y,model,optimizer,criterion)
            lossTotal += loss # add MSE -> sum of square errors 
        loss_history = np.vstack([loss_history, lossTotal])
        acc = test_acc(model, test_data, hidden_size)
        all_accs = np.vstack([all_accs,acc])

    return loss_history, all_accs

def run_loss(model,optimizer,criterion, train_data, test_data, epochs, hidden_size, verbose = False):
    
    loss_history = np.empty((0,1))
    test_loss_history = np.empty((0,len(test_data)))
    for epoch in range(epochs):
        lossTotal = 0
        for i, (x,y) in enumerate(train_data):
            output, loss = train(x,y,model,optimizer,criterion)
            lossTotal += loss # add MSE -> sum of square errors 
        loss_history = np.vstack([loss_history, lossTotal])
        test_loss = test_modloss(model, test_data, hidden_size)
        test_loss_history = np.vstack([test_loss_history, test_loss])

    return loss_history, test_loss_history


def test_acc(model, testdata, hidden_size, verbose = False):
    """ Args: model and test data 
        Returns: test accuracy """
    model.eval()
    accs = np.empty((1, 0))
    for testset in testdata:
        batch_correct = []
        for x,y in testset:
            correct = 0
            for i in range(len(x)):
                hidden = torch.zeros(1, hidden_size)[0]
                for step in x[i]:
                    hidden, y_hat = model.get_activations(step,hidden)
                correct += sum(torch.round(y[i]) == torch.round(y_hat)).item()
            batch_correct.append(correct/len(y))
        acc = np.mean(batch_correct)
        accs = np.append(accs, [acc])
    if verbose:
        print('test accuracy: %f ' % (acc))
    return accs

def test_modloss(model, testdata, criterion, hidden_size=20):
    model.eval()
    losses_testset = []
    
    for testset in testdata:
        loss_set = 0
        
        for x,y in testset:
            for i in range(len(x)):
                hidden = torch.zeros(1, hidden_size)[0]
                for step in x[i]:
                    hidden, y_hat = model.get_activations(step,hidden)
                loss_set += criterion(y_hat,y[i]).item()
     
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


def shuffle_weights(model):
    model2 = OneStepRNN(input_size, output_size, hidden_size, num_layers)
    mod_dict = model.state_dict()
    shuffled_dict = {layer: shuffle_tensor(val) for layer, val in mod_dict.items()}
    model2.load_state_dict(shuffled_dict)
    return model2

def shuffle_tensor(t):
    idx = torch.randperm(t.nelement())
    t = t.view(-1)[idx].view(t.size())
    return t

def run_sim(train_trials, test_trials):
    model = OneStepRNN(input_size, output_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    loss1, acc1 = run_acc(model,optimizer,criterion, train_trials, test_trials, epochs)
    return loss1, acc1, model

def run_sims(i, train_trials, test_trials):
    print('########## rep', i, '#########')
    model = OneStepRNN(input_size, output_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    loss1, acc1 = run_acc(model,optimizer,criterion, train_trials[0], test_trials, epochs)
    loss2, acc2 = run_acc(model,optimizer,criterion, train_trials[1], test_trials, epochs)
    losses = np.vstack([loss1,loss2])
    accs = np.vstack([acc1,acc2])
    return losses, accs, model


def change_dict(seqs, new_dict):
    """ recalculates sequence output"""
    inps = [s[:-1] for s in seqs]
    for inp in inps:
        inp.append(calculate_output(inp, new_dict))

    return inps

def predcorr(mods, tests, hidden_size, plot_corr = True):
    dfs1 = []
    for i in range(len(mods)):
        df = test(mods[i], [tests[i]], hidden_size)
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

def predcorr_ind_mod(mod, test, hidden_size, plot_corr = True):
    
    df = test_preds(mod, [test], hidden_size)
    preds, labs = df['pred'], df['label']
    r2_val = r2_score(preds,labs)
    
    return r2_val 



# ----------
# plotting
# ----------

def plotNNs(loss_arrays, acc_arrays, labels, colors, title, shift = 0):

    fig, axs = plt.subplots(2, 1)
    loss_cols = ['blue', 'darkblue']
    loss_labs = ['loss_with_primitive', 'loss_without_primitive']
    for i, arr in enumerate(loss_arrays):
        x = np.arange(0,arr.shape[0],1) + shift
        mn = arr.mean(axis=1)
        errs = arr.std(axis=1)
        
        axs[0].plot(x, mn, label = loss_labs[i], color = loss_cols[i])
        axs[0].fill_between(x, mn - errs, mn + errs, alpha = 0.3, facecolor = loss_cols[i])
    
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    axs[0].legend()
    
    for i, arr in enumerate(acc_arrays):
        x = np.arange(0,arr.shape[0],1) + shift
        mn = arr.mean(axis=1)
        errs = arr.std(axis=1)
        
        axs[1].plot(x, mn, label = labels[i], color = colors[i])
        axs[1].fill_between(x, mn - errs, mn + errs, alpha = 0.3, facecolor = colors[i])
    
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('accuracy')
    axs[1].legend()

    fig.suptitle(title, fontsize=10)
    
def heatmap_acc(num_inputs, df, ax):
    
    total_syms = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    map_syms = total_syms[:num_inputs]
    data_accs = np.empty((num_inputs, num_inputs))
    data_accs[:] = np.NaN
    for r, trial in enumerate(df.index):
        i = map_syms.index(eval(trial)[0])
        j = map_syms.index(eval(trial)[2])
        acc = round(df.iloc[r]['acc'], 2)
        data_accs[i,j] = acc
    
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(num_inputs), labels=map_syms)
    ax.set_yticks(np.arange(num_inputs), labels=map_syms)

    #cmap = mpl.colors.ListedColormap(['yellow', 'orange', 'darkorange','red'])
    new_reds = cm.get_cmap('Reds', 10)
    cmap=new_reds
    bounds = list(np.arange(0,1.1,0.1))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(data_accs, cmap=cmap, norm=norm)

    # Loop over data dimensions and create text annotations.
    for i in range(num_inputs):
        for j in range(num_inputs):
            if np.isnan(data_accs[i, j]):
                pass
            else:
                text = ax.text(j,i, data_accs[i, j],
                              ha="center", va="center", color="black", fontsize=12)

def heatmap_acc_sign(num_inputs, df, ax):
    
    total_syms = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    map_syms = total_syms[:num_inputs]
    data_accs = np.empty((num_inputs, num_inputs))
    data_accs[:] = np.NaN
    for r, trial in enumerate(df.index):
        i = map_syms.index(eval(trial)[1])
        j = map_syms.index(eval(trial)[3])
        acc = round(df.iloc[r]['acc'], 2)
        data_accs[i,j] = acc
    
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(num_inputs), labels=map_syms)
    ax.set_yticks(np.arange(num_inputs), labels=map_syms)
    new_reds = cm.get_cmap('Reds', 10)
    cmap=new_reds
    bounds = list(np.arange(0,1.1,0.1))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(data_accs, cmap=cmap, norm=norm)

    # Loop over data dimensions and create text annotations.
    for i in range(num_inputs):
        for j in range(num_inputs):
            if np.isnan(data_accs[i, j]):
                pass
            else:
                text = ax.text(j,i, data_accs[i, j],
                              ha="center", va="center", color="black", fontsize=12)

#####################################
# Analysis
#####################################

def get_reps(model, testdata, hidden_size):
    """ get hidden layer activations at each step"""
    model.eval()
    trials = []
    hiddens = []
    for testset in testdata:
        for x,y in testset:
            for i in range(len(x)):
                hidden_arr = np.empty((0,  hidden_size))
                hidden = torch.zeros(1, hidden_size)[0]
                for step in x[i]:
                    hidden, y_hat = model.get_activations(step,hidden)
                    hidden_arr = np.vstack([hidden_arr, hidden.detach().numpy()])
            hiddens.append(hidden_arr)
            trials.append(str(onehot2seq(x)))
    return hiddens, trials 

def find_99_acc_idx(acc_vals):
    """find number of epochs until 99% train accuracy is reached"""
    count = 0
    thresh_idx = -1
    for i, rval in enumerate(acc_vals):
        if rval > 0.99:
            count +=1
        else:
            count = 0
        if count > 10:
            thresh_idx = i - 10
            break
    return thresh_idx

###########################################
## RSA
###########################################

def calculate_RDMs_old(res, testseq, fully_trained = True):
    
    acc_df = res['acc_df']
    if fully_trained:
        all_acc_mods = acc_df[(acc_df['acc_train'] == 1) & (acc_df['acc_train_p'] == 1)].index
    else:
        all_acc_mods = acc_df.index
    print('no. 100% trained RNNs: ', len(all_acc_mods))
    mod_list = all_acc_mods # choose subset of rnns 
    
    rdms = [[] for _ in range(5)] # initialise empty lists/arrays
    rdms_p = [[] for _ in range(5)]

    # extracts results from dictionary
    mods = res['mods']
    mods_p = res['mods_p']
    cue_dicts = res['cue_dicts']
    ft_cue_dicts = [cue_dicts[j] for j in mod_list]

    for ind, m in enumerate(mod_list): # for each model 
        
        testseqs = change_dict(testseq, cue_dicts[m])
        test_inputs = convert_seq2inputs(testseq, num_classes=num_classes, seq_len=5)
        testset = DataLoader(test_inputs, batch_size=batchsize, shuffle=False)
        
        # get activations for control model
        hiddens, trials = get_reps(mods[m], [testset], hidden_size)
        for h in range(5): 
            hid_vals = np.array([hid[h,:] for hid in hiddens]) # combine activations from each trial for the time step
            rep_mat = euclidean_distances(hid_vals) # calculate euclidean distance matrix between trials
            rdms[h].append(rep_mat)
            
        # get activations for primitive trained model
        hiddens_p, trials = get_reps(mods_p[m], [testset], hidden_size)    
        for h in range(5):
            hid_vals = np.array([hid[h,:] for hid in hiddens_p])
            rep_mat = euclidean_distances(hid_vals)
            rdms_p[h].append(rep_mat)
            
    return {'rdms': rdms, 'rdms_p': rdms_p, 'ft_cue_dicts': ft_cue_dicts}

# def calculate_RDMs(res1, testseq, num_classes=22, batchsize=1,hidden_size=20, subset = 'ft', Tmax=4):
    
#     acc_df = res1['acc_df']
#     if subset == 'ft':
#         all_acc_mods = acc_df[(acc_df['acc_train'] == 1) & (acc_df['acc_train_b'] == 1)&\
#                               (acc_df['acc_train_bp'] == 1) & (acc_df['acc_train_p'] == 1)].index
#     elif subset == 'all':
#         all_acc_mods = acc_df.index
#     print('no. 100% trained RNNs: ', len(all_acc_mods))
#     mod_list = all_acc_mods # choose subset of rnns 

#     rdms = [[] for _ in range(Tmax)] # initialise empty lists/arrays
#     rdms_p = [[] for _ in range(Tmax)]
#     rdms_b = [[] for _ in range(Tmax)] # initialise empty lists/arrays
#     rdms_bp = [[] for _ in range(Tmax)]
#     # extracts res1ults from dictionary
#     mods = res1['mods']
#     mods_p = res1['mods_p']
#     mods_b = res1['mods_b']
#     mods_bp = res1['mods_bp']

#     cue_dicts = res1['cue_dicts']
#     ft_cue_dicts = [cue_dicts[j] for j in mod_list]

#     for ind, m in enumerate(mod_list): # for each model 

#         testseqs = change_dict(testseq, cue_dicts[m])
#         test_inputs = convert_seq2inputs(testseqs, num_classes=num_classes, seq_len=5)
#         testset = DataLoader(test_inputs, batch_size=batchsize, shuffle=False)

#         # get activations for control model
#         hiddens, trials = get_reps(mods[m], [testset], hidden_size)
#         for h in range(Tmax): 
#             hid_vals = np.array([hid[h+1,:] for hid in hiddens]) # combine activations from each trial for the time step
#             rep_mat = euclidean_distances(hid_vals) # calculate euclidean distance matrix between trials
#             rdms[h].append(rep_mat)

#         # get activations for primitive trained model
#         hiddens_p, trials = get_reps(mods_p[m], [testset], hidden_size)    
#         for h in range(Tmax):
#             hid_vals = np.array([hid[h+1,:] for hid in hiddens_p])
#             rep_mat = euclidean_distances(hid_vals)
#             rdms_p[h].append(rep_mat)

#         # get activations for control model
#         hiddens_b, trials = get_reps(mods_b[m], [testset], hidden_size)
#         for h in range(Tmax): 
#             hid_vals = np.array([hid[h+1,:] for hid in hiddens_b]) # combine activations from each trial for the time step
#             rep_mat = euclidean_distances(hid_vals) # calculate euclidean distance matrix between trials
#             rdms_b[h].append(rep_mat)

#         # get activations for primitive trained model
#         hiddens_bp, trials = get_reps(mods_bp[m], [testset], hidden_size)    
#         for h in range(Tmax):
#             hid_vals = np.array([hid[h+1,:] for hid in hiddens_bp])
#             rep_mat = euclidean_distances(hid_vals)
#             rdms_bp[h].append(rep_mat)

#     return {'rdms': rdms, 'rdms_p': rdms_p, 'rdms_b': rdms_b, 'rdms_bp': rdms_bp, 'ft_cue_dicts': ft_cue_dicts}

def calculate_RDMs(res1, testseq, num_classes=22, batchsize=1,hidden_size=20, Tmax=4):
    
    # initialise empty lists/arrays
    rdms_p = [[] for _ in range(Tmax)]
    rdms_b = [[] for _ in range(Tmax)] # initialise empty lists/arrays
    # extracts res1ults from dictionary
    mods_p = res1['mods_p']
    mods_b = res1['mods_b']
    cue_dicts = res1['cue_dicts']
       
    for m in range(len(cue_dicts)): # for each model 

        testseqs = change_dict(testseq, cue_dicts[m])
        test_inputs = convert_seq2inputs(testseqs, num_classes=num_classes, seq_len=5)
        testset = DataLoader(test_inputs, batch_size=batchsize, shuffle=False)

        # get activations for primitive trained model
        hiddens_p, trials = get_reps(mods_p[m], [testset], hidden_size)    
        for h in range(Tmax):
            hid_vals = np.array([hid[h+1,:] for hid in hiddens_p])
            rep_mat = euclidean_distances(hid_vals)
            rdms_p[h].append(rep_mat)

        # get activations for control model
        hiddens_b, trials = get_reps(mods_b[m], [testset], hidden_size)
        for h in range(Tmax): 
            hid_vals = np.array([hid[h+1,:] for hid in hiddens_b]) # combine activations from each trial for the time step
            rep_mat = euclidean_distances(hid_vals) # calculate euclidean distance matrix between trials
            rdms_b[h].append(rep_mat)

    return {'rdms_p': rdms_p, 'rdms_b': rdms_b, 'cue_dicts': cue_dicts}


def calculate_RDMs_prims(res1, testseq, num_classes=22, batchsize=1, step_num = 2, hidden_size=10):
    """ calculates RDM of trials and primitives
        Args: result dictionary (res1), trials to represent (testseq)
        Returns: list of RDMs for each model in the two training regimes
        """
    # initialise empty lists/arrays
    rdms_p = []
    rdms_b = [] # initialise empty lists/arrays
    
    # extracts res1ults from dictionary
    mods_p = res1['mods_p']
    mods_b = res1['mods_b']
    cue_dicts = res1['cue_dicts']

    for m in range(len(mods_p)): # for each model 

        testseqs = change_dict(testseq, cue_dicts[m])
        test_inputs = convert_seq2inputs(testseqs, num_classes=num_classes, seq_len=5)
        testset = DataLoader(test_inputs, batch_size=batchsize, shuffle=False)

        # get activations for control model
        rmat = np.empty((0, len(testseq)))
        hiddens, trials = get_reps(mods_p[m], [testset], hidden_size)
        for hid in hiddens:
            if hid.shape[0] < 4:
                rmat = np.vstack([rmat, hid[1,:]])
            else:
                rmat = np.vstack([rmat, hid[step_num,:]])
        rdms_p.append(euclidean_distances(rmat))  

        rmat = np.empty((0, len(testseq)))
        hiddens, trials = get_reps(mods_b[m], [testset], hidden_size)
        for hid in hiddens:
            if hid.shape[0] < 4:
                rmat = np.vstack([rmat, hid[1,:]])
            else:
                rmat = np.vstack([rmat, hid[step_num,:]])
        rdms_b.append(euclidean_distances(rmat))   

    return {'rdms_p': rdms_p, 'rdms_b': rdms_b,'ft_cue_dicts': ft_cue_dicts}



########### regression

def regress_RDM(time_step, rdm, ft_cue_dicts, valset_idx, ranked = False, rank_dict=None):
    rs = []
    for i, cuedict in enumerate(ft_cue_dicts):
        if ranked:
            curr_tests = change_dict(testseqs, rank_dict)
        else:
            curr_tests = change_dict(testseqs, cuedict)
        control_outs = [t[-1] for t in curr_tests]
        control_RDM = abs(np.array([control_outs]*16) - np.array([control_outs]*16).T)
        x = []
        y = []
        for p in valset_idx:
            for q in valset_idx:
                x.append(rdm[time_step][i][p,q])
                y.append(control_RDM[p,q])
        x = np.array(x).reshape(-1,1)
        y = np.array(y)
        model = LinearRegression().fit(x, y)
        r_sq = model.score(x, y)
        rs.append(r_sq)
    return rs

#############################################
#    MDS
#############################################

ca, cb, cc, cd = 'green', 'blue', 'orange', 'red'
colors1 = [ca]*4 + [cb]*4 + [cc]*4 + [ cd]*4 + ['black']*4
colors2 = [ca, cb, cc, cd]*4 + [ca, cb, cc, cd]

msize = 12
legend_elements = [Line2D([0], [0], marker=6, color='w', markerfacecolor =ca, markersize=msize, label=' + A _'),
                   Line2D([0], [0], marker=6, color='w', markerfacecolor =cb, markersize=msize, label=' + B _'), 
                   Line2D([0], [0], marker=6, color='w', markerfacecolor =cc, markersize=msize, label=' + C _'),
                   Line2D([0], [0], marker=6, color='w', markerfacecolor =cd, markersize=msize, label=' + D _'),
                   Line2D([0], [0], marker=7, color='w', markerfacecolor=ca, markersize=msize, label=' _ + A'),
                   Line2D([0], [0], marker=7, color='w', markerfacecolor=cb, markersize=msize, label=' _ + B'), 
                   Line2D([0], [0], marker=7, color='w', markerfacecolor=cc, markersize=msize, label=' _ + C'),
                   Line2D([0], [0], marker=7, color='w', markerfacecolor=cd, markersize=msize, label=' _ + D')
                   ]


def MDS_plot(matlist, testseqs, trainseqs, MDStype = 'MDS', title = '', min_dim = 0, rand_state = 0, plotlines=False):
    
    valset = [t for t in testseqs if t not in trainseqs]
    valset_idx = [testseqs.index(val) for val in valset]
    
    plt.rcParams['figure.figsize'] = 6, 6
    fig, axs = plt.subplots(2,2)

    for j, dist in enumerate(matlist):
        if MDStype == 'PCA':
            mds = PCA(n_components=3)
        if MDStype == 'MDS':
            mds = MDS(dissimilarity='precomputed',random_state=rand_state, n_components=3)

        X_transform = mds.fit_transform(dist)
        ax = axs[math.floor(j/2), j%2]
        ax.title.set_text('step: '+str(j+1))
        for i in range(len(testseqs)):
            if i in valset_idx:
                alph = 1
            else:
                alph = 0.2
            ax.scatter(X_transform[i,min_dim], X_transform[i,min_dim+1], marker=7, s=50, color = colors1[i], alpha = alph)
            ax.scatter(X_transform[i,min_dim], X_transform[i,min_dim+1], marker=6, s=50, color=colors2[i], alpha = alph)
            
        if plotlines:
            for k in range(4):
                ax.plot([X_transform[4*k,0], X_transform[4*k+3,0]], [X_transform[4*k,1], X_transform[4*k+3,1]], color = colors1[k])
                ax.plot([X_transform[k,0], X_transform[12+k,0]], [X_transform[k,1], X_transform[12 + k,1]], color = colors1[k])


    plt.suptitle('2D-MDS'+title)
    fig.legend(handles=legend_elements,  loc='center left', bbox_to_anchor=(1, 0.5)) 

def MDS_plot_3D(matlist, testseqs, trainseqs, MDStype = 'MDS', title = ''):
    
    valset = [t for t in testseqs if t not in trainseqs]
    valset_idx = [testseqs.index(val) for val in valset]    
    
    plt.rcParams['figure.figsize'] = 6, 6
    fig, axs = plt.subplots(2,2,  subplot_kw=dict(projection='3d'))

    for j, dist in enumerate(matlist):
        if MDStype == 'PCA':
            mds = PCA(n_components=3)
        if MDStype == 'MDS':
            mds = MDS(dissimilarity='precomputed',random_state=0, n_components=3)
        X_transform = mds.fit_transform(dist)
        ax = axs[math.floor(j/2), j%2]
        ax.title.set_text('step: '+str(j+1))
        if j>1:
            ax.set_xlabel('Dimension 1')
        if j%2 == 0:
            ax.set_ylabel('Dimension 2')
        for i in range(len(testseqs)):
            if i in valset_idx:
                alph = 1
            else:
                alph = 0.2
            ax.scatter(X_transform[i,0], X_transform[i,1], X_transform[i,2], color = colors1[i], alpha = alph)
            ax.scatter(X_transform[i,0], X_transform[i,1],  X_transform[i,2],s=100, facecolors='none', edgecolors=colors2[i], alpha = alph)

    plt.suptitle('2D-'+MDStype+': '+title)
    fig.legend(handles=legend_elements,  loc='center left', bbox_to_anchor=(1, 0.5)) 
    
def MDS_plot_prims(meanRDM, testseqs, MDStype = 'MDS', title = '', plotlines=True, rand_state=0, min_dim=0):
    
    plt.rcParams['figure.figsize'] = 6, 6
    fig, ax = plt.subplots()

    if MDStype == 'PCA':
        mds = PCA(n_components=3)
    if MDStype == 'MDS':
        mds = MDS(dissimilarity='precomputed',random_state=rand_state, n_components=3)

    X_transform = mds.fit_transform(meanRDM[0])
    ax.title.set_text('step: '+str(step_num))
    for i in range(16):
        ax.scatter(X_transform[i,min_dim], X_transform[i,min_dim+1], marker=7, color = colors1[i], s=180)
        ax.scatter(X_transform[i,min_dim], X_transform[i,min_dim+1], marker=6, color = colors2[i], s=180)
    for j in range(16,len(testseqs)):
        ax.plot([X_transform[j,min_dim]], [X_transform[j,min_dim+1]], marker=7, color=colors1[j], markersize = 16)
        ax.plot([X_transform[j,min_dim]], [X_transform[j,min_dim+1]], marker='_', color = colors2[j], markersize = 16,\
               markeredgewidth=3)
    if plotlines:
        for k in range(4):
            ax.plot([X_transform[4*k,0], X_transform[4*k+3,0]], [X_transform[4*k,1], X_transform[4*k+3,1]], color = colors2[k])
            ax.plot([X_transform[k,0], X_transform[12+k,0]], [X_transform[k,1], X_transform[12 + k,1]], color = colors2[k])
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    plt.suptitle('2D-'+MDStype+': '+title)
    fig.legend(handles=legend_elements,  loc='center left', bbox_to_anchor=(1, 0.5))
    
    
#### Plotting preferences

## fontsize
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
 
