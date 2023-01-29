import itertools
import copy
import collections
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import pickle
import random
from sklearn.model_selection import train_test_split

######################
### Generate sequences
######################

convert_inputcue = {'X': 0,
                    'Y': 1,
                    'A': 2, 
                    'B': 3,
                    'C': 4,
                    'D': 5,
                    'E': 6, 
                    'F': 7,
                    'G': 8,
                    'H': 9
                    }

convert_operation = {'+': 10,
                     '*': 11,
                     '-': 12,
                     '%': 13}

default_cues = {'X': 0,
                'Y': 1,
                'A': 2, 
                'B': 3,
                'C': 5,
                'D': 7,
                'E': 1, 
                'F': 4,
                'G': 9,
                'H': 11}
          

def generate_trials(operators, input_ids, len_seq,\
                    init_values, rand, rep):
    
    ''' This function defines all possible permutations of initial value & sequence of input cues and operators.
    Args: 
        operators, input_ids, init_values: lists of operator, input cue and initial values
        len_seq: number of operation and input pairs per sequence
        replacement: whether an operation/input can be repeated in a sequence
    Returns:
        Output is an array of shape n X len_seq+1.
        Each row is one of n unique ordered permutations.
        First column indicates the initial value at t=0.
        Every following column contains a tuple, where the first position indicates
        the operator and the second position indicates the input cue.
        Final column indicates the outcome of the opperatioons on the initial value'''
    
    seq = []
    combi_operators = list(itertools.product(operators, repeat=len_seq))*rep
    if rand:
        for op in combi_operators:
            cue = random.choices(input_ids, k=2)
            seq.append([random.choice(init_values),
                        *zip(tuple(op), tuple(cue))]) #group per time point t
        
    else:
        combi_inputcue = list(itertools.product(input_ids, repeat=len_seq))
        for init in init_values:
            for cue in combi_inputcue:
                for op in combi_operators:
                    seq.append([init,
                                *zip(tuple(op), cue)]) #group per time point t

    return seq

def operate_op(currval, step_tuple, cue_dict):
    """ Function applies operations to input value
    Args:
        ...
    Returns:
        ...
    """
    nextval = cue_dict[step_tuple[1]]
    if step_tuple[0] == '+': # add
        currval = currval + nextval
    elif step_tuple[0] == '*': # multiply
        currval = currval * nextval
    elif step_tuple[0] == '-': # subtract
        currval = currval - nextval
    
    return currval

def calculate_output(step_tuple, cue_dict, bidmas):
    """ Function applies operations to input value
    Args:
        ...
    Returns:
        ...
    """
    if bidmas:
        calc_string = str(cue_dict[step_tuple[0]])
        for i in range(1,len(step_tuple)):
            calc_string = calc_string + step_tuple[i][0] + str(cue_dict[step_tuple[i][1]])
        curr_val = eval(calc_string)
    else:
        curr_val = cue_dict[step_tuple[0]]
        for i in range(1,len(step_tuple)):
            curr_val = operate_op(curr_val, step_tuple[i], cue_dict)
    return curr_val


def generate_sequences(operators, input_ids, len_seq, cue_dict = default_cues,\
                       init_values = list(range(1,6)), rand=False, rep = 1, bidmas = False):
    """ Function applies operations to input value
    Args:
        ...
    Returns:
        ...
    """
    all_trials = generate_trials(operators, input_ids, len_seq, init_values, rand, rep)
    for trial in all_trials:
        trial_output = calculate_output(trial, cue_dict, bidmas)
        trial.append(trial_output)
    
    return(all_trials)

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def pad_select(sequences, pos, padder):
    assert len(sequences[0][1:-1]) == len(pos), 'invalid position'
    pad_seqs = []
    for s in sequences: 
        step = s[1:-1]
        pad_trial = [padder]*3
        for i in range(len(pos)):
            pad_trial[pos[i]] = step[i]
        pad_seqs.append([s[0]] + pad_trial + [s[-1]])
    return pad_seqs


def pad_seqs_2step(sequences, padder=('+','X')):
    pos = [[0,1], [1,2], [0,2]]
    pad_seqs = []
    for s in sequences: 
        pad_trials = []
        step = s[1:-1]
        if len(step) == 1:
            for i in range(2):
                pad_trial = [padder]*2
                pad_trial[i] = step[0]
                pad_trials.append([s[0]] + pad_trial + [s[-1]])
            pad_seqs += pad_trials
        else:
            pad_seqs = sequences
    return pad_seqs     

def pad_seqs_1step(sequences, padder=('+','X')):
    pad_seqs = []
    for s in sequences: 
        pad_seqs.append([s[0]] + [padder]*2 + [default_cues[s[0]]])
    return pad_seqs

##################################################
## Transform data to rnn data
##################################################


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
        sequence = self.data[index,:].astype(np.float32)
        out_state = np.array(self.labels[index]).astype(np.float32)
        return sequence, out_state
    
    
def convert_seq2inputs(sequences, seq_len=5, stages = False, cont_out = True, num_classes=14):
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


def convert_seq2onehot(seq, stages, num_classes=14):
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
            if i==0:
                init = torch.tensor(convert_inputcue[t])
                init = torch.nn.functional.one_hot(init, num_classes=num_classes)
                trial_data.append(init)
                continue
            else:
                op = torch.tensor(convert_operation[t[0]])
                op = torch.nn.functional.one_hot(op, num_classes=num_classes)
                inputcue = torch.tensor(convert_inputcue[t[1]])
                inputcue = torch.nn.functional.one_hot(inputcue, num_classes=num_classes)
                trial_data.append(op)
                trial_data.append(inputcue)
        data.append(torch.stack(trial_data))

    data = torch.stack(data,dim=0) #combine into tensor of shape n_trials X n_time_steps X inputvector_size
    data = data.numpy()

    return data


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