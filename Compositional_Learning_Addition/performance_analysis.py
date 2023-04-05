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

from cryptic_rnn import *

############### Functions: R squared comparison ###############

def get_r2_and_preds(mods, tests, hidden_size=20):
    dfs = []
    r2s = []
    for i in range(len(mods)):
        df = test_preds(mods[i], [tests[i]], hidden_size)
        dfs.append(df) # append data frame of labels and predictions
        r2s.append(r2_score(df['pred'], df['label'])) # individual model r2 score
    all_dfs = pd.concat(dfs)
    all_r2s = r2_score(all_dfs['pred'],all_dfs['label'])
    
    return {'ind_dfs': dfs, 'all_dfs': all_dfs, 'ind_r2s': r2s, 'all_r2s': all_r2s}

############### RMSE ############################

def test_modloss(model, testdata, criterion, hidden_size=20):
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


def extract_MSELoss(res1, hidden_size=20):
    tests = res1['tests']
    mods_b = res1['mods_b']
    mods_p = res1['mods_p']
    criterion = nn.MSELoss()
    mse_b = []
    mse_p = []
    for i in range(len(mods_b)):
        mse_b.append(test_modloss(mods_b[i], [tests[i]], criterion, hidden_size)[0]) 
        mse_p.append(test_modloss(mods_p[i], [tests[i]], criterion, hidden_size)[0]) 
        
    return {'mse_b': mse_b, 'mse_p': mse_p}

def testset_MSELoss(res1, testsets, hidden_size=20):
    mods_b = res1['mods_b']
    mods_p = res1['mods_p']
    criterion = nn.MSELoss()
    mse_b = []
    mse_p = []
    for i in range(len(mods_b)):
        mse_b.append(test_modloss(mods_b[i], [testsets[i]], criterion, hidden_size)[0]) 
        mse_p.append(test_modloss(mods_p[i], [testsets[i]], criterion, hidden_size)[0]) 
    return {'mse_b': mse_b, 'mse_p': mse_p}

criterion = nn.MSELoss()

def ind_losses(mods, test_seqs, hidden_size=20, num_classes=22):
    losses = []
    for i, mod in enumerate(mods):
        cuedict = cue_dicts[i]
        testseqs = change_dict(test_seqs, cuedict)
        test_inputs = convert_seq2inputs(testseqs, num_classes=num_classes, seq_len=5)
        testset = DataLoader(test_inputs, batch_size=batchsize, shuffle=True)
        losses.append(test_loss(mod, [testset], criterion)[0])
    return losses
