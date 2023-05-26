import torch
import pickle
from cryptic_rnn import *

def MDS_plot_prims(meanRDM, testseqs, MDStype = 'MDS', title = '', min_dim = 0, step_num = 4, plotlines=True, rand_state = 0):
    
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
    plt.savefig('figures/mds'+ arch + affix +'_lines.png')

def MDS_plot(matlist, testseqs, trainseqs, MDStype = 'MDS', title = '', min_dim = 0, rand_state = 0, plotlines=False):
    
    plt.rcParams['figure.figsize'] = 9, 3
    fig, axs = plt.subplots(1,4)

    for j, dist in enumerate(matlist[1:]):
        if MDStype == 'PCA':
            mds = PCA(n_components=3)
        if MDStype == 'MDS':
            mds = MDS(dissimilarity='precomputed',random_state=rand_state, n_components=3)

        X_transform = mds.fit_transform(dist)
        ax = axs[j]
        ax.title.set_text('step: '+str(j+2))
        ax.set_xlabel('Dimension 1')
        if j == 0:
            ax.set_ylabel('Dimension 2')
        else:
            ax.set_yticks([])
        
        for i in range(len(testseqs)):
            alph = 1
            ax.scatter(X_transform[i,min_dim], X_transform[i,min_dim+1], marker=7, s=100, color = colors1[i], alpha = alph)
            ax.scatter(X_transform[i,min_dim], X_transform[i,min_dim+1], marker=6, s=100, color=colors2[i], alpha = alph)
            

    plt.suptitle('2D-'+MDStype+': '+title, y=1.05)
    fig.legend(handles=legend_elements,  loc='center left', bbox_to_anchor=(1, 0.5)) 
    plt.savefig('figures/mds'+ arch + affix +'.png')


arch = '_20' #how many hidden units
affix = '_b' #plot primitive or balanced
mod_names = 'final_mod' + affix #'init_mod_b', 'best_mod_b', 'final_mod_b', 'init_mod_p', 'best_mod_p', 'final_mod_p'

# read in models & losses
d_models = torch.load('results/2seqs_res'+ arch +'_1000_modelonly.pt')
with open('results/2seqs_res'+ arch +'_1000_losses.pkl', 'rb') as f:
    data = pickle.load(f)

n_losses, n_epochs, n_sim = data.shape

mod_config = d_models['config_model']
cue_dict = d_models['cue_dict'][0]

trainseqs = d_models['train' + affix][0].dataset
testseqs = d_models['test'][0]
testseq_all = torch.utils.data.ConcatDataset([trainseqs, testseqs])

#for each model reinstantiate the model with the weights
rdms = [[] for _ in range(5)]
for i_sim in range(len(d_models[mod_names])):
    state_dict = d_models[mod_names][i_sim]
    model = OneStepRNN(mod_config['input_size'], mod_config['output_size'], 
                    mod_config['hidden_size'], mod_config['num_layers'], mod_config['xavier_gain'])
    model.load_state_dict(state_dict)

    # change testsequence labels to fit this model's dict
    new_testseqs = []
    for datset in [d_models['train_b'][0].dataset, d_models['test'][0].dataset]:
        datset.recompute_label(d_models['cue_dict'][i_sim])
        new_testseqs.append(datset)

    testset = torch.utils.data.ConcatDataset(new_testseqs)
    testset = DataLoader(testset, batch_size=1, shuffle=False)
    
    # get activations for primitive trained model
    hiddens_p, trials = get_reps(model, [testset], mod_config['hidden_size']) 
    for h in range(5): #5 steps in sequencd
        hid_vals = np.array([hid[h,:] for hid in hiddens_p])
        rep_mat = euclidean_distances(hid_vals)
        rdms[h].append(rep_mat)

matlist = [np.array(d).mean(axis = 0) for d in rdms]

MDS_plot(matlist, testset, [], MDStype = 'MDS', title='Balanced', rand_state=48)
