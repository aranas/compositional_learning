# imports

import torch
import matplotlib.pyplot as plt
from cryptic_rnn import *
from run_models_functions import *
from sklearn.metrics import r2_score
from sklearn.manifold import MDS


# functions
def plot_predcorr(config, fname, model,test_data, model_config,title, cond = 'b'):
    plt.figure()
    r2, _, _ = predcorr(config, model, test_data, model_config['hidden_size'], plot_corr =True)
    
    title = title + 'balanced' if cond == 'b' else title + 'primitive'
    plt.title(title + '; r^2 = '+ str(round(r2, 3)))
    
    plt.savefig(f'figures/predcorr_{fname}_{cond}.png')

def plot_loss(data, fname, cond = 'b', title='',  colors=['green', 'yellow', 'red']):
    data = data.sel(loss_type=data['loss_type'].str.endswith(cond))

    fig, ax = plt.subplots()
    for i, level in enumerate(data.coords['loss_type'].values):
        level_data = data.sel(loss_type=level)
        mean = level_data.mean(dim='sim')
        std = level_data.std(dim='sim')
        mean.plot.line(ax=ax, label=level, color = colors[i])
        ax.fill_between(mean['epoch'], mean-std, mean+std, alpha=0.3,  facecolor = colors[i])
    plt.legend()
    
    fig.suptitle(title, fontsize=10)
    plt.savefig(f'figures/loss_{fname}_{cond}.png')

def plot_rmse_setsize(fname):

    loss_p = []
    loss_b = []
    for len_seq in range(1,4):
        fname = f'{len_seq}{fname[1:]}'
        with open(f'results/{fname}_losses.pkl', 'rb') as f:
            data = pickle.load(f)
    
        n_loss, n_epochs, n_sim = data.shape
        #select test_loss_p from data from final epoch
        loss_p.append(data.sel(loss_type='test_loss_p', epoch=n_epochs-1).to_numpy()[:16])
        loss_b.append(data.sel(loss_type='test_loss_b', epoch=n_epochs-1).to_numpy()[:16])
    
    
    sqrt_lp = np.sqrt(np.stack(loss_p))
    sqrt_lb = np.sqrt(np.stack(loss_b))

    # Find mean and standard deviation for each curriculim
    means_p = np.array(sqrt_lp).mean(axis=1)
    std_p = np.array(sqrt_lp).std(axis=1)
    means_b = np.array(sqrt_lb).mean(axis=1)
    std_b = np.array(sqrt_lb).std(axis=1)  

    j = 0
    labels = ['Balanced','Primitive']
    colors = ['#00A7E1', '#F17720']

    fig, ax = plt.subplots(figsize = (7,3))
    xpos = np.arange(1, means_p.shape[0]+1)
    ax.plot(xpos, means_b, label = labels[0], color=colors[0])
    ax.fill_between(xpos, means_b + std_b, means_b - std_b, color=colors[0], alpha=0.2)

    ax.plot(xpos, means_p, label = labels[1], color=colors[1])
    ax.fill_between(xpos, means_p + std_p, means_p - std_p, color=colors[1], alpha=0.2)


    ax.set_xlabel('Base training trials')
    ax.set_ylabel('RMSE')
    ax.title.set_text('RMSE against number of base training trails')
    plt.legend(loc='upper right')
    plt.show()

def get_hidden_reps(models, config, mod_names = 'final_mod'):
    n_sequence_steps = 5

    rdms = [[] for _ in range(n_sequence_steps)]
    for i_sim in range(len(models[mod_names])):
        #reinstantiate the model with the weights
        state_dict = models[mod_names][i_sim]
        model = OneStepRNN(config['input_size'], config['output_size'], 
                        config['hidden_size'], config['num_layers'], config['xavier_gain'])
        model.load_state_dict(state_dict)

        # change testsequence labels to fit each model's dict
        new_testseqs = []
        for datset in [models['train_b'][0].dataset, models['test'][0].dataset]:
            datset.recompute_label(models['cue_dict'][i_sim])
            datset.sort_by_label(sort='asc')
            new_testseqs.append(datset)

        testset = torch.utils.data.ConcatDataset(new_testseqs)
        testset = DataLoader(testset, batch_size=1, shuffle=False)
        
        # get activations for primitive trained model
        hiddens_p, _ = get_reps(model, [testset], config['hidden_size']) 

        #compute distance matrix
        for h in range(n_sequence_steps): #5 steps in sequence
            hid_vals = np.array([hid[h,:] for hid in hiddens_p])
            rep_mat = euclidean_distances(hid_vals)
            rdms[h].append(rep_mat)

    matlist = [np.array(d).mean(axis = 0) for d in rdms] #average over models

    return matlist, testset

def MDS_plot_prims(fname, meanRDM, testseqs, MDStype = 'MDS', title = '', min_dim = 0, step_num = 4, plotlines=True, rand_state = 0):
    plt.rcParams['figure.figsize'] = 6, 6
    fig, ax = plt.subplots()

    if MDStype == 'PCA':
        mds = PCA(n_components=3)
    if MDStype == 'MDS':
        mds = MDS(dissimilarity='precomputed',random_state=rand_state, n_components=3)

    X_transform = mds.fit_transform(meanRDM[0])
    ax.title.set_text(f'step: {str(step_num)}')
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
    plt.suptitle(f'2D-{MDStype}: {title}')
    fig.legend(handles=legend_elements,  loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'figures/mds_{fname}_lines.png')

def MDS_plot(fname, matlist, testseqs, trainseqs, MDStype = 'MDS', min_dim = 0, rand_state = 0, plotlines=False, affix='b'):
    
    title = 'balanced' if affix == 'b' else 'primitive'

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
    plt.savefig(f'figures/mds_{fname}_{affix}.png')

def get_ground_truth_correlation(models, config, cond = 'final_mod_b'):
    all_r2 = []
    n_sims = len(models[cond])
    for ix, model_dict in enumerate(models[cond]):
        model = OneStepRNN(config['input_size'], config['output_size'], 
                        config['hidden_size'], config['num_layers'], config['xavier_gain'])
        model.load_state_dict(model_dict)

        df = test_preds(model, [models['test'][ix]], config['hidden_size'])
        all_r2.append(r2_score(df['pred'],df['label']))

    r2_mean = np.mean(all_r2)
    r2_sterr = np.std(all_r2)/math.sqrt(n_sims)

    return r2_mean, r2_sterr

def main():
    fname = '2seqs_2_500'
    # read in models & losses
    d_models = torch.load(f'results/{fname}_modelonly.pt')
    with open(f'results/{fname}_losses.pkl', 'rb') as f:
        data = pickle.load(f)
    # get dimensions
    config_model = d_models['config_model']

    plot_loss(data, fname, cond = 'b', colors = ['green', 'orange', 'red'], title = 'balanced -no primitives')
    plot_loss(data, fname, cond = 'p', colors = ['green', 'orange', 'red'], title = 'with primitives')

    plot_predcorr(config_model, fname, d_models['final_mod_b'], d_models['test'], config_model, title = 'final model - ', cond = 'b')
    plot_predcorr(config_model, fname, d_models['final_mod_p'], d_models['test'], config_model, title = 'finalmodel - ',  cond = 'p')

    matlist, testset = get_hidden_reps(d_models, config_model, 'final_mod_p')
    MDS_plot(fname, matlist, testset, [], MDStype = 'MDS', rand_state=48, affix='p')
    
    matlist, testset = get_hidden_reps(d_models, config_model, 'final_mod_b')
    MDS_plot(fname, matlist, testset, [], MDStype = 'MDS', rand_state=48, affix = 'b')

    vals = np.empty((0,4))
    for len_seq in range(1,4):
        #load model
        fname = f'{len_seq}{fname[1:]}'
        d_models = torch.load(f'results/{fname}_modelonly.pt')
        config = d_models['config_model']

        #correlate predictions with gorund truth
        r2_b, sterr_b = get_ground_truth_correlation(d_models, config, cond = 'final_mod_b')
        r2_p, sterr_p = get_ground_truth_correlation(d_models, config, cond = 'final_mod_p')

        #collect across set sizes
        set_vals = [r2_b, r2_p, sterr_b, sterr_p]
        vals = np.vstack([vals, set_vals])

    #plot
    j = 0
    labels = ['Balanced','Primitive']
    colors = ['#00A7E1', '#F17720']
    plt.figure(figsize = (7,3))
    xpos = np.arange(1, vals.shape[0]+1)
    Nn = int(vals.shape[1]/2)
    for j in range(Nn):
        plt.plot(xpos, vals[:,j], label = labels[j], color=colors[j])
        plt.fill_between(xpos, vals[:,j] + vals[:,j+Nn], vals[:,j] - vals[:,j+Nn], color=colors[j], alpha=0.2)
    plt.legend(loc='lower right')
    plt.xlabel('Base training trials')
    plt.ylabel('$R^2$')
    plt.title('Prediction $R^2$ against number of base training trails')
    plt.show()

if __name__ == "__main__":
    main()

