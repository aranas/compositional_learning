import numpy as np
import matplotlib.pyplot as plt

from cryptic_rnn import *
from run_models_functions import *

def corr_weights2hidden_inputs(d_models, n_unit=0, verbose=False):
    '''
    calculate correlation between weights (from input to hidden units) and input values for converged model weights
    in: d_models (dict of lists of dicts)
    out: corr_p (list of floats), corr_b (list of floats)
    '''
    weights_p = []
    weights_b = []
    inputs = []
    for mod_p, mod_b, cuedict in zip(d_models['final_mod_p'],d_models['final_mod_b'],d_models['cue_dict']):
        weights_p.append(mod_p['input2hidden.weight'][n_unit,2:6].detach().numpy()) 
        weights_b.append(mod_b['input2hidden.weight'][n_unit,2:6].detach().numpy())
        # append dict values as numpy array
        inputs.append(np.array(list(cuedict.values())))

    weights_p = np.stack(weights_p)
    weights_b = np.stack(weights_b)
    inputs = np.stack(inputs)

    # for each row in weights_p get correlation with corresponding inputs row
    corr_p = []
    corr_b = []

    for i in range(weights_p.shape[0]):
        corr_p.append(np.corrcoef(weights_p.squeeze()[i,:], inputs[i,:])[0,1])
        corr_b.append(np.corrcoef(weights_b.squeeze()[i,:], inputs[i,:])[0,1])

    if verbose:
        mean_corr_p = np.mean(np.abs(corr_p))
        std_corr_p = np.std(np.abs(corr_p))
        mean_corr_b = np.mean(np.abs(corr_b))
        std_corr_b = np.std(np.abs(corr_b))

        print(f'primitive model weights (mean): {mean_corr_p}')
        print(f'primitive model weights: (std) {std_corr_p}')
        print(f'balanced model weights (mean): {mean_corr_b}')
        print(f'balanced model weights: (std) {std_corr_b}')

    return np.abs(corr_p), np.abs(corr_b)

def main():

    # read in models & losses
    fname = '3seqs_2_500'
    d_models = torch.load(f'results/{fname}_modelonly.pt')
    with open(f'results/{fname}_losses.pkl', 'rb') as f:
        data = pickle.load(f)
    n_loss, n_epochs, n_sim = data.shape
    mod_config = d_models['config_model']

    n_hid = d_models['config_model']['hidden_size']


    # plot correlation between testing loss and input value correlation for primitive and balanced models
    # but make two plots, one for each model type
    fig, axs = plt.subplots(n_hid, 2, figsize=(15, 5))

    for i,ax_x in enumerate(axs):

        # correlation between weights and input values for converged model weights, is on average higher for primitive compared to balanced models
        # suggesting that the balanced model relies on the bias term to push towards the correct solution (this solution won't extrapolate well)
        corr_p, corr_b = corr_weights2hidden_inputs(d_models, n_unit=i, verbose=False)
        # get indices of 3 smallest outliers
        outliers_p = np.where(corr_p < np.mean(corr_p) - np.std(corr_p))[0][:3]
        outliers_b = np.where(corr_b < np.mean(corr_b) - np.std(corr_b))[0][:3]

        #check if ax_x is iterable, if not wrap in list
        if not hasattr(ax_x, "__iter__"):
            ax_x = [ax_x]

        # loop through each subplot and condition
        for ax, cond, corr, outlier in zip(ax_x,['primitive','balanced'],[corr_p,corr_b],[outliers_p,outliers_b]):
            
            test_loss = data.sel(loss_type=f'test_loss_{cond[0]}',  epoch=n_epochs-1).to_numpy()
            test_loss = test_loss[np.abs(corr) < (np.mean(corr) + np.std(corr))]
            corr = corr[np.abs(corr) < np.mean(np.abs(corr)) + np.std(np.abs(corr))]

            #exclude outliers
            test_loss = np.delete(test_loss, outlier)
            corr = np.delete(corr, outlier)

            #exclude most extreme values

            ax.scatter(corr, test_loss
                    , color='red', label=cond)
            
            # add regression line through scatterplot

            ax.plot(np.unique(corr), np.poly1d(np.polyfit(corr, test_loss, 1))(np.unique(corr)), color='blue')
            ax.set_xlabel('correlation between weights and input values')
            ax.set_ylabel('testing loss')
            ax.set_title(cond)
            ax.legend()

        # save to results
        plt.savefig(f'figures/weightCorr_{fname}.png')

if __name__ == "__main__":
    if False:
        main()