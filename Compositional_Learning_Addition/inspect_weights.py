import numpy as np
import matplotlib.pyplot as plt

from cryptic_rnn import *
from run_models_functions import *

# read in models & losses
arch = '_1'
d_models = torch.load('results/2seqs_res' + arch+ '_2000_trainlarge_modelonly.pt')
with open('results/2seqs_res' + arch+ '_2000_trainlarge_losses.pkl', 'rb') as f:
    data = pickle.load(f)

mod_config = d_models['config_model']

# get dimensions
n_loss, n_epochs, n_sim = data.shape

# correlation between weights and input values for converged model weights, is on average higher for primitive compared to balanced models
# suggesting that the balanced model relies on the bias term to push towards the correct solution
weights_p = []
weights_b = []
inputs = []
for ix in range(len(d_models['final_mod_p'])):
    weights_p.append(d_models['final_mod_p'][ix]['input2hidden.weight'][:,2:6].detach().numpy())
    weights_b.append(d_models['final_mod_b'][ix]['input2hidden.weight'][:,2:6].detach().numpy())
    # append dict values as numpy array
    inputs.append(np.array(list(d_models['cue_dict'][ix].values())))

weights_p = np.stack(weights_p)
weights_b = np.stack(weights_b)
inputs = np.stack(inputs)
# for each row in weights_p get correlation with corresponding inputs row
corr_p = []
corr_b = []
for i in range(weights_p.shape[0]):
    corr_p.append(np.corrcoef(weights_p.squeeze()[i,:], inputs[i,:])[0,1])
    corr_b.append(np.corrcoef(weights_b.squeeze()[i,:], inputs[i,:])[0,1])


print('primitive model weights (mean): ' + str(np.mean(np.abs(corr_p))))
print('primitive model weights: (std)' + str(np.std(np.abs(corr_p))))
print('balanced model weights (mean): ' + str(np.mean(np.abs(corr_b))))
print('balanced model weights: (std)' + str(np.std(np.abs(corr_b))))

# plot correlation between testing loss and input value correlation for primitive and balanced models
plt.figure(figsize=(10,5))
#plt.scatter(np.abs(corr_p), data.sel(loss_type='test_loss_p',  epoch=n_epochs-1).to_numpy()
#            , color='blue', label='primitive')
plt.scatter(np.abs(corr_b), data.sel(loss_type='test_loss_b',  epoch=n_epochs-1).to_numpy()
            , color='red', label='balanced')
# add regression line through scatterplot
# get dimensions
n_loss, n_epochs, n_sim = data.shape
# select those with train loss of epoch lower than 1
sel = data.sel(loss_type=['test_loss_b'], epoch=n_epochs-1) < 600
sel_c = np.abs(corr_b) > 0.5
# get indices of those that are true for both losses
sel = sel.all(dim='loss_type')
# select data test_loss_b for simulations using mask sel
dat_sel = data.sel(loss_type='test_loss_b', epoch=n_epochs-1).where(sel).to_numpy()
#drop nan from array
dat_sel = dat_sel[~np.isnan(dat_sel)]
plt.plot(np.unique(np.abs(corr_b)), np.poly1d(np.polyfit(np.abs(np.array(corr_b)[sel]), dat_sel, 1))(np.unique(np.abs(corr_b))), color='blue')
plt.xlabel('correlation between weights and input values')
plt.ylabel('testing loss')
plt.legend()
plt.show

# save to results
plt.savefig('figures/corr_weights_value_representation'+ arch +'_balanced.png')

