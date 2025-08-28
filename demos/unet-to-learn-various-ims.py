# Reco de los datos cortos del Rafa (3 slices de 1.5s cada una)
# %%

import jax.numpy as np 
from jax import random, jit 
from inrmri.dip import UNet 
from inrmri.advanced_training import OptimizerWithExtraState, train_with_updates
import optax 
import matplotlib.pyplot as plt 

folder = '/mnt/storage/IPRE24-2-JC/' # cambiarlo al tuyo, debería terminar en / 

# %%
# ------------------------------------------------------- # 
# Cargar los datos disponibles 
# ------------------------------------------------------- # 

gtim = np.load(folder + 'sl5-reco-reference.npy') # 

selected_idxs = [0, 8, 20] # 
referencias = gtim[...,selected_idxs]

fig, axs = plt.subplots(1,3, figsize=(9,3))
for i, ax in enumerate(axs): 
    ax.imshow(np.abs(referencias[...,i]), cmap='bone')
    ax.set_title(f'Imagen {i+1}')

# %% 

key = random.PRNGKey(0)
key_space, key_params, key_eval, key_train = random.split(key, 4)

n_output_ims = len(selected_idxs)
random_features = 32 # 32 en el paper, es muy distinto a usar 1? 
net = UNet(2, dropout_rate=0.1, encoding_features=128, skip_features=4, upsampling_method='nearest', levels=4, output_features=n_output_ims*2)
noise_scaling = 1.0
input_noise = noise_scaling * random.normal(key_space, (2,256,256,random_features)) # (1, nx, ny, 32)

params = net.init(key_params, input_noise, training=False)

# %%

from inrmri.basic_nn import mse 

def get_im(params, key, training:bool): 
    out_im, updates = net.apply(params, input_noise, training=training, rngs={'dropout':key}, mutable=['batch_stats'])
        
    new_shape = out_im.shape[:3] + (n_output_ims,2)
    new_out_im = np.reshape(out_im, new_shape) # (1, nx, ny, 6) -> (1, nx, ny,3,2)
    new_out_im = new_out_im[0,...,0] + 1j * new_out_im[0,...,1] # (nx, ny, 3)
    return new_out_im, updates 

def loss(params, X, Y, key): 
    new_out_im, updates = get_im(params, key, training=True)
    loss_val = mse(new_out_im, referencias)
    return loss_val, updates 

loss(params, 0, 0, key)[0]

# %%

learning_rate = 2e-3
optimizer = OptimizerWithExtraState(optax.adam(learning_rate))

total_kiter = 1 # 1000 iteraciones 
dummy_train_X = np.arange(1)
dummy_train_Y = np.arange(1)

results = train_with_updates(jit(loss), dummy_train_X, dummy_train_Y, params, optimizer, key_train, nIter = total_kiter * 1000 + 1, batch_size = 1)

# %% 

final_param = results['param_history'][f'param-{total_kiter}']
out, _ = get_im(final_param, key, training=False)

fig, axs = plt.subplots(1,3, figsize=(9,3))
for i, ax in enumerate(axs): 
    ax.imshow(np.abs(out[...,i]), cmap='bone')
    ax.set_title(f'Imagen {i+1}')