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

reco_sense = np.load(folder + 'sl5-reco-sense.npy') 
gtim = np.load(folder + 'sl5-reco-reference.npy') # 

selected_idxs = [0, 8, 20] # 
referencias = gtim[...,selected_idxs]

fig, axs = plt.subplots(1,3, figsize=(9,3))
for i, ax in enumerate(axs): 
    ax.imshow(np.abs(referencias[...,i]), cmap='bone')
    ax.set_title(f'Imagen {i+1}')

# %% 
# partiría intentando hacer un autoencoder con la UNet 

key = random.PRNGKey(0)
key_space, key_params, key_eval, key_train = random.split(key, 4)

random_features = 32 # 32 en el paper, es muy distinto a usar 1? 
net = UNet(2, dropout_rate=0.1, encoding_features=32, skip_features=4, upsampling_method='nearest', levels=4, output_features=2)
noise_scaling = 1.0
# input_noise = noise_scaling * random.normal(key_space, (2,256,256,random_features)) # (1, nx, ny, 32)

input_im = np.moveaxis(reco_sense, -1, 0) # (frames, nx, ny)
input_im = np.stack((np.real(input_im), np.imag(input_im)), axis=-1) # (frames, nx, ny, 2)
input_im = input_im / np.percentile(np.abs(input_im), 99)

params = net.init(key_params, input_im[:2], training=False)

# %%

from inrmri.basic_nn import mse 

def to_complex(im):
    """
    - im.shape (...,2)
    - output: (...)
    """
    im = im[...,0] + 1j * im[...,1] # (frames, nx, ny)
    return im 

def get_im(params, frames_idx, key, training:bool): 
    out_im, updates = net.apply(params, input_im[frames_idx], training=training, rngs={'dropout':key}, mutable=['batch_stats']) # (frames, nx, ny, 2) 
    out_im = to_complex(out_im) # (frames, nx, ny)
    return out_im, updates 

def loss(params, X, Y, key): 
    new_out_im, updates = get_im(params, X, key, training=True) # (frames, nx, ny)
    loss_val = mse(new_out_im, to_complex(input_im[X]))
    return loss_val, updates 

# %%

learning_rate = 2e-3
optimizer = OptimizerWithExtraState(optax.adam(learning_rate))

total_kiter = 1 # 1000 iteraciones 
NFRAMES = reco_sense.shape[-1]
dummy_train_X = np.arange(NFRAMES)
dummy_train_Y = np.arange(NFRAMES)

results = train_with_updates(jit(loss), dummy_train_X, dummy_train_Y, params, optimizer, key_train, nIter = total_kiter * 1000 + 1, batch_size = 1)

# %% 

final_param = results['param_history'][f'param-{total_kiter}']
out, _ = get_im(final_param, np.array([8,15]), key, training=False)

fig, axs = plt.subplots(1,3, figsize=(9,3))
for i, ax in enumerate(axs): 
    ax.imshow(np.abs(out[i]), cmap='bone')
    ax.set_title(f'Imagen {i+1}')