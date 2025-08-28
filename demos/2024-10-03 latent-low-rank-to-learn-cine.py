# la idea es hacer como td-dip en la entrada
# t a mapnet, pero el mapnet da varios valores 
# esos combinan varios ruidos 
# %%

import jax.numpy as np 
from jax import random, jit 
from inrmri.dip import LatentLowRankTDDIP
from inrmri.coord_maps import circle_map 
from inrmri.basic_nn import mse 
from inrmri.advanced_training import OptimizerWithExtraState, train_with_updates
import optax 
key = random.key(0)
key_latent, key_init, key_train = random.split(key, 3)

# %%

latent_features = 8 
latent_shape = (8,8)
net = LatentLowRankTDDIP([16,16], latent_features, latent_shape, 64, 4)
latent_x = random.uniform(key_latent, latent_shape + (latent_features,))

gtim = np.load('/mnt/storage/CRUZ_THOMAS_data/GN/numpy/sl_5_Batch_R1.npy') # (nx, ny, frames)
gtim = np.moveaxis(gtim, -1,0)# (frames, nx, ny)
NFRAMES = gtim.shape[0]
ts = np.linspace(0,1,NFRAMES, endpoint=False)
params = net.init(key_init, circle_map(ts[:1]), latent_x, training=False)

# %%

def get_im(params, t_idx, key): 
    im, updates = net.apply(params, circle_map(ts[t_idx]), latent_x, training=True, rngs={'dropout':key}, mutable=['batch_stats'])
    im = im[...,0] + 1j * im[...,1] # (batch, nx, ny)
    return im, updates['batch_stats'] 

def loss(params, X, Y, key): 
    # X = t_idxs
    pred_im, updates = get_im(params, X, key) # batch_frame, nx, ny
    return mse(pred_im, Y), updates 

# %%

learning_rate = 1e-3 
optimizer = OptimizerWithExtraState(optax.adam(learning_rate))

total_kiter = 1
results = train_with_updates(jit(loss), np.arange(NFRAMES), gtim, params, optimizer, key_train, nIter = total_kiter * 1000 + 1, batch_size = 3)

# %%
import matplotlib.pyplot as plt 
from inrmri.basic_nn import mse 
from jax.lax import map as laxmap

selected_kiter = 1
final_param = results['param_history'][f'param-{selected_kiter}']
# plt.imshow(np.abs(y[...,3]))

ims = laxmap(lambda t_idxs:  get_im(final_param, t_idxs, key)[0], np.arange(30).reshape(3,10))
ims = ims.reshape((30,256,256))
ims = np.moveaxis(ims, 0, -1)
ims.shape
# %%

from inrmri.image_processor import reduce_FOV 
from inrmri.basic_plotting import full_halph_FOV_space_time 

full_halph_FOV_space_time([reduce_FOV(ims)], crop_ns=[35,35,40,40], saturation=0.75, frame=8)


