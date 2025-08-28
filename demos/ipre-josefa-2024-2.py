# Reco de los datos cortos del Rafa (3 slices de 1.5s cada una)
# %%

from scipy.io import loadmat 
import numpy as onp 
import matplotlib.pyplot as plt 
from inrmri.image_processor import reduce_FOV
from inrmri.basic_plotting import full_halph_FOV_space_time
from inrmri.bart import bart_acquisition_from_files, make_grasp_name, read_csmap, bart_read, bart_write

folder = '/mnt/storage/IPRE24-2-JC/' # cambiarlo al tuyo, debería terminar en / 

# %%
# ------------------------------------------------------- # 
# Cargar los datos disponibles 
# ------------------------------------------------------- # 

import jax.numpy as np 
from inrmri.radial_acquisitions import RadialAcquisitions

print("Loading data...")

data = np.load(folder + 'sl5-data.npy')
trajs = np.load(folder + 'sl5-traj.npy')
radial_acquisition = RadialAcquisitions(trajs, data)

csmap = np.load(folder + 'sl5-csmap.npy')

# Reconstrucción con iterative sense (la reco más básica cuando se conocen las bobinas)
reco_sense = np.load(folder + 'sl5-reco-sense.npy') 

# Reconstrucción con GRASP (una reco que además de las bobinas incluye regularización (TV en tiempo))
reco_grasp = np.load(folder + 'sl5-reco-grasp.npy') 

print("Data loaded!")
print(radial_acquisition)
print(f"csmap has shape {csmap.shape}: {csmap.shape[0]} sensitivity maps of size {csmap.shape[1:3]}")
print(f"reco_sense has shape: {reco_sense.shape}: {reco_sense.shape[-1]} frames of size {reco_sense.shape[:2]}")
print(f"reco_grasp has shape: {reco_grasp.shape}: {reco_grasp.shape[-1]} frames of size {reco_grasp.shape[:2]}")


# %%
# ------------------------------------------------------- # 
# Exploración de los datos 
# ------------------------------------------------------- # 

# Visualizar las trayectorias 

selected_frame_1 = 21
selected_frame_2 = 3

plt.figure(figsize=(6,3))
read_out_jump = 10 # mostrar 1 de cada 10 puntos de read out 
plt.subplot(121)
plt.scatter(
    radial_acquisition.trajs[selected_frame_1, :, ::read_out_jump, 0],
    radial_acquisition.trajs[selected_frame_1, :, ::read_out_jump, 1],
    s=2
)
plt.subplot(122)
plt.scatter(
    radial_acquisition.trajs[selected_frame_2, :, ::read_out_jump, 0],
    radial_acquisition.trajs[selected_frame_2, :, ::read_out_jump, 1],
    s=2
)

# %%
# Visualizar las bobinas (en la misma escala) 

plt.figure(figsize=(4*3, 7*3))
for i in range(28):
    plt.subplot(7,4,i+1)
    plt.imshow(np.abs(csmap[i]), vmax=np.abs(csmap).max(), cmap='bone')

# %%
# Visualizar la imagen "sensitivity weighted" (pesada por las bobinas)

selected_frame = 8 
plt.figure(figsize=(4*3, 7*3))
for i in range(28):
    plt.subplot(7,4,i+1)
    plt.imshow(np.abs(reco_grasp[...,selected_frame] * csmap[i]), vmax=np.abs(reco_grasp).max(), cmap='bone')


# %%
# Visualizar los perfiles temporales x-t de una imagen, en una zona cropeada 

from inrmri.basic_plotting import full_halph_FOV_space_time 
from inrmri.image_processor import reduce_FOV 

fig, axs = full_halph_FOV_space_time([reduce_FOV(reco_sense), reduce_FOV(reco_grasp)], crop_ns=[35,35,30,40], saturation=0.7, frame=10)
axs[0,0].set_title('SENSE')
axs[1,0].set_title('GRASP')

# %%
from jax import jit, vmap, random
import jax.numpy as np 

from inrmri.data_harvard import load_data
from inrmri.radon import calculate_angle 
from inrmri.utils import to_complex, is_inside_of_radial_lim, meshgrid_from_subdiv_autolims
from inrmri.radial_acquisitions import kFOV_limit_from_spoke_traj, check_correct_dataset
from inrmri.new_radon import ForwardRadonOperator

from inrmri.basic_nn import weighted_loss 

import optax 
from inrmri.advanced_training import train_with_updates, OptimizerWithExtraState

key = random.PRNGKey(0)
key_net, key_params, key_train = random.split(key,3) # keys for reproducibility 

print("Creating dataset...")
train_X, train_Y = radial_acquisition.generate_dataset()
train_Y = 100 * train_Y
 #para que quede en un orden de magnitud adecuado
# la salida de load_data está pensada para trabajar con INRs, por eso hago un poco más 
# de procesamiento después 
# train_X, train_Y, csmap, _, hollow_mask = load_data(config_data) 

check_correct_dataset(train_X)

spclim = kFOV_limit_from_spoke_traj(train_X[0,1:,:])
scalingparam = 1.

angles = vmap(calculate_angle)(train_X[:,1:])
times = train_X[:,0,0]
train_X = np.stack([angles, times], axis=-1) # sampled angles with their respective time 

# %% 

gtim = reco_grasp 
NFRAMES = gtim.shape[-1]
IMSHAPE = gtim.shape[:2]
N = IMSHAPE[0]

radon_operator = ForwardRadonOperator(csmap, spclim)

grid = meshgrid_from_subdiv_autolims(IMSHAPE)

def post_processing(im): # im.shape (px,py,nframes)
    is_inside = is_inside_of_radial_lim(grid, 1.)
    masks = is_inside
    return im * masks[:,:,None]

# %%
from inrmri.dip import TimeDependant_DIP_Net, helix_generator
from inrmri.dip import SpaceTime_LR_UNet
features = 64

ts = np.linspace(0,1,gtim.shape[-1], endpoint=False)

L = 16
net = SpaceTime_LR_UNet(NFRAMES, IMSHAPE, key_net, encoding_features=64, output_features=L, noise_scaling=0.1)
params = net.init_params(key_params)

# %%

# la idea ahora es usar la base temporal aprendida a partir de GRASP (Vh) como punto de partida
# para la red
# el primer paso es entrenar los parámetros de la red temporal para aprender la base 
from inrmri.basic_nn import mse 

def loss_temporal_basis(params_time, X_dummy, Y_dummy, key):
    y_time, batch_stats_time = net.train_eval_time(params_time, key, t_index_batch=None) # se entrenan todos los frames (t_index = None)
    y_time.shape # (1, frames, L)
    y_time = np.moveaxis(y_time[0], -1, 0) # (L, frames)
    lossval = mse(y_time, Vh[:L]) # comparar con las primeras L bases 
    return lossval, batch_stats_time

learning_rate = 1e-3
optimizer = OptimizerWithExtraState(optax.adam(learning_rate))

total_kiter = 1 # 1000 iteraciones 
train_X_dummy = np.arange(1)
train_Y_dummy = np.arange(1)

results = train_with_updates(jit(loss_temporal_basis), train_X_dummy, train_Y_dummy, net.split_params(params)[1], optimizer, key_train, nIter = total_kiter * 1000 + 1, batch_size = 1)

# %%

y_time, batch_stats_time = net.train_eval_time(results['param_history']['param-1'], key, t_index_batch=None) # se entrenan todos los frames (t_index = None)
y_time.shape 

plt.plot(np.abs(y_time[0, :, :3]), c='blue')
plt.plot(np.abs(Vh[:3, :]).transpose(), c='red')

# tal vez puedo agregar la ortonormalidad como una función de pérdida, eso podría guiar a la red sin ser excesivamente restrictivo 

# %%
from inrmri.new_radon import get_weight_freqs
from inrmri.regularizations import TV_space
from inrmri.linalg_utils import gram_schmidt
from inrmri.dip import to_complex, split_last_dim 

WEIGHT_FREQS = get_weight_freqs(N)
lmbda_tv = 0

def process_basis(y_base): 
    """
    - y_base: (1, ..., 2L) (float)
    output 
    - y_base_processed: (..., L) (complex) 
    """
    y_base_processed = to_complex(split_last_dim(y_base))[0,...,0] # (nx, ny, outfeatures//2)
    return y_base_processed

def ortonormalize_time_basis(y_time):
    """
    - y_time: (frames, L)
    
    output 
    - y_time_ort: (frames, L)
    """
    y_time = np.moveaxis(y_time, 0, -1) # (L, nframes)
    print("y_time.shape before gram schmidt: ", y_time.shape)
    y_time = gram_schmidt(y_time)
    y_time = np.moveaxis(y_time, 0, -1) # (nframes, L)
    return y_time

def combine_space_and_time(y_space, y_time): 
    """
    - y_space: (nx, ny, L)
    - y_time: (frames, L)
    
    output 
    - (frames, nx, ny)
    """
    ims = y_space[:,:,None,:] * y_time[None,None,:,:] # (nx, ny, nframes, outfeatures//2)
    ims = np.mean(ims, axis=-1) # (nx, ny, nframes)
    ims = np.moveaxis(ims, -1, 0) # (nframes, nx, ny)
    return ims 

weight_ort = 1. 

def loss(params, X, Y, key): 
    batch_size = X.shape[0]
    alphas, times = X[:,0], X[:,1]
    t_idx = np.int32(times * NFRAMES)
    y_space, y_time, update = net.train_basis_and_update(params, key, None)
    print("y_space.shape, y_time.shape out of net: ", y_space.shape, y_time.shape)
    y_space = process_basis(y_space) # (nx, ny, L)
    y_time  = process_basis(y_time) # (frames, L)

    y_time_ort = ortonormalize_time_basis(y_time)
    ims = combine_space_and_time(y_space, y_time[t_idx])
    spoke_radon_kspace = radon_operator.radon_transform(ims, alphas) # (batch-frame-alpha, ncoils, N)

    Y = Y[...,0] # (frames, cmap, nx)
    print("shapes: spoke ", spoke_radon_kspace.shape, " , Y ", Y.shape)
    loss_value = weighted_loss(spoke_radon_kspace, Y, (1. + WEIGHT_FREQS)[None,None,:])
    loss_ort = mse(y_time_ort, y_time)
    # tv_val = TV_space(np.moveaxis(ims, 0, -1))
    return loss_value + weight_ort * loss_ort, update 

# %% Training 

learning_rate = 1e-3
optimizer = OptimizerWithExtraState(optax.adam(learning_rate))

total_kiter = 1 # 1000 iteraciones 

config_training = {
    'learning_rate': learning_rate,
    'total_kiter': total_kiter,
    'key_train': key_train
}
# para saber cuanta memoria realmente esta usando jax hay que desactivar la preallocation por default 
results = train_with_updates(jit(loss), train_X, train_Y, params, optimizer, key_train, nIter = total_kiter * 1000 + 1, batch_size = 3)

# %% Examining results 

final_param = results['param_history'][f'param-{total_kiter}']

# %%
def loss_with_basis_smoothing(params, X, Y, key): 
    batch_size = X.shape[0]
    alphas, times = X[:,0], X[:,1]
    t_idx = np.int32(times * NFRAMES)
    ims, update = net.train_forward_with_exponential_reg(params, key, t_idx, 0.99)
    spoke_radon_kspace = radon_operator.radon_transform(ims, alphas) # (batch-frame-alpha, ncoils, N)

    Y = Y[...,0] # (frames, cmap, nx)
    print("shapes: spoke ", spoke_radon_kspace.shape, " , Y ", Y.shape)
    loss_value = weighted_loss(spoke_radon_kspace, Y, (1. + WEIGHT_FREQS)[None,None,:])
    tv_val = TV_space(np.moveaxis(ims, 0, -1))
    return loss_value + lmbda_tv*tv_val, update 

yspace, ytime, _ = net.train_basis_and_update(final_param, key, None)
params_and_basis = {**final_param, 'basis':(yspace, ytime)}

final_k_iters = 1
results2 = train_with_updates(jit(loss_with_basis_smoothing), train_X, train_Y, params_and_basis, optimizer, key_train, nIter = final_k_iters * 1000 + 1, batch_size = 3)
# %%

# evaluar en modo training  
from jax.lax import map as laxmap 

# final_param_smoothed = results2['param_history'][f'param-{final_k_iters}']

# predim = net.inference_image(final_param, np.arange(NFRAMES))
# y_space, y_time, _= net.train_basis_and_update(final_param, key, np.arange(NFRAMES))
y_space, y_time = net.inference_basis(final_param, np.arange(NFRAMES))
predim = ortonorm_im_from_net_basis(y_space, y_time)

predim = np.moveaxis(predim, 0, -1)
predim = post_processing(predim) / np.abs(predim).max()
predim.shape

# %%

from inrmri.basic_plotting import full_halph_FOV_space_time
from inrmri.image_processor import BeforeLinRegNormalizer, reduce_crop_abs
import matplotlib.pyplot as plt 

crop_ns = [35,30,20,38]
gtim = np.load(folder + 'sl5-reco-reference.npy') # 
improc = BeforeLinRegNormalizer(gtim, [0,0,0,0]) # normaliza todas las imagenes a la escala de gtim 

gt_proc = improc.process(gtim)

def final_processing(im):
    im = improc.process(im)
    im = onp.roll(im, 15, axis=-1) # 15 es el argmin del error con gastao luego de probar todos los rolls 
    return im 

# el [1:] es porque gtim solo tiene 30 frames (no 31)
pred_proc = final_processing(predim[...,1:]) 
sense_proc = final_processing(reco_sense[...,1:])
grasp_proc = final_processing(reco_grasp[...,1:])

frame = 8 # 8, 23 
vmax = np.abs(gtim).max()
recos = [gt_proc, pred_proc, grasp_proc, sense_proc]
titles = ['MCCINE (Gastao)', 'LR-DIP', 'GRASP', 'SENSE']
fig, axs = full_halph_FOV_space_time(recos, crop_ns, frame = frame,saturation=0.5, vmax=vmax)
for ax, title in zip(axs, titles):
    ax[0].set_title(title)

# %%
from inrmri.dip import split_last_dim 

yspace, ytime = net.inference_basis(final_param, np.arange(31))
yspace = to_complex(split_last_dim(yspace))[0,...,0] 
ytime = to_complex(split_last_dim(ytime))[0,...,0] 

yspace.shape, ytime.shape

# %%

fig, axs = plt.subplots(5,4, figsize=(4*3,5*3))
for l,ax in enumerate(axs.flatten()): 
    ax.imshow(np.abs(yspace)[...,l], cmap='bone')

# %%

plt.plot(np.abs(ytime), label=[f'base {l}' for l in range(20)])
plt.legend()

# %%


import jax.numpy as np
import jax 
from jax import grad, jit, vmap, lax, random
from inrmri.linalg_utils import gram_schmidt

# from jax import config
# config.update("jax_enable_x64", True)

matrices = random.normal(key, (4, 3))

Q = gram_schmidt(matrices) # runs fine

Q, np.dot(Q[:,0], Q[:,1]), np.dot(Q[0,:], Q[1,:])

np.dot(Q[:,2], Q[:,3])


# Q_tfp =  tfp.math.gram_schmidt(matrices) # runs fine
# %%


# %%
net.latent_z_time.shape

# %%

plt.imshow(np.abs(reco_grasp[...,26]))
# %%
# 256, 256,31

import numpy as onp 

grasp_reshaped = onp.reshape(reco_grasp, (N**2, NFRAMES)) # N = 256, NFRAMES = 31
grasp_reshaped.shape 

U, S, Vh = onp.linalg.svd(grasp_reshaped, full_matrices=False)
U.shape, S.shape, Vh.shape

# %%
n_values_list = [1,2, 5,10,15]

ims = {}

for n_values in n_values_list: 
    S_short = S.copy()
    S_short[n_values:] = 0. 
    im3 = onp.dot(U * S_short, Vh)
    #onp.sum(onp.abs(im2 - im3))
    ims[n_values] = onp.reshape(im3, (N, N,NFRAMES))

# %%

np.abs(np.dot(U[:,2], np.conjugate(U[:,3])))
# %%
U.shape, Vh.shape

idx = 2

first_component = S[idx] * onp.matmul(U[:, idx:idx+1], Vh[idx:idx+1,:])
first_component = onp.reshape(first_component, (N, N,NFRAMES))

full_halph_FOV_space_time([reduce_FOV(ims[1]), reduce_FOV(ims[10]), reduce_FOV(first_component)], crop_ns, frame = frame)
# %%

from inrmri.dip import Decoder
key = random.PRNGKey(0)
key_noise, key_params, key_eval, key_train = random.split(key, 4)
key_noise_space, key_noise_time = random.split(key_noise, 2)
key_params_space, key_params_time = random.split(key_params, 2)

L = 6
random_features = 32 
noise_scaling = 1.0

net = Decoder(features=64, levels=4, out_features=L*2)
input_noise = noise_scaling * random.normal(key_noise_space, (1,8,8,random_features)) # (1, nx, ny, 32)
params_space = net.init(key_params_space, input_noise, training=False)

net_time = Decoder(features=64, levels=4, out_features=L*2, dimensions=1)
input_noise_time = noise_scaling * random.normal(key_noise_time, (1,2,random_features)) # (1, frames, 32)
params_time = net_time.init(key_params_time, input_noise_time, training=False)


# %%

from inrmri.basic_nn import mse 

def get_im(params, key, training:bool): 
    out_im, updates = net.apply(params, input_noise, training=training, rngs={'dropout':key}, mutable=['batch_stats'])
    print(out_im.shape)
    new_shape = out_im.shape[:3] + (L,2)
    new_out_im = np.reshape(out_im, new_shape) # (1, nx, ny, 6) -> (1, nx, ny,3,2)
    new_out_im = new_out_im[0,...,0] + 1j * new_out_im[0,...,1] # (nx, ny, 3)
    return new_out_im, updates 

temporal_basis = np.array(S[:L,None] * Vh[:L])
def get_cine_from_basis(space_base, t_idx):
    nframes = t_idx.shape[0]
    space_base = np.reshape(space_base, (N*N, L))
    im = np.dot(space_base, temporal_basis[:,t_idx])
    im = np.reshape(im, (N,N,nframes))
    return im 

get_cine_from_basis(get_im(params, key, False)[0], np.array([1,2,5])).shape

# %%
def loss_only_space(params, X, Y, key): 
    # batch_size = X.shape[0]
    alphas, times = X[:,0], X[:,1]
    t_idx = np.int32(times * NFRAMES)
    space_base, update = get_im(params, key, training=True)
    ims = get_cine_from_basis(space_base, t_idx)
    ims = np.moveaxis(ims, -1, 0)
    spoke_radon_kspace = radon_operator.radon_transform(ims, alphas) # (batch-frame-alpha, ncoils, N)

    Y = Y[...,0] # (frames, cmap, nx)
    print("shapes: spoke ", spoke_radon_kspace.shape, " , Y ", Y.shape)
    loss_value = weighted_loss(spoke_radon_kspace, Y, (1. + WEIGHT_FREQS)[None,None,:])
    tv_val = TV_space(np.moveaxis(ims, 0, -1))
    return loss_value + lmbda_tv*tv_val, update 
# %%

learning_rate = 1e-3
optimizer = OptimizerWithExtraState(optax.adam(learning_rate))

total_kiter = 3 # 1000 iteraciones 
results = train_with_updates(jit(loss_only_space), train_X, train_Y, params, optimizer, key_train, nIter = total_kiter * 1000 + 1, batch_size = 3)

# %%
kiters = [1,2]

fig, axs = plt.subplots(len(kiters),L, figsize=(3*L,3*len(kiters)))
for kit, ax_row in zip(kiters, axs): 
    final_param = results['param_history'][f'param-{kit}']
    out, _ = get_im(final_param, key, training=False)

    for l, ax in enumerate(ax_row): 
        ax.imshow(np.abs(out[...,l]), cmap='bone', vmax=10.)
        # ax.set_title(f'Base $l={l+1}$, {kit}k iters')

for l, ax in enumerate(axs[0]): 
    ax.set_title(f'Base $l={l+1}$')
for kit, ax in zip(kiters, axs): 
    ax[0].set_ylabel(f'{kit}k iters')

np.abs(out).max(axis=(0,1))
# %%

crop_ns = [35,30,20,38]
frame = 26
reco_cine = get_cine_from_basis(out, np.arange(31))
full_halph_FOV_space_time([reduce_FOV(reco_cine)], crop_ns, frame = frame)

# %%

plt.plot(temporal_basis.transpose())