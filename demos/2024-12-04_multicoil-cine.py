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

# gtim = np.ones((N,N,NFRAMES))
# gtim = reco_grasp 
NFRAMES = gtim.shape[-1]
IMSHAPE = gtim.shape[:2]
N = IMSHAPE[0]

radon_operator = ForwardRadonOperator(csmap, spclim)
radon_operator_no_coil = ForwardRadonOperator(np.ones((1, N, N)), spclim)

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

CONFIG_NET = {
    'mapnet_layers':[64, 64],
    'cnn_latent_shape':(8,8),
    'levels':4,
    'features':features
}

def circle_generator(nframes, total_cycles):
    circ = np.stack([np.cos(2 * np.pi * ts), np.sin(2 * np.pi *ts)], axis=-1)
    return circ 

net = TimeDependant_DIP_Net(
    nframes=NFRAMES, 
    total_cycles=1,
    latent_generator=circle_generator,
    imshape=gtim.shape[:2],
    **CONFIG_NET
    )

params = net.init_params(key_params)

net_multi_coil = TimeDependant_DIP_Net(
    nframes=NFRAMES, 
    total_cycles=1,
    latent_generator=circle_generator,
    imshape=gtim.shape[:2],
    **CONFIG_NET, 
    out_images=csmap.shape[0]
    )

params_multicoil = net_multi_coil.init_params(key_params)
coils_scalings = np.abs(train_Y).max(axis=(0,2,3)) # (ncoils,)

# %%
from inrmri.new_radon import get_weight_freqs
from inrmri.regularizations import TV_space
from inrmri.linalg_utils import gram_schmidt
from inrmri.dip import to_complex, split_last_dim 

WEIGHT_FREQS = get_weight_freqs(N)
lmbda_tv = 0

def loss(params, X, Y, key): 
    batch_size = X.shape[0]
    alphas, times = X[:,0], X[:,1]
    t_idx = np.int32(times * NFRAMES)
    ims, update = net.train_forward_pass(params, key, t_idx)
    ims = ims[...,0]
    # y_space, y_time, update = net.train_basis_and_update(params, key, t_idx)
    # ims = ortonorm_im_from_net_basis(y_space, y_time)
    spoke_radon_kspace = radon_operator.radon_transform(ims, alphas) # (batch-frame-alpha, ncoils, N)

    Y = Y[...,0] # (frames, cmap, nx)
    print("shapes: spoke ", spoke_radon_kspace.shape, " , Y ", Y.shape)
    loss_value = weighted_loss(spoke_radon_kspace, Y, (1. + WEIGHT_FREQS)[None,None,:])
    tv_val = TV_space(np.moveaxis(ims, 0, -1))
    return loss_value + lmbda_tv*tv_val, update 

def loss_multicoil(params, X, Y, key): 
    batch_size = X.shape[0]
    alphas, times = X[:,0], X[:,1]
    t_idx = np.int32(times * NFRAMES)
    ims, update = net_multi_coil.train_forward_pass(params, key, t_idx) # (batch, nx, ny, ncoils)
    ims = ims * coils_scalings
    print("ims.shape: ", ims.shape)
    spoke_radon_kspace = vmap(radon_operator_no_coil.radon_transform, in_axes=(-1, None), out_axes=1)(ims, alphas) # (batch-frame-alpha, coil, 1, N)
    print("spoke_radon_kspace.shape: ", spoke_radon_kspace.shape)
    spoke_radon_kspace = spoke_radon_kspace[:,:,0,:] # (batch-frame-alpha, coil, N)
    Y = Y[...,0] # (frames, cmap, nx)
    print("shapes: spoke ", spoke_radon_kspace.shape, " , Y ", Y.shape)
    loss_value = weighted_loss(spoke_radon_kspace, Y, (1. + WEIGHT_FREQS)[None,None,:])
    tv_val = TV_space(np.moveaxis(ims, 0, -1))
    return loss_value + lmbda_tv*tv_val, update 

# %% Training 

learning_rate = 2e-3
optimizer = OptimizerWithExtraState(optax.adam(learning_rate))

total_kiter = 5 # 1000 iteraciones 

# para saber cuanta memoria realmente esta usando jax hay que desactivar la preallocation por default 
results = train_with_updates(jit(loss), train_X, train_Y, params, optimizer, key_train, nIter = total_kiter * 1000 + 1, batch_size = 10)

# %%

learning_rate = 2e-3
optimizer = OptimizerWithExtraState(optax.adam(learning_rate))

total_kiter = 5 # 1000 iteraciones 

# para saber cuanta memoria realmente esta usando jax hay que desactivar la preallocation por default 
results_multicoil = train_with_updates(jit(loss_multicoil), train_X, train_Y, params_multicoil, optimizer, key_train, nIter = total_kiter * 1000 + 1, batch_size = 10)

# %% Examining results 
from jax.lax import map as laxmap 

selected_kiter = 5

final_param = results['param_history'][f'param-{selected_kiter}']

predim = laxmap(lambda t: net.train_forward_pass(final_param, key, t[None])[0][0,...,0], np.arange(NFRAMES))
predim = np.moveaxis(predim, 0, -1)
predim = predim * (1-data_slice['hollow_mask'][...,None])
predim.shape 

final_param = results_multicoil['param_history'][f'param-{selected_kiter}']
predim_multicoil = laxmap(lambda t: net_multi_coil.train_forward_pass(final_param, key, t[None])[0][0], np.arange(NFRAMES))

predim_multicoil = predim_multicoil * coils_scalings
predim_multicoil = np.sqrt(np.sum(predim_multicoil ** 2, axis=-1))
predim_multicoil = np.moveaxis(predim_multicoil, 0, -1)

# plt.imshow(np.abs(predim_multicoil[..., 0]))

# td_dip_multicoil_3k = predim
# %%

from inrmri.basic_plotting import full_halph_FOV_space_time
from inrmri.image_processor import BeforeLinRegNormalizer, reduce_crop_abs
import matplotlib.pyplot as plt 

crop_ns = [35,30,20,38]
# gtim = np.load(folder + 'sl5-reco-reference.npy') # 
gtim = predim
improc = BeforeLinRegNormalizer(gtim, [0,0,0,0]) # normaliza todas las imagenes a la escala de gtim 

gt_proc = improc.process(gtim)

def final_processing(im):
    im = improc.process(im)
    # im = onp.roll(im, 15, axis=-1) # 15 es el argmin del error con gastao luego de probar todos los rolls 
    return im 

# el [1:] es porque gtim solo tiene 30 frames (no 31)
pred_proc = final_processing(predim) 
pred_multic_proc = final_processing(predim_multicoil) 

# pred_proc = final_processing(predim[...,1:]) 
# pred_multic_proc = final_processing(predim_multicoil[...,1:]) 
# sense_proc = final_processing(reco_sense[...,1:])
# grasp_proc = final_processing(reco_grasp[...,1:])

frame = 8 # 8, 23 
vmax = np.abs(gtim).max()
# recos = [gt_proc, pred_proc, pred_multic_proc, grasp_proc, sense_proc]
# titles = ['MCCINE (Gastao)', 'TD-DIP', 'AC-TD-DIP', 'GRASP', 'SENSE']

recos = [ pred_proc, pred_multic_proc]
titles = ['TD-DIP', 'AC-TD-DIP']
fig, axs = full_halph_FOV_space_time(recos, crop_ns, frame = frame,saturation=0.35, vmax=vmax)
for ax, title in zip(axs, titles):
    ax[0].set_title(title)

plt.suptitle(f'{selected_kiter}k iters')
# %%
