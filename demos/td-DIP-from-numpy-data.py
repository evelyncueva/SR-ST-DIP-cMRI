# Reco de los datos cortos del Rafa (3 slices de 1.5s cada una)
# %%

import numpy as onp 
import matplotlib.pyplot as plt 
from inrmri.image_processor import reduce_FOV
from inrmri.basic_plotting import full_halph_FOV_space_time

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

from inrmri.radon import calculate_angle 
from inrmri.utils import  is_inside_of_radial_lim, meshgrid_from_subdiv_autolims
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
from inrmri.dip.tddip import TimeDependant_DIP_Net, helix_generator

ts = np.linspace(0,1,gtim.shape[-1], endpoint=False)

net = TimeDependant_DIP_Net(NFRAMES, 1, helix_generator, IMSHAPE, [64,64], (8,8), features=64, levels=4, out_images=1)

params = net.init_params(key_params)

# %%
from inrmri.new_radon import get_weight_freqs

WEIGHT_FREQS = get_weight_freqs(N)

def loss(params, X, Y, key): 
    alphas, times = X[:,0], X[:,1]
    t_idx = np.int32(times * NFRAMES)
    ims, updates = net.train_forward_pass(params, key, t_idx) # (batch, n,n, 1)
    ims = ims[...,0] # (batch, n, n)

    rotated_im = radon_operator.rotated_csweighted_ims(ims, alphas) # (batch-frame-alpha, ncoils, N)
    spoke_radon_kspace = radon_operator.radon_transform(rotated_im) # (batch-frame-alpha, ncoils, N)

    Y = Y[...,0] # (frames, cmap, nx)
    print("shapes: spoke ", spoke_radon_kspace.shape, " , Y ", Y.shape)
    loss_value = weighted_loss(spoke_radon_kspace, Y, (1. + WEIGHT_FREQS)[None,None,:])

    return loss_value, updates 

# %% Training 

learning_rate = 1e-3
optimizer = OptimizerWithExtraState(optax.adam(learning_rate))

total_kiter = 1 # 1000 iteraciones 

config_training = {
    'learning_rate': learning_rate,
    'total_kiter': total_kiter,
    'key_train': key_train
}

results = train_with_updates(jit(loss), train_X, train_Y, params, optimizer, key_train, nIter = total_kiter * 1000 + 1, batch_size = 3)

# %% Examining results 

final_param = results['param_history'][f'param-{total_kiter}']

predim = net.lax_cine(final_param, key, np.arange(NFRAMES), 5)
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
titles = ['MCCINE (Gastao)', f'TD-DIP {total_kiter}k', 'GRASP', 'SENSE']
fig, axs = full_halph_FOV_space_time(recos, crop_ns, frame = frame,saturation=0.5, vmax=vmax)
for ax, title in zip(axs, titles):
    ax[0].set_title(title)

