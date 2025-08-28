# Reco de los datos cortos del Rafa (3 slices de 1.5s cada una)
# %%

from scipy.io import loadmat 
import numpy as onp 
import matplotlib.pyplot as plt 
from inrmri.image_processor import reduce_FOV
from inrmri.basic_plotting import full_halph_FOV_space_time
from inrmri.bart import bart_acquisition_from_files, make_grasp_name, read_csmap, bart_read, bart_write

folder = '/mnt/storage/RafaData/2024-09-05_3lices1-5s/'

# %%
# ------------------------------------------------------- # 
# Guardar los datos 
# La versión multiframe y una versión con todos los datos 
# comprimidos a un único frame para estimar las bobinas.
# ------------------------------------------------------- # 
data = loadmat(folder + 'data_cine_d1.5_slice1_d_used1.5_frames30_nc15.mat')

bart_write(data['ktraj'], folder + 'bart/slice1-traj')
bart_write(data['kdata'], folder + 'bart/slice1-data')

def compress_all_frames(bart_array):
    kdata = onp.moveaxis(bart_array, 10, 3)
    kdata = onp.reshape(kdata, kdata.shape[:2] + (kdata.shape[2]*kdata.shape[3],) + kdata.shape[4:] + (1,))
    return kdata 

bart_write(compress_all_frames(data['ktraj']), folder + 'bart/slice1-1frame-traj')
bart_write(compress_all_frames(data['kdata']), folder + 'bart/slice1-1frame-data')

# %%
# ------------------------------------------------------- # 
# Reconstruir con bart 
# Calculo las coils y luego reconstruyo con bart 
# ------------------------------------------------------- # 

baq_1frame = bart_acquisition_from_files(folder + 'bart/', 'slice1-1frame')
csaq = baq_1frame.calculate_coil_sens()
csaq.write_csmap() # saves Bart data 
baq = bart_acquisition_from_files(folder + 'bart/', 'slice1')
grasp_params = {'lmbda': 0.01, 'lagrangian':0.5, 'iters':50}
grasp_name = make_grasp_name(grasp_params)
im = baq.calculate_bart_reco_with_external_csmappath(csaq.obj_path(), 'reco-' + grasp_name, **grasp_params)
im = onp.squeeze(im)

full_halph_FOV_space_time([reduce_FOV(im)], [35,30,20,38], saturation=0.35)

# %%
# ------------------------------------------------------- # 
# Cargar datos directamente desde archivos bart 
# ------------------------------------------------------- # 

csaq = read_csmap(folder + 'bart/', 'slice1-1frame') # esto permite leer directamente de los datos bart 
csmap = csaq.to_std_coil()
hollow_mask = csaq.hollow_mask()

im = bart_read('/mnt/storage/RafaData/2024-09-05_3lices1-5s/bart/slice1-reco-lmbda0-1_lagrangian0-5_iters50')
im = onp.squeeze(im)

im_sense = baq.make_sense_reconstruction(csaq.obj_path(), 'sense')
im_sense = onp.squeeze(im_sense)

# %%
from jax import jit, vmap, random
import jax.numpy as np 

from inrmri.data_harvard import load_data
from inrmri.radon import calculate_angle 
from inrmri.utils import to_complex, is_inside_of_radial_lim, meshgrid_from_subdiv_autolims
from inrmri.radial_acquisitions import kFOV_limit_from_spoke_traj
from inrmri.radial_acquisitions import check_correct_dataset 
from inrmri.new_radon import ForwardRadonOperator

from inrmri.basic_nn import weighted_loss 

import optax 
from inrmri.advanced_training import train_with_updates, OptimizerWithExtraState

key = random.PRNGKey(0)
key_net, key_params, key_train = random.split(key,3) # keys for reproducibility 


# %% Configurations and data loading 
# available configuration, you can change this 
from inrmri.radial_acquisitions import create_radial_acq_from_bart 

ra = create_radial_acq_from_bart(baq)
train_X, train_Y = ra.generate_dataset()
train_Y = 100 * train_Y #para que quede en un orden de magnitud adecuado
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

# gtim = np.load(f'/home/tabita/ACIP-MRI/ACIP-MRI/data_GTs/GT-{config_data["chosen_patient"]}.npy') # load gtim made with nufft, shape (px, py, frames)
gtim = im 
NFRAMES = gtim.shape[-1]
IMSHAPE = gtim.shape[:2]
N = IMSHAPE[0]

radon_operator = ForwardRadonOperator(csmap, spclim)

grid = meshgrid_from_subdiv_autolims(IMSHAPE)

def post_processing(im): # im.shape (px,py,nframes)
    is_inside = is_inside_of_radial_lim(grid, 1.)
    masks = (1 - hollow_mask) * is_inside
    return im * masks[:,:,None]

# %% Network 
from inrmri.dip import TimeDependant_DIP_Net, helix_generator

features = 64

CONFIG_NET = {
    'mapnet_layers':[128, 128,],
    'cnn_latent_shape':(8,8),
    'levels':4,
    'features':features
}

ts = np.linspace(0,1,gtim.shape[-1], endpoint=False)
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

# %% Loss function 

from inrmri.new_radon import get_weight_freqs

WEIGHT_FREQS = get_weight_freqs(N)

def loss(params, X, Y, key): 
    batch_size = X.shape[0]
    alphas, times = X[:,0], X[:,1]
    t_idx = np.int32(times * NFRAMES)
    ims, update = net.train_forward_pass(params, key, t_idx) # (frames, px, py)    
    spoke_radon_kspace = radon_operator.radon_transform(ims, alphas) # (batch-frame-alpha, ncoils, N)

    Y = Y[...,0] # (frames, cmap, nx)
    print("shapes: spoke ", spoke_radon_kspace.shape, " , Y ", Y.shape)
    loss_value = weighted_loss(spoke_radon_kspace, Y, (1. + WEIGHT_FREQS)[None,None,:])
    return loss_value, update 

# %% Training 

learning_rate = 7.5e-4
optax.exponential_decay(7.5e-4, transition_steps=500, decay_rate=0.7)
optimizer = OptimizerWithExtraState(optax.adam(learning_rate))

total_kiter = 3 # 1000 iteraciones 

config_training = {
    'learning_rate': learning_rate,
    'total_kiter': total_kiter,
    'key_train': key_train
}

# para saber cuanta memoria realmente esta usando jax hay que desactivar la preallocation por default 
results = train_with_updates(jit(loss), train_X, train_Y, params, optimizer, key_train, nIter = total_kiter * 1000 + 1, batch_size = 10)

# %% Examining results 

final_param = results['param_history'][f'param-{total_kiter}']

# evaluar en modo training  
from jax.lax import map as laxmap 

predim = laxmap(lambda t: net.train_forward_pass(final_param, key, t[None])[0][0], np.arange(gtim.shape[-1])) # TODO make lax version, vectorized evaluation takes too much memory 
predim = np.moveaxis(predim, 0, -1)
predim = post_processing(predim) / np.abs(predim).max()
predim.shape

# %%

from inrmri.basic_plotting import full_halph_FOV_space_time
from inrmri.image_processor import BeforeLinRegNormalizer, reduce_crop_abs
import matplotlib.pyplot as plt 

crop_ns = [35,30,20,38]
improc = BeforeLinRegNormalizer(gtim, [0,0,0,0])

gt_proc = improc.process(gtim)
pred_proc = improc.process(predim)
sense_proc = improc.process(im_sense)

frame = 8 # 8, 23 

fig, axs = full_halph_FOV_space_time([gt_proc, pred_proc, sense_proc], crop_ns, frame = frame,saturation=0.5)
axs[0,0].set_title('GRASP')
axs[1,0].set_title('TD-DIP')
axs[2,0].set_title('CG-SENSE')

# %%

im_2 = np.reshape(im_sense, (im_sense.shape[0]*im_sense.shape[1], im_sense.shape[-1]))
im_mean = np.mean(im_2, axis=0)
im_2 = im_2 - im_mean
im_std = np.std(np.abs(im_2), axis=0)
im_2 = im_2 / im_std
cov_matrix = np.cov(im_2, rowvar=False)
# %%

U, S, Vt = onp.linalg.svd(cov_matrix, full_matrices=False)

S2 = S.copy()

# %%
# from einops import rearrange

im_small = np.einsum('jk,kl->jl', im_2 * im_std, Vt[:,:5])

plt.figure()
plt.subplot(121)
plt.imshow(np.abs(im_small[...,0].reshape(IMSHAPE)), cmap='bone')
plt.subplot(122)
plt.imshow(np.abs(im_small[...,1].reshape(IMSHAPE)), cmap='bone')

# %%

n = 5
S2[n:] = 0. 




# %%
import numpy as onp 


# %%

U.shape, S.shape, Vt.shape

# %%

