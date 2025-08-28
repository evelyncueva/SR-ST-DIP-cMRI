# tdDIP-reco.py
# Reconstruction of Harvard Radial dataset using TD-DIP.
# For more info in the configurations of the data loading, check
# `notebooks/invivo_radial_data_example.ipynb`
# For more info for modifying TD-DIP, check tutorial/TD-DIP.md
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
args_dic = {
    'spokes': 16, # submuestreo pseudo-golden-angle de 16 spokes por frame 
    'patient_number': 7, # paciente P07 de la base de datos de harvard (ver inrmri.data_harvard)
}

# these allow further modification on the undersampling, don't change if you don't understand what they do 
patient_number = args_dic['patient_number']
patient = f'P{patient_number:02}'

config_data = {
    'chosen_patient': patient, 
    'sub_spokes_per_frame': args_dic['spokes'],
    'relaxed_pad_removal': False,
    'hermitic_fill': True,
    'tiny_number': 1
}

# la salida de load_data está pensada para trabajar con INRs, por eso hago un poco más 
# de procesamiento después 
train_X, train_Y, csmap, _, hollow_mask = load_data(config_data) 

check_correct_dataset(train_X)

spclim = kFOV_limit_from_spoke_traj(train_X[0,1:,:])

angles = vmap(calculate_angle)(train_X[:,1:]) # (total_spokes,)

NFRAMES = 25 

# index of the bin associated to each spoke, in [0, ..., total_bins-1]
times = np.int32(np.round(train_X[:,0,0] * NFRAMES)) # (total_spokes,)

train_X = np.stack([angles, times], axis=-1) # sampled angles with their respective time 
train_Y = 3 * train_Y/ (np.abs(train_Y).max()) 

# %% 
plt.imshow(hollow_mask)

# %%
gtim = np.load(f'/home/tabita/ACIP-MRI/ACIP-MRI/data_GTs/GT-{config_data["chosen_patient"]}.npy') # load gtim made with nufft, shape (px, py, frames)
IMSHAPE = gtim.shape[:2] # (N,N)
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
    'cnn_latent_shape':(13,13),
    'levels':4,
    'features':features
}

net = TimeDependant_DIP_Net(
    nframes=NFRAMES, 
    total_cycles=1,
    latent_generator=helix_generator,
    imshape=IMSHAPE,
    **CONFIG_NET
    )

params = net.init_params(key_params)

# %% Loss function 

from inrmri.new_radon import get_weight_freqs

WEIGHT_FREQS = get_weight_freqs(N)

def loss(params, X, Y, key): 
    batch_size = X.shape[0]
    alphas, times = X[:,0], X[:,1]
    t_idx = np.int32(times)
    ims, update = net.train_forward_pass(params, key, t_idx) # (frames, px, py, 1)    
    ims = ims[...,0] # (frames, px, py)
    rotated = radon_operator.rotated_csweighted_ims(ims, alphas) #
    spoke_radon_kspace = radon_operator.radon_transform(rotated) # (batch-frame-alpha, ncoils, N)

    Y = Y[...,0] # (frames, cmap, nx)
    print("shapes: spoke ", spoke_radon_kspace.shape, " , Y ", Y.shape)
    loss_value = weighted_loss(spoke_radon_kspace, Y, (1. + WEIGHT_FREQS)[None,None,:])
    return loss_value, update 

# %% Training 

learning_rate = 2e-3
optimizer = OptimizerWithExtraState(optax.adam(learning_rate))

total_kiter = 1 # 1000 iteraciones 

config_training = {
    'learning_rate': learning_rate,
    'total_kiter': total_kiter,
    'key_train': key_train
}

# para saber cuanta memoria realmente esta usando jax hay que desactivar la preallocation por default 
results = train_with_updates(jit(loss), train_X, train_Y, params, optimizer, key_train, nIter = total_kiter * 1000 + 1, batch_size = 5)

# %% Examining results 
from jax.lax import map as laxmap 

final_param = results['param_history'][f'param-{total_kiter}']

def lax_cine(params, key, t_idx, batch:int = 5):
    """
        Reconstruction of a big batch of frames, in lax mode (vectorization takes too much memory)
    - t_idx: array(int), indexs to reconstruct
    """
    Nt = t_idx.shape[0]
    corrected_Nt = batch * (Nt // batch)
    stacked_tidxs = t_idx[:corrected_Nt].reshape(Nt//batch,batch)
    predim = laxmap(lambda t: net.train_forward_pass(params, key, t)[0], stacked_tidxs) # (nbatches, batch, n, n, 1)
    predim = np.reshape(predim, (corrected_Nt , N, N))
    predim = np.moveaxis(predim, 0, -1) # (N, N, frames)
    predim = post_processing(predim)
    return predim 

# evaluar en modo training  
predim = lax_cine(final_param, key, np.arange(NFRAMES)) 
predim.shape

# %%

from inrmri.basic_plotting import full_halph_FOV_space_time
from inrmri.data_harvard import PATIENTS_HEART_LOCATION
from inrmri.image_processor import BeforeLinRegNormalizer, reduce_crop_abs
import matplotlib.pyplot as plt 

crop_ns = PATIENTS_HEART_LOCATION[patient]
improc = BeforeLinRegNormalizer(gtim, [0,0,0,0])

gt_proc = improc.process(gtim)
pred_proc = improc.process(predim)

frame = 8 # 8, 23 

fig, axs = full_halph_FOV_space_time([gt_proc, pred_proc], crop_ns, frame = frame,saturation=0.5)

# %%
