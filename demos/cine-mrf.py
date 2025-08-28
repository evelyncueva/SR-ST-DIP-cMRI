# Reco de los datos cortos del Rafa (3 slices de 1.5s cada una)
# %%

from scipy.io import loadmat 
import numpy as onp 
import matplotlib.pyplot as plt 
from inrmri.image_processor import reduce_FOV
from inrmri.basic_plotting import full_halph_FOV_space_time
from inrmri.bart import bart_acquisition_from_files, make_grasp_name, read_csmap, bart_read, bart_write

from inrmri.advanced_training import OptimizerWithExtraState, train_with_updates
import optax 
from jax import jit 
data_folder = '/mnt/storage/data-MRF/' # cambiarlo al tuyo, debería terminar en / 

# %%
# ------------------------------------------------------- # 
# Cargar los datos disponibles 
# ------------------------------------------------------- # 

import jax.numpy as np 
from inrmri.radial_acquisitions import RadialAcquisitions

print("Loading data...")

# data = np.load(data_folder + 'data_dict_MRF_v2.npy', allow_pickle=True)[()]
data = np.load(data_folder + 'data_dict_mrf_data_hv7.npy', allow_pickle=True)[()]
im_singular = data['MRF_Reg_sing']

# %%

scalings = np.percentile(np.abs(im_singular), 99, axis=(0,1))

# %%

n_ims = im_singular.shape[2]
fig, axs = plt.subplots(1,n_ims, figsize=(n_ims*3, 3))
for n_sing in range(n_ims):
    axs[n_sing].imshow(np.abs(im_singular[...,n_sing, 0] / scalings[n_sing, 0]), vmax=2., cmap='bone')

# %%

from jax import random 
from inrmri.dip import Decoder 

key = random.PRNGKey(0)
key_space, key_params, key_eval, key_train = random.split(key, 4)

ncoils = data['csm'].shape[-1]
net = Decoder(features=32, levels=4, out_features= n_ims*2)
noise_scaling = 1.0
random_features = 1 
input_noise = noise_scaling * random.normal(key_space, (8,8,random_features)) # (1, nx, ny, 32)

params = net.init(key_params, input_noise, training=False)

# %%

# from inrmri.dip import TimeDependant_DIP_Net, helix_generator

# features = 64

# CONFIG_NET = {
#     'mapnet_layers':[128, 128,],
#     'cnn_latent_shape':(8,8),
#     'levels':4,
#     'features':features
# }

# NFRAMES=375
# ts = np.linspace(0,1,NFRAMES, endpoint=False)
# def line_generator(nframes, total_cycles):
#     # circ = np.stack([np.cos(2 * np.pi * ts), np.sin(2 * np.pi *ts)], axis=-1)
#     return ts[:,None]

# net = TimeDependant_DIP_Net(
#     nframes=NFRAMES, 
#     total_cycles=1,
#     latent_generator=line_generator,
#     imshape=(256,256),
#     **CONFIG_NET
#     )

# params = net.init_params(key_params)

# %% 
from inrmri.basic_nn import mse 

def get_im(params, key, training:bool): 
    out_im, updates = net.apply(params, input_noise, training=training, rngs={'dropout':key}, mutable=['batch_stats']) 
    new_shape = out_im.shape[:2] + (n_ims,2)
    new_out_im = np.reshape(out_im, new_shape) # (nx, ny, 12) -> (nx, ny,6,2)
    new_out_im = new_out_im[...,0] + 1j * new_out_im[...,1] # (nx, ny,6)
    return new_out_im, updates 

def compress_multicoil_im(im, coil_axis): 
    return np.sqrt(np.sum(np.abs(im**2), axis=coil_axis))

chosen_echo = 0
def loss_im(params, X, Y, key): 
    new_out_im, updates = get_im(params, key, training=True) # (nx, ny, 6)
    #new_out_im = compress_multicoil_im(new_out_im, coil_axis=-2)
    loss_val = mse(new_out_im, np.abs(im_singular[...,chosen_echo] / (scalings[None,None,:, chosen_echo] / 250)))
    return loss_val, updates 

# %%

learning_rate = 2e-3
optimizer = OptimizerWithExtraState(optax.adam(learning_rate))

total_citer = 2 # 1000 iteraciones 
dummy_train_X = np.arange(1)
dummy_train_Y = np.arange(1)

results = train_with_updates(jit(loss_im), dummy_train_X, dummy_train_Y, params, optimizer, key_train, nIter = total_citer * 100 + 1, batch_size = 1, save_at=200)
# %%

selected_citer = 1
pred_sing_im, _ = get_im(results['param_history'][f'param-{selected_citer}'], key, True)
vmax = 300.
fig, axs = plt.subplots(2,n_ims, figsize=(n_ims*3, 2*3))
for n_sing in range(n_ims):
    axs[0,n_sing].imshow(np.abs(pred_sing_im[...,n_sing]), vmax=vmax, cmap='bone')
    axs[1,n_sing].imshow(np.abs(im_singular[...,n_sing, chosen_echo] /(scalings[n_sing, chosen_echo]/250)), vmax=vmax, cmap='bone')
axs[0,0].set_ylabel(f'DIP denoising {selected_citer}k iters')
axs[1,0].set_ylabel(f'Sing im Diego')
np.abs(pred_sing_im).max()

# %%

# plt.imshow(np.abs(train_Y[:,9]))
# plt.hlines([40], xmin=0, xmax=255.5)
# plt.ylabel('frame')
# plt.xlabel('k-freq')

# %%
from inrmri.basic_plotting import remove_ticks
vmax = 0.6 * np.abs(pred_sing_im[...,0]).max()
fig, axs = plt.subplots(5,3, figsize=(9,15))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(np.abs(pred_sing_im[...,0] * radon_operator.csmap[i]), cmap='bone', vmax=vmax)
    # ax.imshow(np.abs(radon_operator.csmap[i]), cmap='bone', vmax=vmax)
    ax.set_title(f'coil {i}')
    remove_ticks(ax)
# %%
from inrmri.radon import calculate_angle
from jax import vmap 

NFRAMES = 375 
N = 256 
trajs = data['kp'][0].reshape((NFRAMES, N, 2)) * N 
# trajs = data['kpalternate'].reshape((NFRAMES, 256, 2))

# angles = np.mod(vmap(calculate_angle)(trajs) + np.pi, 2 * np.pi) # esto no funcionó 
angles = vmap(calculate_angle)(trajs) 
t_idxs = np.arange(NFRAMES)

train_X = np.stack([angles, t_idxs], axis=-1)

train_Y = data['kdata_Echoes'][...,0,:] # (ro, frames, coils)
train_Y = np.moveaxis(train_Y, 0, -1) # (frames, coils,ro)
train_Y = train_Y * 3 / np.abs(train_Y).max()

# %%

from inrmri.basic_nn import weighted_loss 
from inrmri.new_radon import get_weight_freqs, ForwardRadonOperator
from inrmri.bart import calculate_hollow_mask

radon_operator = ForwardRadonOperator(csmap=np.array(data['csm'].transpose((2,0,1))), spclim=0.5)
#radon_operator = ForwardRadonOperator(csmap=np.ones((1,N, N)), spclim=0.5)

UR = np.array(data['Ur'][0]) # (frames, 6)
lmbda_tv = 0.

scale_factor = 1e9

# def get_k_space_from_singulars_in_kspace(pred_sing_im, angle, t_idx:int): 
#     pred_sing_ims = np.moveaxis(pred_sing_im, -1, 0) # (6, nx,ny)
#     pred_sing_ims = pred_sing_ims*scalings[:,None,None]*scale_factor
#     repeated_angle = np.ones(6)*angle
#     kspaces = radon_operator.radon_transform(pred_sing_ims, repeated_angle)
#     frame_UR = UR[t_idx,:,None,None]
#     # frame_UR = np.conjugate(UR)[t_idx,:,None,None]
#     data_y = np.sum(frame_UR * kspaces, axis=0) 
#     return data_y 

def get_k_space_from_singulars_in_im(pred_sing_im, angles, t_idx): 
    """
    pred_sing_im: (n, n, singular)
    angles: array[int], shape (batch,)
    t_idx: array[int], shape (batch,)

    ## Output 
    
    - data_y: complex shape (batch, coil, nx)
    """
    print(pred_sing_im.shape)
    pred_sing_ims = np.moveaxis(pred_sing_im, -1, 0) # (singular, nx,ny)
    pred_sing_ims = pred_sing_ims*scalings[:,None,None, 0]*scale_factor # (singular, nx,ny)
    frame_UR = UR[t_idx,:,None,None] # (batch, singular, 1, 1)
    combined_im = np.sum(pred_sing_ims[None] * frame_UR, axis=1) # (batch, singular, nx,ny) -> (batch,nx,ny)
    rotated = radon_operator.rotated_csweighted_ims(combined_im, angles) # (batch, coil, n,n)
    data_y = radon_operator.radon_transform(rotated) # (batch, coil, nx)
    return data_y 

WEIGHT_FREQS = get_weight_freqs(N)

DCF = np.moveaxis(np.array(data['dcf'][0]), 0, -1) 

def loss(params, X, Y, key): 
    batch_size = X.shape[0]
    alphas, times = X[:,0], X[:,1]
    t_idx = np.int32(times)
    sing_ims, updates = get_im(params, key, training=True) # (n,n, singular)
    sing_ims = sing_ims * data['coil_rss'][:,:,None] # intensity correction in zone with no coil sensitivity 
    
    spoke_radon_kspace = get_k_space_from_singulars_in_im(sing_ims, alphas, t_idx) # (batch, coil, n)
    spoke_radon_kspace = spoke_radon_kspace * DCF[t_idx,None,:]
    print("shapes: spoke ", spoke_radon_kspace.shape, " , Y ", Y.shape)
    loss_value = 100. * mse(spoke_radon_kspace, Y, )
    #loss_value = weighted_loss(spoke_radon_kspace, Y, (1. + WEIGHT_FREQS)[None,None,:])
    tv_val = 0 #TV_space(np.moveaxis(ims, 0, -1))
    return loss_value + lmbda_tv*tv_val, updates 
# %%
selected_coil = 9 
selected_t = 160
from jax.numpy.fft import fftfreq, fftshift

freqs = fftshift(fftfreq(256))

combined = np.sum(im_singular[:,:,:,0] * UR[None,None,selected_t,:], axis=-1) # (n,n,frames)
# *WEIGHT_FREQS[selected_t]
plt.plot(freqs, np.real(WEIGHT_FREQS[selected_t, :] * get_k_space_from_singulars_in_im(im_singular[...,0], train_X[selected_t:selected_t+1, 0], np.array([selected_t]))[0,selected_coil,:]))
plt.plot(freqs, 13e-8*np.real(train_Y[selected_t, selected_coil]))

# %%
# t_idx = 100
# angle = np.pi + train_X[t_idx,0]

# # data_y_1 = get_k_space_from_singulars_in_kspace(im_singular/scalings[None,None,:], angle, t_idx)
# data_y_2 = get_k_space_from_singulars_in_im(im_singular, angle, t_idx) * scale_factor / 3000

# selected_coil = 9
# funcs = [np.abs, np.real, np.imag]
# fig, axs = plt.subplots(1,3, figsize=(9,3))

# for ax, fun in zip(axs, funcs): 
#     ax.plot(fun(train_Y[t_idx,selected_coil]))
#     ax.plot(fun(data_y_1[selected_coil]), label='model in kspace')
#     ax.plot(fun(data_y_2[selected_coil]), label='model in im space') # parece dar lo mismo usar la matriz en imagen o en k-space 
# plt.legend()
# # %%
# def loss(params, X, Y, key): 
#     batch_size = X.shape[0]
#     alphas, times = X[:,0], X[:,1]
#     t_idx = np.int32(times)
#     batch_frames, updates = net.train_forward_pass(params, key, t_idx) # (frames, nx, ny)
#     batch_frames = batch_frames * (1-hollow_mask)[None,:,:]
#     spoke_radon_kspace = 100 * radon_operator.radon_transform(batch_frames, alphas) # (batch-frame-alpha, ncoils, N)
#     print("shapes: spoke ", spoke_radon_kspace.shape, " , Y ", Y.shape)
#     # loss_value = weighted_loss(spoke_radon_kspace, Y, (1. + WEIGHT_FREQS)[None,None,:])
#     loss_value = 0.01 * weighted_loss(spoke_radon_kspace, Y, (1. + WEIGHT_FREQS)[t_idx,None,:])
    
#     tv_val = 0 #TV_space(np.moveaxis(ims, 0, -1))
#     return 0.1 * loss_value + lmbda_tv*tv_val, updates 


# %%
learning_rate = 2e-3
optimizer = OptimizerWithExtraState(optax.adam(learning_rate))

total_kiter = 3 # 1000 iteraciones 

results = train_with_updates(jit(loss), train_X, train_Y, params, optimizer, key_train, nIter = total_kiter * 1000 + 1, batch_size = 5)

# %%
from inrmri.basic_plotting import remove_ticks
selected_param = 3
final_param = results['param_history'][f'param-{selected_param}']
pred_sing_im, _ = get_im(final_param, key, False) # (nx, ny, 6)
pred_sing_im = pred_sing_im * data['coil_rss'][:,:,None]

total_singular = im_singular.shape[2]
vmax = 0.7 * np.abs(im_singular[...,0] / scalings[...,0]).max()

pred_sing_im = vmax * pred_sing_im / np.abs(pred_sing_im).max()

fig, ax = plt.subplots(2,total_singular, figsize=(total_singular*3, 2*3))
for i in range(total_singular):
    ax[0,i].set_title(f'Singular {i}')
    ax[0,i].imshow(reduce_FOV(np.abs(pred_sing_im[...,i])), cmap='bone', vmax=vmax)
    remove_ticks(ax[0,i])

    ax[1,i].set_title(f'Singular {i}')
    ax[1,i].imshow(reduce_FOV(np.abs(im_singular[...,i, 0])/scalings[i,0]), cmap='bone', vmax=vmax)
    remove_ticks(ax[1,i])

ax[0,0].set_ylabel(f'DIP')
ax[1,0].set_ylabel(f'Diego')

# %%

selected_frames =  np.array([50,125,160,200,250])
combined_frames_dip = np.sum(pred_sing_im[:,:,None,:] * UR[None,None,selected_frames,:], axis=-1) # (n,n,frames)
combined_frames_diego = np.sum(im_singular[:,:,None,:,0] * UR[None,None,selected_frames,:], axis=-1) # (n,n,frames)

vmax = 0.4 * np.abs(combined_frames_dip).max() * np.abs(UR).max()

nplots = selected_frames.shape[0]
fig, axs = plt.subplots(2,nplots, figsize=((nplots+1)*3, 2*3))
for i, frame in enumerate(selected_frames):
    axs[0,i].set_title(f'frame {frame}')
    axs[0,i].imshow(reduce_FOV(np.abs(combined_frames_dip[...,i])), cmap='bone')
    axs[1,i].imshow(reduce_FOV(np.abs(combined_frames_diego[...,i])), cmap='bone')
axs[0,0].set_ylabel('TD-DIP')
axs[1,0].set_ylabel('Combined singular ims')

# %%

batch_frames, updates = net.train_forward_pass(final_param, key, selected_frames) #get_im(params, key, training=True) # (nx, ny, 6)
batch_frames = batch_frames * data['coil_rss'][None,:,:]
batch_frames = np.moveaxis(batch_frames, 0, -1)

combined_frames = np.sum(im_singular[:,:,None,:] * UR[None,None,selected_frames,:], axis=-1)
vmax = 0.4 * np.abs(batch_frames).max() * np.abs(UR).max()
nplots = selected_frames.shape[0]
fig, axs = plt.subplots(2,nplots+1, figsize=((nplots+1)*3, 2*3))
for n_sing in range(nplots):
    axs[0,n_sing].set_title(f'frame {selected_frames[n_sing]}')
    axs[0,n_sing].imshow(reduce_FOV(np.abs(batch_frames[...,n_sing])), cmap='bone', vmax= 0.4 * np.abs(batch_frames).max())
    axs[1,n_sing].imshow(reduce_FOV(np.abs(combined_frames[...,n_sing])), cmap='bone', vmax= 1.* np.abs(combined_frames).max())
axs[0,0].set_ylabel('TD-DIP')
axs[1,0].set_ylabel('Combined singular ims')

axs[0,-1].imshow(reduce_FOV(np.sqrt(np.sum(np.abs(data['ZF'])[...,0]**2, axis=-1))), cmap='bone')
axs[0,-1].set_title('Zero filled')

# %%


# %%
plt.subplot(121)
plt.imshow(reduce_FOV(np.abs(np.mean(np.sum(im_singular[:,:,None,:] * UR[None,None,:,:], axis=-1), axis=-1))), cmap='bone')
# plt.imshow(reduce_FOV(np.abs(np.mean(np.sum(im_singular[:,:,None,:] * UR[None,None,:,:], axis=-1), axis=-1))), cmap='bone')
# el Vk se extrae en espacio de imagen, se obtiene siempre en imagen, y se puede generar en imagen pero tambien se puede aplicar en el espacio k, también se pueden comprimir en espacio k 
# probar para un par de bobinas, debería funcionar bien solo con una :smile 
# tratar las bobinas por separado 
# se obtiene el kspace por bobina 
plt.subplot(122)
plt.imshow(reduce_FOV(np.sqrt(np.sum(np.abs(data['ZF'])[...,0]**2, axis=-1))), cmap='bone')


# %%
plt.imshow()
# data['coil_rss'].shape#[0][:,0]#.shape #.keys()
# data_y.shape, spoke_radon_kspace.shape

# np.abs(np.sum(data_y[None,:]* np.conjugate(spoke_radon_kspace), axis=-1))

# plt.plot(np.abs(data_y))
# plt.plot(np.abs(spoke_radon_kspace).transpose(), alpha=0.3)
# np.abs(spoke_radon_kspace).max(), np.abs(train_Y[t_idx]).max(), np.abs(batch_frames).max()

# plt.imshow(np.abs(batch_frames[0]), cmap='bone')
# plt.imshow(np.abs(batch_frames[selected_tidx]),cmap='bone')

# %%

# np.sum(im_singular[...,0]/(2.11e-6) * np.conj(im_singular[...,1])/(2.11e-6))

# %%

# data.keys()

# %%

# np.abs(UR)[169]

# np.sum(UR[:,0] * np.conjugate(UR[:,1]))
# np.dot(UR[:,5], np.conjugate(UR[:,1]))
a_val = np.linalg.norm(im_singular, axis=(0,1))
np.sum(np.sum(im_singular[:,:,1]/a_val[1] * np.conjugate(im_singular[:,:,0])/a_val[0]))
