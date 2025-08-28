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
im_singular = data['MRF_Reg_sing'] # (n,n, nsing, echos)

# %%

scalings = np.percentile(np.abs(im_singular), 99, axis=(0,1)) # (nsing, echos)

# %%
# Mostrar las imágenes singulares para todos los echos 
n_ims = im_singular.shape[2]
fig, axs = plt.subplots(2,n_ims, figsize=(n_ims*3, 2*3))
for n_sing in range(n_ims):
    for echo_num in range(2):
        axs[echo_num, n_sing].imshow(np.abs(im_singular[...,n_sing, echo_num] / scalings[n_sing, echo_num]), vmax=2., cmap='bone')

axs[0,0].set_ylabel(f'Echo 1')
axs[1,0].set_ylabel(f'Echo 2')

for n_sing in range(n_ims):
    axs[0,n_sing].set_title(f'Imagen singular {n_sing}')

# %%

from jax import random 
# from inrmri.dip.tddip import Decoder
from inrmri.dip.unet import UNet

key = random.PRNGKey(0)
key_space, key_params, key_eval, key_train = random.split(key, 4)

# net = Decoder(features=128, levels=4, out_features=n_ims*2)
net = UNet(dimension=2, levels=3, output_features=n_ims*2, encoding_features=32, dropout_rate=0.05, skip_features=4, upsampling_method='nearest')
noise_scaling = 1.0
random_features = 1 
# input_noise = noise_scaling * random.normal(key_space, (8,8,random_features)) # (1, nx, ny, 32)
# input_noise = noise_scaling * random.uniform(key_space, (8,8,random_features)) # (1, nx, ny, 32)
zf_reco = np.abs(data['ZF'][...,0])
zf_reco = zf_reco / zf_reco.max()

params = net.init(key_params, zf_reco, training=False)
zf_reco.max(axis=(0,1))
# %% 
from inrmri.basic_nn import mse 

def get_im(params, key, training:bool): 
    out_im, updates = net.apply(params, zf_reco, training=training, rngs={'dropout':key}, mutable=['batch_stats']) 
    new_shape = out_im.shape[:2] + (n_ims,2)
    new_out_im = np.reshape(out_im, new_shape) # (nx, ny, 6) -> (nx, ny,3,2)
    new_out_im = new_out_im[...,0] + 1j * new_out_im[...,1] # (nx, ny, 3)
    new_out_im = new_out_im * data['coil_rss'][:,:,None]
    return new_out_im, updates 

# %%

plt.imshow(np.abs(get_im(params, key, training=False)[0][...,0]))

# %%
from inrmri.radon import calculate_angle
from jax import vmap 

from inrmri.basic_nn import weighted_loss 
from inrmri.new_radon import get_weight_freqs, ForwardRadonOperator
from inrmri.bart import calculate_hollow_mask

NFRAMES = 375 
N = 256 
SELECTED_COILS = np.arange(15)
IN_PHASE_ECHO = 0 
OUT_OF_PHASE_ECHO = 1 

SELECTED_ECHO = IN_PHASE_ECHO

trajs = data['kp'][IN_PHASE_ECHO].reshape((NFRAMES, N, 2)) * N # las out_of_phase siguen la trayectoria al reves en el k-space
# trajs = data['kpalternate'].reshape((NFRAMES, 256, 2))

# angles = np.mod(vmap(calculate_angle)(trajs) + np.pi, 2 * np.pi) # esto no funcionó 
angles = vmap(calculate_angle)(trajs) 
t_idxs = np.arange(NFRAMES)

train_X = np.stack([angles, t_idxs], axis=-1)
train_Y = data['kdata_Echoes'][...,SELECTED_ECHO,:] # creo que es igual a data['kdata'] # (n, frames, csmap)

train_Y = np.moveaxis(train_Y, 0, -1) # (frames, coils,ro)
train_Y = train_Y[:,SELECTED_COILS,:]
train_Y = train_Y * 3 / np.abs(train_Y).max()

radon_operator = ForwardRadonOperator(csmap=np.array(data['csm'].transpose((2,0,1))[SELECTED_COILS]), spclim=0.5)

# %%

fig, axs = plt.subplots(1,n_ims, figsize=(n_ims*3, 3))
for n_sing in range(n_ims):
    axs[n_sing].imshow(np.abs(im_singular[...,n_sing, SELECTED_ECHO] / scalings[n_sing, SELECTED_ECHO] * radon_operator.csmap[0]), vmax=1., cmap='bone')
axs[0].set_ylabel(f'Sing im Diego')
#np.abs(pred_sing_im).max() 

# %% 
UR = np.array(data['Ur'][SELECTED_ECHO]) # (frames, 6)
lmbda_tv = 0.

# scale_factor = 1e9
scale_factor = 1e7 # max en 5 

def get_k_space_from_singulars_in_kspace(pred_sing_im, angle, t_idx:int): 
    """
        Aplica el modelo forward radial para angulo alpha, combinando las 
        imagenes singulares en k-space 

    pred_sing_im: array complex (n,n,nsing)
    angle: shape ()
    t_idx: int 

    ## Output 

    - data_y: shape (cs, n)
    """
    pred_sing_ims = np.moveaxis(pred_sing_im, -1, 0) # (nsing, nx,ny)
    pred_sing_ims = pred_sing_ims*scalings[:,None,None, SELECTED_ECHO]*scale_factor
    repeated_angle = np.ones(6)*angle
    rotated_ims = radon_operator.rotated_csweighted_ims(pred_sing_ims, repeated_angle)
    kspaces = radon_operator.radon_transform(rotated_ims) # (nsing, csmap, n)
    frame_UR = UR[t_idx,:,None,None] # (nsing,1,1)
    # frame_UR = np.conjugate(UR)[t_idx,:,None,None]
    data_y = np.sum(frame_UR * kspaces, axis=0) # (csmap, n)
    return data_y 

def get_k_space_from_singulars_in_im(pred_sing_im, angle, t_idx:int): 
    """
        Aplica el modelo forward radial para angulo alpha, combinando las 
        imagenes singulares en espacio de imagen 

    pred_sing_im: array complex (n,n,nsing)
    angle: shape ()
    t_idx: int 

    ## Output 

    - data_y: shape (cs, n)
    """
    pred_sing_ims = np.moveaxis(pred_sing_im, -1, 0) # (nsing, nx,ny)
    #pred_sing_ims = pred_sing_ims * scalings[:,None,None, SELECTED_ECHO] * 1000000
    pred_sing_ims = pred_sing_ims*scalings[:,None,None, SELECTED_ECHO]*scale_factor
    # frame_UR = UR[t_idx,:,None,None] # (nsing,1,1)
    frame_UR = np.conjugate(UR)[t_idx,:,None,None]
    combined_im = np.sum(pred_sing_ims * frame_UR, axis=0)[None] # (nsing, nx, ny)-> (1, nx, ny)
    rotated_ims = radon_operator.rotated_csweighted_ims(combined_im, angle[None])[0] # (coil, n, n)
    data_y = radon_operator.radon_transform(rotated_ims) # (cs, n)
    return data_y 

# WEIGHT_FREQS = get_weight_freqs(N, 'ramp')[None,None,:]
# WEIGHT_FREQS   = (get_weight_freqs(N, 'ramp') / 128)[None,:] * data['dcf'][0][0,:][:,None] # (frames, ro)
# WEIGHT_FREQS   = np.moveaxis(data['dcf'][0], -1, 0)[:, None, :] # (frames, 1, ro)
WEIGHT_FREQS = get_weight_freqs(N, 'ramp')[None,None,:] * data['dcf'][0][0,:][:, None, None]
# WEIGHT_FREQS = np.ones(N)[None,None,:]

#hollow_mask = calculate_hollow_mask(radon_operator.csmap) # el Diego tiene una variable data['coil_rss'] que es 1-hollow_mask

def loss(params, X, Y, key): 
    batch_size = X.shape[0]
    alphas, times = X[:,0], X[:,1]
    t_idx = np.int32(times)
    sing_ims, updates = get_im(params, key, training=True) # (nx, ny, 6)
    
    spoke_radon_kspace = vmap(get_k_space_from_singulars_in_im, in_axes=(None,0,0))(sing_ims, alphas, t_idx) #* np.sqrt(1 + WEIGHT_FREQS[t_idx])
    print("shapes: spoke ", spoke_radon_kspace.shape, " , Y ", Y.shape)
    loss_value = weighted_loss(spoke_radon_kspace, Y, 1 + WEIGHT_FREQS[t_idx])
    # loss_value = weighted_loss(spoke_radon_kspace, Y, 1.)
    tv_val = 0 #TV_space(np.moveaxis(ims, 0, -1))
    return loss_value + lmbda_tv*tv_val, updates 

# %%
from inrmri.fourier import fastinvshiftfourier

t_idx = 170
angle = np.mod(train_X[t_idx,0], 2 * np.pi)
#angle = np.mod(np.pi/2 + train_X[t_idx,0], 2 * np.pi)
maxv = np.abs(im_singular).max()
data_y_1 = get_k_space_from_singulars_in_kspace(im_singular[...,SELECTED_ECHO], angle, t_idx)
data_y_2 = get_k_space_from_singulars_in_im(im_singular[...,SELECTED_ECHO], angle, t_idx)

selected_coil = 8
funcs = [np.abs, np.real, np.imag]

from inrmri.new_radon import radon_integration
fig, axs = plt.subplots(1,3, figsize=(18,3))

combined_im = np.sum(im_singular[:,:,:,SELECTED_ECHO] * UR[t_idx][None,None,:], axis=-1) # (n,n,nsing)->(n,n)
# combined_im = np.sum(im_singular[:,:,:,SELECTED_ECHO] * np.conjugate(UR[t_idx])[None,None,:], axis=-1) # (n,n,nsing)->(n,n)
rotated_im =  radon_operator.rotated_csweighted_ims(combined_im[None,:,:], angle[None])[0] # (csmap, n,n)
sinogram = radon_integration(rotated_im, radon_operator.ds) # (csmap, N)

for ax, fun in zip(axs, funcs): 
    ax.plot(fun(fastinvshiftfourier(train_Y[t_idx,selected_coil])))
    ax.plot(fun(sinogram[selected_coil] * 256 * 1e6))

from inrmri.fourier import fastshiftfourier
from jax.numpy.fft import fft, fftshift, ifftshift, ifft

a = fftshift(fft(sinogram[selected_coil] *256 * 1e6, norm='forward'))

fig, axs = plt.subplots(1,3, figsize=(18,3))
for ax, fun in zip(axs, funcs): 
    ax.plot(fun(train_Y[t_idx,selected_coil]))
    ax.plot(fun(a))

# %%

learning_rate = 5e-3
optimizer = OptimizerWithExtraState(optax.adam(learning_rate))

total_citer = 10 # 1000 iteraciones 

train_Y_v2 = np.load('/mnt/storage/data-MRF/2025-07-30_synthetic-kspace/train_Y.npy')
train_Y_v2 = train_Y_v2 * 3 / np.abs(train_Y_v2).max()

results = train_with_updates(jit(loss), train_X, train_Y_v2, params, optimizer, key_train, nIter = total_citer * 100 + 1, batch_size = 30, save_at=200)
# all_res['water-simple-kspace'] = results
# echos_im[SELECTED_ECHO] = results

# %%

selected_kit = 4

# for results_name, results in all_res.items():
final_param = results['param_history'][f'param-{selected_kit}']
pred_sing_im, _ = get_im(final_param, key, False)
print(np.abs(pred_sing_im).max(axis=(0,1)))

vmax = 0.6 * np.abs(pred_sing_im).max()

from inrmri.image_processor import compute_scaling_factor

factor = compute_scaling_factor(np.abs(pred_sing_im[...,0]), np.abs(im_singular[...,0,SELECTED_ECHO]/scalings[0,SELECTED_ECHO]))

nrows = 3
fig, axs = plt.subplots(nrows,n_ims, figsize=(n_ims*3, nrows*3))
for n_sing in range(n_ims):
    im_sing_pred = reduce_FOV(np.abs(pred_sing_im[...,n_sing]))
    axs[0,n_sing].imshow(im_sing_pred, cmap='bone', vmax=vmax)
    im_sing = reduce_FOV(np.abs(im_singular)[...,n_sing, SELECTED_ECHO]/scalings[n_sing,SELECTED_ECHO])
    axs[1,n_sing].imshow(im_sing, cmap='bone', vmax= vmax/factor)
    axs[2,n_sing].imshow(np.abs(im_sing - im_sing_pred/factor), cmap='gray', vmax= 0.1 * vmax/factor)

axs[1,0].set_ylabel('Singulares del Diego')

# %%

plt.subplot(121)
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(pred_sing_im[...,2]), axes=(0,1)))))
plt.subplot(122)
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(im_singular[...,2,0]), axes=(0,1)))))

# %% 
final_pred = pred_sing_im/factor * scalings[None,None,:,SELECTED_ECHO]

error_rel = np.mean(np.abs(final_pred - im_singular[...,SELECTED_ECHO]), axis=(0,1)) / np.percentile(np.abs(im_singular[...,SELECTED_ECHO]), 99, axis=(0,1))

# np.save(data_folder + 'pred_ft64_3k.npy', final_pred)
# %%
# Comparar dataset sintético con original en distintos frames 
selected_coil = 5
frame_list = [53, 100, 163, 231, 286, 326]
for selected_tidx in frame_list:

    funcs = [np.abs, np.real, np.imag]
    fig, axs = plt.subplots(1,3, figsize=(15,3))
    for ax, fun in zip(axs, funcs):
        ax.plot(fun(train_Y[selected_tidx, selected_coil]), label='original')
        ax.plot(fun(train_Y_v2[selected_tidx, selected_coil]), label='synthetic') 
    plt.suptitle(f'Frame {selected_tidx}, coil {selected_coil}')
    plt.legend()
    plt.tight_layout()

# %%

fig, axs = plt.subplots(1, len(frame_list), dpi=200, sharex=True, sharey=True)
for t_idx, ax in zip(frame_list, axs):
    
    key = random.key(t_idx)
    noise = 2e-6 * random.normal(key, (2, 15, 256))
    noise = noise[0] + 1j * noise[1] # (cs, ro)
    ax.imshow(np.abs(noise.transpose()))
    ax.set_title(f'frame {t_idx}')
    ax.set_xlabel('coil')
    ax.set_ylabel('read out')

plt.tight_layout()

# %%
np.percentile(np.abs(final_pred), 99, axis=(0,1)), np.percentile(np.abs(im_singular[...,SELECTED_ECHO]), 99, axis=(0,1)) 

# %%
plt.imshow(np.abs(final_pred[...,4]) * (1-hollow_mask))
# plt.imshow(hollow_mask)

# %%
from matplotlib.lines import Line2D

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

pair_list = [(60,40), (74, 50), (68, 66), (53,60)]

fig, axs = plt.subplots(2,1 + len(pair_list), figsize=(5*(1 + len(pair_list)), 6), dpi=200)

plt.sca(axs[0,0])
plt.imshow(np.abs(reduce_FOV(im_singular[...,0,0])), cmap='bone')
for color, (px, py) in zip(colors, pair_list): 
    plt.scatter(px, py, color=color, marker='x')

for color, (px, py), ax in zip(colors, pair_list, axs[0,1:]): 
    plt.sca(ax)
    plt.plot(factor * np.abs(reduce_FOV(im_singular[...,0])[py, px]), '.-', color=color)
    plt.plot(np.abs(reduce_FOV(pred_sing_im * scalings[None,None,:,0])[py, px]), '.--', color=color)
    plt.ylim(0, 3.5e-6)

for color, (px, py), ax in zip(colors, pair_list, axs[1,1:]): 
    plt.sca(ax)
    pix_vals_sing = factor * reduce_FOV(im_singular[...,0])[py, px]
    pix_evo_sing = np.sum(pix_vals_sing[None,:] * np.conjugate(UR), axis=-1) 

    pix_vals_pred = reduce_FOV(pred_sing_im * scalings[None,None,:,0])[py, px]
    pix_evo_pred = np.sum(pix_vals_pred[None,:] * np.conjugate(UR), axis=-1)
    plt.plot(pix_evo_sing, '-', color=color)
    plt.plot(pix_evo_pred, '--', color=color)
    plt.ylim(0, 0.12e-6)

custom_lines = [Line2D([0], [0], linestyle ='-', color='gray'),
                Line2D([0], [0], linestyle ='--', color='gray')]

plt.legend(custom_lines, ['singulares diego', 'DIP'])
# %%


# %% 
# plt.imshow(data['dcf'][0])#.shape 
# plt.imshow(np.abs(batch_frames[...,0]), cmap='bone')
# plt.plot(data['dcf'][0][:,150])
# plt.plot(data['dcf'][0][:,200])
# plt.plot(data['dcf'][0][:,250])

plt.plot()
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
