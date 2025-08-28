# Funciones útiles para entrenar la red con datos radiales en el k-espacio, 
# usando la Transformada de Radon 

from jax.scipy.interpolate import RegularGridInterpolator as RGI
import jax.numpy as np
from jax import random, grad, jit, vmap
from inrmri.fourier import fastshiftfourier, get_freqs 
from inrmri.radon import _radon_points, calculate_angle, get_ds
from inrmri.utils import is_inside_of_radial_lim
from inrmri.basic_nn import weighted_loss


#-----------------------------------------------------------------------
# TrainX functions 
#-----------------------------------------------------------------------

def space_sampling_from_ksampling(k_samplinginterval_len, k_nsampledpoints):
  return 1/(k_samplinginterval_len * k_nsampledpoints)

def number_of_spokes(train_data_X): 
  return train_data_X.shape[0]

def samples_per_spoke(train_data_X):
  return train_data_X.shape[1]

def get_spoke_coords(train_data_X, m):
  """
  points of mth spoke, (n,2)  
  """
  return train_data_X[m]

def are_centered(points): 
  """
  - points: (n,2)
  """
  flipped = np.flip(points, axis = 0)
  return np.allclose(points + flipped, 0., atol = 1e-06)

def iscentered(train_data_X):
  return np.all(np.array(
    [are_centered(get_spoke_coords(train_data_X, i)) for i in range(number_of_spokes(train_data_X))]
    )
)

def interval_limit_from_sampling(samplinginterval, nsampledpoints):
  """
  Calcula la distancia al origen del punto mas lejano para un sampling centrado
  (los puntos forman una linea que pasa por 0 en el centro)
  """
  lim = (nsampledpoints - 1) * samplinginterval / 2 
  return lim 

def ks_spacing_from_spoke(spoke):
  return np.linalg.norm(spoke[1] - spoke[0])

def ks_inverval_len_from_spoke(spoke):
  return np.linalg.norm(spoke[-1] - spoke[0])

def ks_spacing(train_data_X): 
  return ks_spacing_from_spoke(get_spoke_coords(train_data_X, 0))

def spacelim(train_data_X): 
  ksamples = samples_per_spoke(train_data_X)
  dk = ks_spacing(train_data_X)
  
  if not iscentered(train_data_X): 
    print("Warning: not centered data")
  spacelim = interval_limit_from_sampling(space_sampling_from_ksampling(dk, ksamples), ksamples)
  print(f"space lims: ({-spacelim:.2f}, {spacelim:.2f})")
  return spacelim 


#-----------------------------------------------------------------------
# Net Radon Loss
#-----------------------------------------------------------------------

def interpolate_points_to_grid(points, csmap): 
  """
  - points: narray [...,2]
  - csmap: array 2d (px,py)
  """
  Nx, Ny = csmap.shape[:2]
  x, y = np.linspace(-1,1,Nx, endpoint = False), np.linspace(-1,1,Ny, endpoint = False)
  interpolator = RGI((x,y), csmap, bounds_error = False, fill_value = 0.)
  return interpolator(points)

def coils_radon_transform(params, t, alpha, N, csmaps, FFnet): 
  lims = (-1., 1.)
  ss, radonxy, perpendicularxy = _radon_points(alpha, lims, N)
  radonpoints = (radonxy[:,None,:] + perpendicularxy[None,:,:])
  print("radonpoints shape", radonpoints.shape)
  is_inside = is_inside_of_radial_lim(radonpoints, lims[1])
  print("is_inside shape", is_inside.shape)
  NNx = FFnet.eval_x_and_t(params, radonpoints, t)
  #NNx = to_complex(FF_periodtime_net_forward_pass_not_sqeezed(params, concat(radonpoints, t), B))
  print("NNx shape", NNx.shape)
  # csmap = csmaps[calculate_time_index(t, csmaps.shape[0])]
  print("csmaps shape", csmaps.shape)
  ds = get_ds(ss)
  coil_sens_radon = lambda csmap: np.sum(NNx * interpolate_points_to_grid(radonpoints,csmap)[:,:,None] * is_inside[:,:, None] * ds, axis = 1)
  print(" coil_sens_radon (cmap) shape", coil_sens_radon(csmaps[0]).shape)
  return vmap(coil_sens_radon, in_axes = 0)(csmaps)

def spoke_coil_radon(params, t, spoke, spacelim, scalingparam, csmaps, FFnet):
  alpha = calculate_angle(spoke)
  print("alpha shape:", alpha.shape)
  # print(f"alpha: {alpha/ np.pi:.3}π")
  N = spoke.shape[0] 
  radon = coils_radon_transform(params, t, alpha, N, csmaps, FFnet) 
  print('radon shape', radon.shape)
  radon = np.squeeze(radon, axis = -1) 
  L = 2 * spacelim 
  return fastshiftfourier(L * scalingparam * radon )[:,:,None]

def spoke_loss_fourierspace_phase(params, data_spoke_X, data_Y, spacelim, scalingparam, csmaps, FFnet, filter_name:str):
  t = data_spoke_X[0,0]
  spoke_X = data_spoke_X[1:,:]
  print("spoke shape: ", spoke_X.shape)
  sffrad = spoke_coil_radon(params, t, spoke_X, spacelim, scalingparam, csmaps, FFnet)
  print(f"data_Y.shape: {data_Y.shape}")
  print(f"sffrad.shape: {sffrad.shape}")
  N = data_Y.shape[1]
  freqs = np.abs(get_freqs(N))
  if filter_name == 'ramp':
    filter_freqs = np.ones(N)
  elif filter_name == 'shepp-logan':
    filter_freqs = np.fft.fftshift(np.sinc(np.fft.fftfreq(N))) # shepp logan filter 
  elif filter_name == 'cosine':
    filter_freqs = np.sin(np.linspace(0, np.pi, N, endpoint=False)) # cosine filter 
  print(f"filter freqs.shape: {filter_freqs.shape}")
  print(f"freqs.shape: {freqs.shape}")
  loss = weighted_loss(sffrad, data_Y, (1. + filter_freqs * freqs)[None,:,None])
  return loss 