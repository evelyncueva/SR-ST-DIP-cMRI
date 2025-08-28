import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Tuple, Sequence
from jax import random 
from functools import partial

# %%

# ---------------------------------------------------------------------- # 
# LR-DIP 
# ---------------------------------------------------------------------- # 
class ConvolutionBlock(nn.Module): # blue arrow 
  dimensions    : int # 1, 2
  kernel        : int # 3, 1
  stride        : int
  features      : int
  dropout_rate  : float

  @nn.compact
  def __call__(self, x, training:bool):
    # voy a implementarla primero sin considerar las skip connections 
    x = nn.Conv(features=self.features, kernel_size=(self.kernel,) * self.dimensions, strides=(self.stride,)*self.dimensions)(x)
    x = nn.leaky_relu(x)
    x = nn.BatchNorm(use_running_average = not training)(x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
    return x   
  
class ForwardConvolution(nn.Module): 
    dimensions  : int # 1, 2
    kernel      : int # 3, 1
    features    : int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, training:bool):
        x = ConvolutionBlock(dimensions=self.dimensions,
                            kernel       = self.kernel,
                            stride       = 1,
                            features     = self.features, 
                            dropout_rate = self.dropout_rate
                            )(x, training)
        return x 
   
class DownwardConvolution(nn.Module):
    dimensions    : int # 1, 2
    features    : int
    dropout_rate: float 

    @nn.compact
    def __call__(self, x, training:bool):
        x = ConvolutionBlock(
            dimensions=self.dimensions,
            kernel       = 3,
            stride       = 2,
            features     = self.features, 
            dropout_rate = self.dropout_rate
            )(x, training)
        return x   
    
def upsampling_1d(t, new_shape:int, method:str): 

    old_lenght, channels = t.shape[-2:]
    batch_shape = t.shape[:-2]
    vectorsize = new_shape[-2]

    newt = jax.image.resize(t, shape= batch_shape + (vectorsize, channels), method=method)
    return newt

def upsampling_2d(x, new_imshape:Tuple[int,int], method:str):
   old_height, old_width, channels = x.shape[-3:]
   batch_shape = x.shape[:-3]
   imsize = new_imshape[-3:-1]
   new_height, new_width = imsize

   x = jax.image.resize(x, shape= batch_shape + (new_height, new_width, channels), method=method)
   return x   

class UNet(nn.Module):
  dimension         : int
  dropout_rate      : float
  encoding_features : int # 128 
  skip_features     : int # 4 
  levels            : int # 4 
  upsampling_method : str # bilinear, nearest 
  output_features   : int
  @nn.compact
  def __call__(self, x, training:bool):

    skips = []
    shapes = []

    # encode 
    for _ in range(self.levels):
        skip = ForwardConvolution(dimensions=self.dimension, kernel=3, features=self.skip_features, dropout_rate=self.dropout_rate)(x, training)
        skips.append(skip)
        
        shapes.append(x.shape)
        x = DownwardConvolution(dimensions=self.dimension, features=self.encoding_features, dropout_rate=self.dropout_rate)(x, training)
        
        x = ForwardConvolution(dimensions=self.dimension, kernel=3, features=self.encoding_features, dropout_rate=self.dropout_rate)(x, training)

    skips.reverse()
    shapes.reverse()
    # decode         
    for skip, shape in zip(skips, shapes):
        x = self.upsampling(x, shape)
        x = jnp.concatenate((skip, x), axis=-1) # concatenar los features 

        x = ForwardConvolution(dimensions=self.dimension, kernel=3, features=self.encoding_features, dropout_rate=self.dropout_rate)(x, training)
        x = ForwardConvolution(dimensions=self.dimension,kernel=1, features=self.encoding_features, dropout_rate=self.dropout_rate)(x, training)
    x = ForwardConvolution(dimensions=self.dimension, kernel=1, features=self.output_features, dropout_rate=self.dropout_rate)(x, training)    
    return x 
  
  def upsampling(self, x, newshape):
    if self.dimension == 1:
       x = upsampling_1d(x, newshape, self.upsampling_method)
    elif self.dimension == 2:
       x = upsampling_2d(x, newshape, self.upsampling_method)
    return x
    
# %%

def split_last_dim(an_array):
   old_shape = an_array.shape
   new_shape = old_shape[:-1] + (old_shape[-1]//2, 2)
   return an_array.reshape(new_shape)

def to_complex(an_array): 
   real, imag = jnp.split(an_array, 2, axis=-1)
   return real + 1j * imag 

class SpaceTime_LR_UNet():

    def __init__(self, 
                nframes           : int, # usualmente entre 10 y 30 
                space_resolution  : Tuple[int,int],
                key               : jax.Array,
                dropout_rate      : float = 0.1,
                encoding_features : int = 128,
                skip_features     : int = 4,
                levels            : int = 4,
                upsampling_method : str = 'nearest',
                output_features   : int = 2000,
                input_features    : int = 32,
                noise_scaling     : float = 0.1):

        self.space_unet  = UNet(2, dropout_rate, encoding_features, skip_features, levels, upsampling_method, output_features)
        self.time_unet   = UNet(1, dropout_rate, encoding_features, skip_features, levels, upsampling_method, output_features)
         
        key_space, key_time = random.split(key, 2)

        self.noise_scaling = noise_scaling 
        self.space_resolution = space_resolution
        self.nframes = nframes
        self.input_features = input_features 

        self.latent_z_space = noise_scaling * random.normal(key_space, (1,) + space_resolution + (input_features,)) # (1, nx, ny, 32)
        self.latent_z_time  = noise_scaling * random.normal(key_time, (1, nframes, input_features,)) # (1, nframes, 32)

    def make_dict_param(self,params_space, params_time):
        params = {
           'space_unet': params_space['params'], 
           'time_unet': params_time['params']
        }

        batch_stats = {
           'space_unet': params_space['batch_stats'],
           'time_unet': params_time['batch_stats']
        }
        
        full_params = {'params': params, 'batch_stats': batch_stats}
        return full_params 

    def split_params(self, dict_param): 
       params = dict_param['params']
       batch_stats = dict_param['batch_stats']
       params_space = {
          'batch_stats':batch_stats['space_unet'], 
          'params': params['space_unet']
       }
       params_time = {
          'batch_stats':batch_stats['time_unet'], 
          'params': params['time_unet']
       }
       return params_space, params_time

    def init_params(self, key: jax.Array):

        key_space, key_time = random.split(key, 2)
        
        params_space = self.space_unet.init(key_space, self.latent_z_space, training=False)
        params_time = self.time_unet.init(key_time, self.latent_z_time, training=False)

        return self.make_dict_param(params_space, params_time)
    
    def train_eval_space(self, params_space, key): 
        y_space, updates = self.space_unet.apply(params_space, self.latent_z_space, training=True, rngs={'dropout':key}, mutable=['batch_stats'])
        batch_stats_space = updates['batch_stats']
        return y_space, batch_stats_space
    
    def train_eval_time(self, params_time, key, t_index_batch): 
        if t_index_batch is None:
            latent = self.latent_z_time 
        else:
            latent = self.latent_z_time[:,t_index_batch,:]
        y_time, updates = self.time_unet.apply(params_time, latent, training=True, rngs={'dropout':key}, mutable=['batch_stats'])
        batch_stats_time = updates['batch_stats']
        return y_time, batch_stats_time
    
    def combine_space_time(self, y_space, y_time): 
        y_space = to_complex(split_last_dim(y_space))[0,...,0] # (nx, ny, outfeatures//2)
        y_time = to_complex(split_last_dim(y_time))[0,...,0] # (nframes, outfeatures//2)

        # im = y_space[:,:,None,:] * jnp.conjugate(y_time[None,None,:,:])
        im = y_space[:,:,None,:] * y_time[None,None,:,:] # (nx, ny, nframes, outfeatures//2)
        im = jnp.mean(im, axis=-1) # (nx, ny, nframes)
        im = jnp.moveaxis(im, -1, 0) # (nframes, nx, ny)
        return im 

    def train_basis_and_update(self, params, key, t_index=None):
        params_space, params_time = self.split_params(params)
        key_dropout_space, key_dropout_time = random.split(key)

        y_space, batch_stats_space = self.train_eval_space(params_space, key_dropout_space)
        y_time, batch_stats_time = self.train_eval_time(params_time, key_dropout_time, t_index)

        batch_stats = {
           'space_unet': batch_stats_space,
           'time_unet': batch_stats_time
        }
        return y_space, y_time, batch_stats
    
    def train_forward_pass(self, params, key, t_index=None):
        y_space, y_time, batch_stats = self.train_basis_and_update(params, key, t_index)
        im = self.combine_space_time(y_space, y_time)
        return im, batch_stats 
    
    def train_forward_with_exponential_reg(self, params, key, old_basis, alpha):
        y_space_old, y_time_old = old_basis
        y_space, y_time, batch_stats = self.train_basis_and_update(params, key)
        y_space_new = alpha * y_space + (1-alpha) * y_space_old
        y_time_new = alpha * y_time + (1-alpha) * y_time_old
        new_basis = (y_space_new, y_time_new)
        im = self.combine_space_time(y_space_new, y_time_new) # no está interfiriendo con el entrenamiento 
        return im, (batch_stats, new_basis)
    
    def forward_pass_with_updating(self, params, key, t_index=None):

        im, batch_stats = self.train_forward_pass(params, key, t_index)
        params['batch_stats'] = batch_stats
        return im 
    
    def inference_image(self, fullparam, t_index_batch):
        params_space, params_time = self.split_params(fullparam)
        if t_index_batch is None:
            latent_t = self.latent_z_time 
        else:
            latent_t = self.latent_z_time[:,t_index_batch,:]
            print(latent_t.shape)
        y_space = self.space_unet.apply(params_space, self.latent_z_space, training=False)
        y_time = self.time_unet.apply(params_time, latent_t, training=False)
        im = self.combine_space_time(y_space, y_time)
        return im 
        
 
# ---------------------------------------------------------------------- # 
# TD-DIP 
# ---------------------------------------------------------------------- # 
    
class ConvolutionalDIPBLock(nn.Module): 
    dimensions    : int # 1, 2
    kernel        : int # 3, 1
    stride        : int
    features      : int
    momentum      : float

    @nn.compact
    def __call__(self, x, training:bool):
        # voy a implementarla primero sin considerar las skip connections 
        x = nn.Conv(features=self.features, kernel_size=(self.kernel,) * self.dimensions, strides=(self.stride,)*self.dimensions)(x)
        x = nn.BatchNorm(use_running_average = not training, momentum=self.momentum)(x)
        x = nn.relu(x)
        return x
    
class Encoder(nn.Module):
    features      : int = 128
    momentum      : float = 0.99
    levels        : int = 3
    out_features  : int = 128
    upsampling_method: str = 'nearest'
    dimensions    : int = 2 
    @nn.compact
    def __call__(self, x, training:bool):
        # voy a implementarla primero sin considerar las skip connections
        downsampling_factor = 2
        x = ConvolutionalDIPBLock(dimensions=self.dimensions, kernel=3, stride=1, features=self.features, momentum=self.momentum)(x, training) #(NxNxfeatures)
        x = ConvolutionalDIPBLock(dimensions=self.dimensions, kernel=3, stride=1, features=self.features, momentum=self.momentum)(x, training) #(NxNxfeatures)
        x = ConvolutionalDIPBLock(dimensions=self.dimensions, kernel=3, stride=downsampling_factor, features=self.features, momentum=self.momentum)(x, training) #(N//2xN//2xfeatures)
        for _ in range(self.levels):
            x = ConvolutionalDIPBLock(dimensions=self.dimensions, kernel=3, stride=1, features=self.features, momentum=self.momentum)(x, training) #(NxNxfeatures)
            x = ConvolutionalDIPBLock(dimensions=self.dimensions, kernel=3, stride=1, features=self.features, momentum=self.momentum)(x, training) #(NxNxfeatures)
            x = ConvolutionalDIPBLock(dimensions=self.dimensions, kernel=3, stride=downsampling_factor, features=self.features, momentum=self.momentum)(x, training) #(N//2xN//2xfeatures)
        x = ConvolutionalDIPBLock(dimensions=self.dimensions, kernel=3, stride=1, features=self.features, momentum=self.momentum)(x, training) #(NxNxfeatures)
        x = ConvolutionalDIPBLock(dimensions=self.dimensions, kernel=3, stride=1, features=self.features, momentum=self.momentum)(x, training) #(NxNxfeatures)
        x = nn.Conv(features=self.out_features, kernel_size=(3,)*self.dimensions, strides=(downsampling_factor,)*self.dimensions)(x)
        return x
    

    
def new_upsampled_shape(initialshape, upsampling_factor:int, dimensions:int): 
    """
    # Ejemplos: 

    ```python
    >>> new_upsampled_shape((100,12,10,30,3), 3, 1)
    (100,12,10,90,3)
    >>> new_upsampled_shape((100,12,10,30,3), 3, 2)
    (100,12,30,90,3)
    >>> new_upsampled_shape((100,12,10,30,3), 3, 3)
    (100,36,30,90,3)
    >>> new_upsampled_shape((100,12,10,30,3), 2, 3)
    (100,24,20,60,3)
    ```
    """
    batch_shape = initialshape[:-(dimensions + 1)]
    convolved_shape = initialshape[-(dimensions + 1):-1]
    features_shape = (initialshape[-1],)
    new_shape = batch_shape  + tuple(n* upsampling_factor for n in convolved_shape) + (initialshape[-1],)
    return new_shape

class Decoder(nn.Module):
    features         : int = 128
    momentum         : float = 0.99
    levels           : int = 3
    out_features     : int = 2
    upsampling_method: str = 'nearest'
    dimensions       : int = 2 
    upsampling_factor: int = 2 

    @nn.compact
    def __call__(self, x, training:bool):
        # voy a implementarla primero sin considerar las skip connections 
        x = ConvolutionalDIPBLock(dimensions=self.dimensions, kernel=3, stride=1, features=self.features, momentum=self.momentum)(x, training) #(8x8x128)
        x = ConvolutionalDIPBLock(dimensions=self.dimensions, kernel=3, stride=1, features=self.features, momentum=self.momentum)(x, training) #(8x8x128)
        x = self.upsampling(x, self.upsample_shape(x.shape)) # #(16x16x128)
        for _ in range(self.levels): 
            x = ConvolutionalDIPBLock(dimensions=self.dimensions, kernel=3, stride=1, features=self.features, momentum=self.momentum)(x, training) #(nxnx128)
            x = ConvolutionalDIPBLock(dimensions=self.dimensions, kernel=3, stride=1, features=self.features, momentum=self.momentum)(x, training) #(nxnx128)
            x = self.upsampling(x, self.upsample_shape(x.shape)) # (2nx2nx128)
        x = ConvolutionalDIPBLock(dimensions=self.dimensions, kernel=3, stride=1, features=self.features, momentum=self.momentum)(x, training) #(128x128x128)
        x = ConvolutionalDIPBLock(dimensions=self.dimensions, kernel=3, stride=1, features=self.features, momentum=self.momentum)(x, training) #(128x128x128)
        x = nn.Conv(features=self.out_features, kernel_size=(3,)*self.dimensions, strides=(1,)*self.dimensions)(x)
        return x

    def upsampling(self, x, newshape):
        if self.dimensions == 1: 
            return upsampling_1d(x, newshape, self.upsampling_method)
        if self.dimensions == 2: 
            return upsampling_2d(x, newshape, self.upsampling_method)
    
    def upsample_shape(self, initialshape):
        return new_upsampled_shape(initialshape, self.upsampling_factor, self.dimensions)

import flax.linen as nn

class LightDecoder(nn.Module):
    features         : int = 128   # constant features at all levels
    momentum         : float = 0.99
    levels           : int = 3
    out_features     : int = 2
    dimensions       : int = 2
    upsampling_factor: int = 2

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Conv(
            features=self.features,
            kernel_size=(3,) * self.dimensions,
            strides=(1,) * self.dimensions,
            padding='SAME'
        )(x)
        x = nn.BatchNorm(
            use_running_average=not training,
            momentum=self.momentum
        )(x)
        x = nn.relu(x)

        for _ in range(self.levels+1):
            # Upsample with ConvTranspose
            x = nn.ConvTranspose(
                features=self.features,
                kernel_size=(3,) * self.dimensions,
                strides=(self.upsampling_factor,) * self.dimensions,
                padding='SAME'
            )(x)
            x = nn.BatchNorm(
                use_running_average=not training,
                momentum=self.momentum
            )(x)
            x = nn.relu(x)

            # Extra conv layer after upsampling
            x = nn.Conv(
                features=self.features,
                kernel_size=(3,) * self.dimensions,
                strides=(1,) * self.dimensions,
                padding='SAME'
            )(x)
            x = nn.BatchNorm(
                use_running_average=not training,
                momentum=self.momentum
            )(x)
            x = nn.relu(x)

        # Final output conv
        x = nn.Conv(
            features=self.out_features,
            kernel_size=(3,) * self.dimensions,
            strides=(1,) * self.dimensions,
            padding='SAME'
        )(x)

        return x



import copy

class MapNet(nn.Module):
    mapnet_layers:Sequence[int] # solo considera los hidden layers 
    cnn_latent_shape:Tuple[int,int]

    def setup(self):
        layers = list(copy.deepcopy(self.mapnet_layers))
        if layers: #not empty 
            px, py = self.cnn_latent_shape
            layers.append(px * py)      

        self.dense_layers = [nn.Dense(layer, name=f'mapnet-{i}') for i, layer in enumerate(layers)]

    def __call__(self, t):
        for dense_layer in self.dense_layers: 
            t = dense_layer(t) # (batch, features)
            t = nn.relu(t)
        return t 

class tDIP(nn.Module):
    mapnet_layers : Sequence[int]
    cnn_latent_shape : Tuple[int,int]
    features      : int 
    momentum      : float
    levels        : int

    @nn.compact
    def __call__(self, t, training:bool):
        print("[TRACING] Recompiling tDIP...")
        mapnet = MapNet(self.mapnet_layers, self.cnn_latent_shape)
        x = mapnet(t)
        x = jnp.reshape(x, x.shape[:-1] + self.cnn_latent_shape)
        x = x[...,None] # add features dimension
        x = Decoder(self.features, self.momentum, self.levels)(x, training)
        return x 

# ------------------- Generators --------------------------------------- # 

def helix_generator(nframes, total_cycles):
    ts = jnp.linspace(0,total_cycles, nframes, endpoint=False)
    helix = jnp.stack([jnp.cos(ts*2*jnp.pi), jnp.sin(ts*2*jnp.pi), ts/total_cycles], axis=-1)
    return helix

def random_generator(nframes, key, addConst):
    latent_rep = jax.random.normal(key, (nframes, 2))
    return latent_rep

def multi_slice_random_generator(nframes, num_slices, key, addConst):
    rand_values = jax.random.normal(key, (nframes, 2))
    ss = jnp.linspace(-1, 1, num_slices, endpoint=True)
    constant_value = jax.random.uniform(key, ())
    arrays_to_stack = []
    for s in ss:
        if addConst:
            circle_s = jnp.stack([rand_values[:,0], rand_values[:,1], s*jnp.ones(nframes), constant_value*jnp.ones(nframes)], axis=-1)
        else:
            circle_s = jnp.stack([rand_values[:,0], rand_values[:,1], s*jnp.ones(nframes)], axis=-1)
        arrays_to_stack.append(circle_s)
    stacked_array = jnp.stack(arrays_to_stack, axis=0)
    return stacked_array

def class_generator(nframes, key, addConst):
    latent_rep = np.eye(nframes)
    return latent_rep

def circle_generator(nframes, key, addConst, radius=1.0):
    ts = jnp.linspace(0, 1, nframes, endpoint=False)
    constant_value = jax.random.uniform(key, ())
    x = radius * jnp.cos(ts * 2 * jnp.pi)
    y = radius * jnp.sin(ts * 2 * jnp.pi)
    if addConst:
        circle = jnp.stack([x, y, constant_value * jnp.ones(nframes)], axis=-1)
    else:
        circle = jnp.stack([x, y], axis=-1)
    return circle

def multi_slice_circle_generator(nframes, num_slices, key, addConst, radius=1.0, z_min=-1.0, z_max=1.0):
    ts = jnp.linspace(0, 1, nframes, endpoint=False)
    ss = jnp.linspace(z_min, z_max, num_slices, endpoint=True)
    constant_value = jax.random.uniform(key, ())
    arrays_to_stack = []
    x = radius * jnp.cos(ts * 2 * jnp.pi)
    y = radius * jnp.sin(ts * 2 * jnp.pi)
    for s in ss:
        if addConst:
            circle_s = jnp.stack([x, y, s * jnp.ones(nframes), constant_value * jnp.ones(nframes)], axis=-1)
        else:
            circle_s = jnp.stack([x, y, s * jnp.ones(nframes)], axis=-1)
        arrays_to_stack.append(circle_s)
    stacked_array = jnp.stack(arrays_to_stack, axis=0)
    return stacked_array

# --- Probability decay from a single target index ---
def index_based_probs(nframes, target_idx, sigma=5.0):
    # this function gives probabilities for all the frames, ordered in a circle, based in the distance between them and the sigma
    # the sigma is the exponential decay, for soft-binning
    if sigma == 0.0:
        probs = jnp.zeros(nframes).at[target_idx].set(1.0)
    else:
        indices = jnp.arange(nframes)
        dists = jnp.abs(indices - target_idx)
        dists = jnp.minimum(dists, nframes - dists)  # circular distance
        unnorm_probs = jnp.exp(-dists**2 / (2 * sigma**2))
        probs = unnorm_probs / jnp.sum(unnorm_probs)
    return probs

# --- Sample one index from a single distribution ---
def safe_sample_index(key, probs, exclude: set, fallback_idx: int):
    attempts = 0
    max_attempts = 20
    while attempts < max_attempts:
        key, subkey = random.split(key)
        idx = random.choice(subkey, probs.shape[0], p=probs)
        if idx not in exclude:
            return key, idx
        attempts += 1
    # Fallback if all samples are exhausted
    return key, fallback_idx

# --- Sample one index per target index, with uniqueness ---
def sample_unique_per_target(key, nframes, target_indices, sigma=5.0):
    sampled = []
    used = set()
    for t_idx in target_indices:
        probs = index_based_probs(nframes, t_idx, sigma)
        key, idx = safe_sample_index(key, probs, used, fallback_idx=t_idx)
        sampled.append(idx)
        used.add(idx)
    return key, jnp.array(sampled)

# ------------------- TD-DIP Net --------------------------------------- # 

    
class TimeDependant_DIP_Net:

    def __init__(self, nframes    : int,
                 key_latent,
                 addConst,
                 latent_generator,
                 imshape          : Tuple[int,int],
                 mapnet_layers    : Sequence[int],
                 cnn_latent_shape : Tuple[int,int] = (8,8),
                 features         : int = 128,
                 momentum         : float = 0.99,
                 levels           : int = 3
                 ):
        """
        - `latent_generator`: Callable(int, int) que devuelve un array de tamaño (nframes, N), donde N es algun número arbitrario de features.
        """
        self.nframes = nframes
        self.key_latent = key_latent
        self.imshape = imshape
        self.addConst = addConst

        self.latent  = latent_generator(nframes, self.key_latent, self.addConst)
        self.net = tDIP(mapnet_layers, cnn_latent_shape, features, momentum, levels)
    
    def init_params(self, key):
        params = self.net.init(key, self.latent[:1], training=False)
        return params
    
    def get_latent(self, t_index):
        if t_index is None:
            latent = self.latent 
        else:
            latent = self.latent[t_index,:]
        return latent

    def train_forward_pass(self, params, key, t_index):
        latent =  self.latent[t_index,:] #self.get_latent(t_index) 
        y, updates = self.net.apply(params, latent, training=True, rngs={'dropout':key}, mutable=['batch_stats'])
        y = to_complex(y)[...,0]
        nx, ny = self.imshape
        y = y[...,:nx,:ny]
        return y, updates['batch_stats'] 

from typing import Callable
from jax import numpy as np 
from jax import vmap 

def find_convex_coefficient(a,b,x):
  """
  Si a <= x < b, encuentra 0<=lmbda<1 tal que (1-lmbda) * a + lmbda * b = x,
  es decir   encuentra el coeficiente tal que x es una combinación convexa
  de a y b.

  ## Ejemplo

  ```python
  >>> find_convex_coefficient(1., 2., 1.)
  0.
  >>> find_convex_coefficient(1., 2., 1.2)
  0.19999999999999996
  ```
  """
  lmbda = (x - a)/(b-a)
  return lmbda

def get_dx(x):
  dx = x[1] - x[0]
  return dx

def identify_max_lessorequal_position(val, x):
  """
  Para un arreglo numpy de valores x ordenado de forma creciente, encuentra
  la posición asociada al máximo valor del arreglo x que es inferior o
  igual a `val`

  ## Ejemplo

  ```python
  >>> x = np.array([1.,2.,3.,4.])
  >>> identify_max_lessorequal_position(2., x)
  [False, True, False, False]
  >>> identify_max_lessorequal_position(3.4, x)
  [False, False, True, False]
  ```
  """
  dx = get_dx(x)
  liminf = (val - dx < x) * (x <= val)
  return liminf

def identify_min_greater_position(val, x):
  """
  Para un arreglo numpy de valores x ordenado de forma creciente, encuentra
  la posición asociada al mínimo valor del arreglo x que es estrictamente
  superior a `val`.

  ## Ejemplo

  ```python
  >>> x = np.array([1.,2.,3.,4.])
  >>> identify_min_greater_position(2., x)
  [False, False, True, False]
  >>> identify_min_greater_position(2.4, x)
  [False, False, True, False]
  ```
  """
  dx = get_dx(x)
  limsup = (val < x) * (x <= val + dx)
  return limsup

def find_array_convex_coefficients(val, x):
  liminf = identify_max_lessorequal_position(val, x)
  limsup = identify_min_greater_position(val, x)
  a = np.sum(x * liminf)
  b = np.sum(x * limsup)
  lmbda = find_convex_coefficient(a,b,val)
  return (1 - lmbda) * liminf +  lmbda * limsup

def random_ndim_helix_encoder(t,total_cycles, key, extra_dims, ts):
  # t = ts[3:5,None]
  nframes = ts.shape[0]
  extra_dims = 14
  total_cycles = 1
  lmbdas = vmap(find_array_convex_coefficients, in_axes=(0, None))(t[...,0], ts) # (selected_frames, nframes)
  noise = random.normal(key, (nframes, extra_dims)) # (nframes, extra_dims)
  interpolated_noise_at_t = np.sum(lmbdas[:,:,None] * noise[None, :,:], axis=1) # (selected_frames, extra_dims), where combined along nframes dim
  big_helix = jnp.concatenate([jnp.cos(t), jnp.sin(t), interpolated_noise_at_t * t/total_cycles], axis=-1) # (selected_frames, 2 + extra_dims)
  return big_helix

def helix_encoder(t, nframes, total_cycles):
    helix = jnp.concatenate([jnp.cos(t), jnp.sin(t), t/total_cycles], axis=-1)
    return helix

class INRTemporalBasis(nn.Module):
  encoding:Callable
  hidden_layers:Sequence[int]
  output:int

  @nn.compact
  def __call__(self, t):
    tx = self.encoding(t)
    for layer in self.hidden_layers:
      tx = nn.Dense(layer)(tx)
      tx = nn.relu(tx)
    tx = nn.Dense(self.output)(tx)
    return tx
  


  # ------   New version of TD-DIP  -------


class MS_TD_DIP_Net:
    def __init__(self, nframes    : int,
                 n_slices         : int,
                 key_latent,
                 addConst,
                 latent_generator,
                 radius,
                 z_min,
                 z_max,
                 imshape          : Tuple[int,int],
                 mapnet_layers    : Sequence[int],
                 cnn_latent_shape : Tuple[int,int] = (8,8),
                 features         : int = 128,
                 momentum         : float = 0.99,
                 levels           : int = 3
                 ):
        """
        - `latent_generator`: Callable(int, int) que devuelve un array de tamaño (nframes, N), donde N es algun número arbitrario de features.
        """
        self.nframes = nframes
        self.n_slices = n_slices
        self.imshape = imshape
        self.key_latent = key_latent
        self.addConst = addConst
        self.radius = radius
        self.z_min  = z_min 
        self.z_max  = z_max
        self.latent = latent_generator(nframes, n_slices, self.key_latent, self.addConst, radius=self.radius, z_min=self.z_min, z_max=self.z_max)
        self.net = tDIP(mapnet_layers, cnn_latent_shape, features, momentum, levels)

        def compiled_apply(params, latent, key):
            return self.net.apply(
                params, latent,
                training=True,
                rngs={'dropout': key},
                mutable=['batch_stats']
            )

        self._compiled_apply = jax.jit(compiled_apply)
    
    def init_params(self, key):
        params = self.net.init(key, self.latent[0, :1], training=False)
        return params

    
    def train_forward_pass(self, params, key, t_index, slice_index):
        latent =  self.latent[slice_index, t_index,:]
        y, updates = self._compiled_apply(params, latent, key)
        y = to_complex(y)[...,0]
        nx, ny = self.imshape
        y = y[...,:nx,:ny]
        return y, updates['batch_stats'] 
