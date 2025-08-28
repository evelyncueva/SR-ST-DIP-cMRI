# ---------------------------------------------------------------------------- #
# Experimental 
# TD-DIP with extra static noise input to Decoder
# ---------------------------------------------------------------------------- #
import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple, Sequence
from jax import random, jit
from inrmri.dip.tddip import Decoder, MapNet
from inrmri.dip.utils import to_complex

def tile_along_batch_dims(batch_shape, static_noise): 
    """
    - batch_shape: Tuple[int,...,int]
    - static_noise: Array, ndim = 2, (típico (8,8)) 

    - repite el array `static_noise` para cada elemento del batch y
    añade una dimensión extra al final (features)

    ## Ejemplos 
    
    ```
    >>> static_noise = random.uniform(key, (8,8))
    >>> static_noise.shape 
    (8,8)
    >>> tile_along_batch_dims((3,5), static_noise).shape 
    (3,5,8,8,1)
    >>> tile_along_batch_dims((1,), static_noise).shape 
    (1,8,8,1)
    >>> np.all(tile_along_batch_dims((1,), static_noise)[0,:,:,0] == static_noise)
    True 
    ```    
    """
    batch_ndim = len(batch_shape)
    noise_shape = static_noise.shape
    static_noise = jnp.reshape(static_noise, (1,) * batch_ndim + noise_shape + (1,))
    static_noise = jit(jnp.broadcast_to, static_argnums=1)(static_noise, batch_shape + noise_shape + (1,)) # repeat 
    # broadcasted_x = np.broadcast_to(fulldeformer.grid[:,:,None,:], batch_shape + (2,))
    return static_noise

class StaticTDDIP(nn.Module):
    mapnet_layers : Sequence[int]
    cnn_latent_shape : Tuple[int,int]
    features      : int 
    momentum      : float
    levels        : int

    @nn.compact
    def __call__(self, t, static_noise, training:bool):
        mapnet = MapNet(self.mapnet_layers, self.cnn_latent_shape)
        assert static_noise.shape == self.cnn_latent_shape
        cnn_noise = mapnet(t)
        batch_shape = cnn_noise.shape[:-1]
        cnn_noise = jnp.reshape(cnn_noise, batch_shape + self.cnn_latent_shape + (1,))
        static_noise = tile_along_batch_dims(batch_shape, static_noise)
        assert static_noise.shape == cnn_noise.shape 
        input_noise = jnp.concatenate((cnn_noise, static_noise), axis=-1)
        x = Decoder(self.features, self.momentum, self.levels)(input_noise, training)
        return x 


class TimeDependantAndStaticDIPNet:

    def __init__(self, nframes    : int,
                 total_cycles     : int,
                 latent_generator,
                 key,
                 imshape          : Tuple[int,int],
                 mapnet_layers    : Sequence[int],
                 noise_generator  = random.uniform,
                 cnn_latent_shape : Tuple[int,int] = (8,8),
                 features         : int = 128,
                 momentum         : float = 0.99,
                 levels           : int = 3
                 ):
        """
        - `latent_generator`: Callable(int, int) que devuelve un array de tamaño (nframes, N),
            donde N es algun número arbitrario de features.
        - `noise_generator`: Callable(key, shape) devuelve un array de ruido de tamaño shape
            por defecto es ruido uniforme 
        """
        self.nframes = nframes
        self.total_cycles = total_cycles
        self.imshape = imshape

        self.latent = latent_generator(nframes, total_cycles)
        self.static_noise = noise_generator(key, cnn_latent_shape)
        self.net = StaticTDDIP(mapnet_layers, cnn_latent_shape, features, momentum, levels)
    
    def init_params(self, key):
        params = self.net.init(key, self.latent[:1], self.static_noise, training=False)
        return params
    
    def get_latent(self, t_index):
        if t_index is None:
            latent = self.latent 
        else:
            latent = self.latent[t_index,:]
        return latent

    def train_forward_pass(self, params, key, t_index):
        latent =  self.latent[t_index,:] #self.get_latent(t_index) 
        y, updates = self.net.apply(params, latent, self.static_noise, training=True, rngs={'dropout':key}, mutable=['batch_stats'])
        y = to_complex(y)[...,0]
        nx, ny = self.imshape
        y = y[...,:nx,:ny]
        return y, updates['batch_stats'] 
