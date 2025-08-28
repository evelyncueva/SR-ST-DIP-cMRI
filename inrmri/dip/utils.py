import jax
from typing import Tuple
import jax.numpy as jnp 

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

def split_last_dim(an_array):
   old_shape = an_array.shape
   new_shape = old_shape[:-1] + (old_shape[-1]//2, 2)
   return an_array.reshape(new_shape)

def to_complex(an_array): 
   real, imag = jnp.split(an_array, 2, axis=-1)
   return real + 1j * imag 
