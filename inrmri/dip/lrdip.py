import jax
import jax.numpy as jnp
from typing import Tuple
from jax import random
from inrmri.dip.unet import UNet
from inrmri.dip.utils import split_last_dim, to_complex

def join_updates(updates_net, updates_basis): 
    return {'batch_stats': updates_net['batch_stats'], 'basis': updates_basis}

def split_updates(params_and_basis):
    basis = params_and_basis['basis']
    params = {**params_and_basis}
    params.pop('basis')
    return params, basis

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
        self.latent_z_time  = noise_scaling * random.normal(key_time, (1, nframes, input_features)) # (1, nframes, 32)

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
        latent = self.latent_z_time 
        y_time, updates = self.time_unet.apply(params_time, latent, training=True, rngs={'dropout':key}, mutable=['batch_stats'])
        if t_index_batch is not None: 
            y_time = y_time[...,t_index_batch, :]
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
        return y_space, y_time, {'batch_stats':batch_stats}
    
    def train_forward_pass(self, params, key, t_index=None):
        y_space, y_time, batch_stats = self.train_basis_and_update(params, key, t_index)
        im = self.combine_space_time(y_space, y_time)
        return im, batch_stats  
    
    def train_forward_with_exponential_reg(self, params_and_basis, key, t_idx, alpha): 
        """
        alpha: en (0,1), 1 no regulariza nada, mientras mas cercano a 0, más usa el anterior 
        """
        params, old_basis = split_updates(params_and_basis)
        y_space_old, y_time_old = old_basis
        y_space, y_time, batch_stats = self.train_basis_and_update(params, key, None) # produce todas las bases 
        y_space_new = alpha * y_space + (1-alpha) * y_space_old
        y_time_new = alpha * y_time + (1-alpha) * y_time_old
        new_basis = (y_space_new, y_time_new)
        im = self.combine_space_time(y_space_new, y_time_new)
        return im[t_idx], join_updates(batch_stats, new_basis) 

    def forward_pass_with_updating(self, params, key, t_index=None):

        im, batch_stats = self.train_forward_pass(params, key, t_index)
        params['batch_stats'] = batch_stats
        return im 

    def inference_basis(self, fullparam, t_index_batch):
        params_space, params_time = self.split_params(fullparam)
        latent_t = self.latent_z_time 
        y_space = self.space_unet.apply(params_space, self.latent_z_space, training=False)
        y_time = self.time_unet.apply(params_time, latent_t, training=False)
        if t_index_batch is not None: 
            y_time = y_time[...,t_index_batch, :]
        return y_space, y_time
        
    def inference_image(self, fullparam, t_index_batch):
        y_space, y_time = self.inference_basis(fullparam, t_index_batch)
        im = self.combine_space_time(y_space, y_time)
        return im 