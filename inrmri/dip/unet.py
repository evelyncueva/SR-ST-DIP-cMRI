# ---------------------------------------------------------------------------- #
# dip.unet
# ---------------------------------------------------------------------------- #

import flax.linen as nn
import jax.numpy as jnp
from inrmri.dip.utils import upsampling_1d, upsampling_2d

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