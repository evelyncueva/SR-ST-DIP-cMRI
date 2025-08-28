import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence, Callable
from jax import vmap, random 

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
  >>> x = jnp.array([1.,2.,3.,4.])
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
  >>> x = jnp.array([1.,2.,3.,4.])
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
  a = jnp.sum(x * liminf)
  b = jnp.sum(x * limsup)
  lmbda = find_convex_coefficient(a,b,val)
  return (1 - lmbda) * liminf +  lmbda * limsup

def random_ndim_helix_encoder(t,total_cycles, key, extra_dims, ts):
  # t = ts[3:5,None]
  nframes = ts.shape[0]
  extra_dims = 14
  total_cycles = 1
  lmbdas = vmap(find_array_convex_coefficients, in_axes=(0, None))(t[...,0], ts) # (selected_frames, nframes)
  noise = random.normal(key, (nframes, extra_dims)) # (nframes, extra_dims)
  interpolated_noise_at_t = jnp.sum(lmbdas[:,:,None] * noise[None, :,:], axis=1) # (selected_frames, extra_dims), where combined along nframes dim
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