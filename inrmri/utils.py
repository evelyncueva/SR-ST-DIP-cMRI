# utils 
import jax.numpy as np
from functools import partial
from jax import jit, random 
from jax import vmap
import gc
import jax
from jax._src import dispatch

def normalize(img): 
    return img/np.max(np.abs(img))
    
@jit
def split(xy): 
  return np.split(xy, 2, axis = -1)

@jit
def is_inside_of_lims(x, lims): 
  return (lims[0] <= x) * (x <= lims[1])

# np.vectorize(grad(squeezednet, 1), signature='(2),(n,2)->(2)', excluded = [0])
@partial(np.vectorize, signature='(2),()->()')
def is_inside_of_radial_lim(x, lim): 
  return np.power(x[0], 2) + np.power(x[1], 2) <= np.power(lim, 2)

def squeeze(X):
  return np.squeeze(X, axis = -1)

def to_complex(array): 
  real, imag = split(array) 
  return real  + 1j * imag 

# def to_complex(array): 
#   mod, phase = split(array) 
#   return mod * np.exp(1j * phase)

def meshgrid_from_subdiv(shape, lims = (0,1), endpoint = False):
  """
  Crea una grilla stackeando los resultados de un meshgrid en el último eje. 

  Para cada valor `N` de la tupla, crea un linspace con esa cantidad de puntos, 
  usando siempre los mismos límites y endpoint. Ver docs de linspace:
  https://numpy.org/doc/stable/reference/generated/numpy.linspace.html

  # Argumentos 
  - `shape`: n-tupla, numero de puntos en cada dimension.
  - `lims`: 2-tupla, limite inferior y superior, el mismo en cada dimension. 
  - `enpoint`: bool, mismo en cada dimension, ver docs de linspace
    
  # Resultado
  - `grid`: una grilla de n+1 dimensiones, con `grid.shape[:-1] = shape` y
    `grid.shape[-1] = n`.

  # Ejemplo
  ```
  >>> meshgrid_from_subdiv((3,2), lims = (-1,1))
  DeviceArray([[[-1.        , -1.        ],
                [-1.        ,  0.        ]],

              [[-0.33333328, -1.        ],
                [-0.33333328,  0.        ]],

              [[ 0.33333337, -1.        ],
                [ 0.33333337,  0.        ]]], dtype=float32)
  ```

  Un meshgrid de `n` arreglos entrega una n-tupla de grillas, todas del mismo
  tamaño, dado por el tamaño de los arreglos. 
  ```
  >>> grid = meshgrid_from_subdiv((4,2,3,6,1))
  >>> grid.shape 
  (4, 2, 3, 6, 1, 5)
  ```
  """
  liml, limr = lims
  coords = []
  for N in shape:
    coords.append(np.linspace(liml, limr, N, endpoint=endpoint))
  grid = np.stack(np.meshgrid(*coords, indexing='ij'), axis = -1)
  return grid 


def meshgrid_from_subdiv_autolims(shape, lims = None, endpoint = False):
  """
  Crea una grilla stackeando los resultados de un meshgrid en el último eje. 

  Para cada valor `N` de la tupla, crea un linspace con esa cantidad de puntos, 
  usando siempre los mismos límites y endpoint. Ver docs de linspace:
  https://numpy.org/doc/stable/reference/generated/numpy.linspace.html

  # Argumentos 
  - `shape`: n-tupla, numero de puntos en cada dimension.
  - `lims`: 2-tupla, limite inferior y superior.
    Por defecto None. Asocia el intervalo [-1,1] a la dimension mas grande, y 
    da intervalos centrados en cero proporcionalmente mas pequeños a las demás
    dimensiones. Por ejemplo, para shape (5,3,10), los intervalos son
    ([(-0.5, 0.5), (-0.3, 0.3), (-1.0, 1.0)]).
  - `enpoint`: bool, mismo en cada dimension, ver docs de linspace
    
  # Resultado
  - `grid`: una grilla de n+1 dimensiones, con `grid.shape[:-1] = shape` y
    `grid.shape[-1] = n`.

  Un meshgrid de `n` arreglos entrega una n-tupla de grillas, todas del mismo
  tamaño, dado por el tamaño de los arreglos. 
  ```
  >>> grid = meshgrid_from_subdiv((4,2,3,6,1))
  >>> grid.shape 
  (4, 2, 3, 6, 1, 5)
  ```
  """
  if lims is not None: 
    lims_dim = [lims for _ in shape] 
  else: 
    maxdim = max(shape)
    lims_dim = [(-N/maxdim, N/maxdim) for N in shape]
  coords = []
  for N, (liml, limr) in zip(shape, lims_dim):
    coords.append(np.linspace(liml, limr, N, endpoint=endpoint))
  grid = np.stack(np.meshgrid(*coords, indexing='ij'), axis = -1)
  return grid 


def flatten_all_but_lastdim(array): 
  """
  Colapsa todas las dimensiones en 1, salvo la última. 
  # Ejemplo 
  ```
  >>> a = np.ones((3,5,2,6)) # a.shape = (3,5,2,6)
  >>> flatten_all_but_lastdim(a).shape
  (30, 6)
  ```  
  Notar que `30 = 3 * 5 * 2`.
  """
  return np.reshape(array, (-1, array.shape[-1]))


def randon_in_domain_points(npoints, key, lims):
  """
  Entrega npoints vectores aleatorios entre lims
  - lims: matriz de (dim,2), con dim la dimension de salida de los vectores. Cada 
  vector entregado x satisface que `lims[i,0] <= x[i] <= lims[i,1]`
  # Ejemplo
  ```
  >>> lims = np.array([[-1.,1.], [0.,1.], [-5.,-1.]]
  >>> xs = randon_in_domain_points(5, key, lims)
  >>> assert arevectorinrange(xs, lims) # debería ser True 
  ```
  """
  lb, ub = split(lims)
  U = random.uniform(key, shape=(npoints,lims.shape[0]))
  return squeeze(U[:,:,None] * (ub - lb) + lb)

def arevectorinrange(vectors, lims):
  areinrange = True
  lb, ub = split(lims)
  for u in vectors: 
    isuinrange = np.logical_and(np.all(u[:,None] <= ub), np.all(lb <= u[:,None]))
    areinrange = areinrange and isuinrange
  return areinrange 

def log_interval(minval, maxval, steps): 
  a = minval 
  b = np.log(maxval/a) / steps 
  x = np.linspace(0, steps, steps+1)
  return a * np.exp(b * x)

def calculate_grid_spacing(grid):
  """
  - grid.shape (Nx, Ny, 2) (creada con meshgrid e indexing='ij')
  """
  dx = grid[1,0,0] - grid[0,0,0]
  dy = grid[0,1,1] - grid[0,0,1]
  return dx, dy 

#### Nuevas funciones de Rafael


# ----------------------------------------
# Create a binary center mask (JAX-native)
# ----------------------------------------
def create_center_mask(img_shape, region_size=0.5):
    """
    Create a binary mask with 1s in the center of the image.
    
    Args:
        img_shape: (height, width) of the image
        region_size: proportion of the image to cover (0.5 means central half)

    Returns:
        mask: array of shape (height, width) with 1s in center region
    """
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.float32)
    x_start, x_end = int(w * (1 - region_size) / 2), int(w * (1 + region_size) / 2)
    y_start, y_end = int(h * (1 - region_size) / 2), int(h * (1 + region_size) / 2)
    mask = mask.at[y_start:y_end, x_start:x_end].set(1.0)
    return mask

# ----------------------------------------
# Total Variation (TV) Loss
# ----------------------------------------
def tv_loss(img, center_mask, epsilon=1e-8):
    grad_x = np.roll(img, -1, axis=1) - img
    grad_y = np.roll(img, -1, axis=0) - img
    grad_x *= center_mask
    grad_y *= center_mask
    return np.sum(np.sqrt(grad_x**2 + grad_y**2 + epsilon))


# ----------------------------------------
# Tikhonov Regularization Loss
# ----------------------------------------
def tikhonov_loss(img, center_mask, epsilon=1e-8):
    lap_x = np.roll(img, -1, axis=1) - 2 * img + np.roll(img, 1, axis=1)
    lap_y = np.roll(img, -1, axis=0) - 2 * img + np.roll(img, 1, axis=0)
    lap_x *= center_mask
    lap_y *= center_mask
    return np.sum(lap_x**2 + lap_y**2 + epsilon)


# ----------------------------------------
# L1 Smoothing Loss
# ----------------------------------------
def l1_loss(img, center_mask, epsilon=1e-8):
    diff_x = np.abs(np.roll(img, -1, axis=1) - img)
    diff_y = np.abs(np.roll(img, -1, axis=0) - img)
    diff_x *= center_mask
    diff_y *= center_mask
    return np.sum(diff_x + diff_y + epsilon)



# ----------------------------------------
# Batched Denoise Loss Function
# ----------------------------------------
def denoise_loss_batch(imgs, loss_fn, center_mask):
    """
    Compute denoising loss over a batch of 2D images.

    Args:
        imgs: JAX array of shape (H, W, N) — stack of N images
        loss_fn: callable — one of tv_loss, tikhonov_loss, or l1_loss
        center_mask: array of shape (H, W) — central mask

    Returns:
        mean denoising loss over the batch
    """
    denoise_fn = lambda img: loss_fn(np.abs(img), center_mask)
    batch_loss = vmap(denoise_fn, in_axes=2)(imgs)
    return np.mean(batch_loss)


def clear_jax_memory():
    gc.collect()
    jax.clear_caches()
    dispatch.xla_primitive_callable.cache_clear()