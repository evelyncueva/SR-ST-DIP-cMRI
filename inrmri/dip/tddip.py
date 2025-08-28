import copy
import flax.linen as nn
import jax.numpy as jnp
from jax.lax import map as laxmap 
from typing import Tuple, Sequence
from inrmri.dip.utils import upsampling_1d, upsampling_2d, to_complex
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
    print(batch_shape, convolved_shape, features_shape)
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
    out_features  : int = 2 

    @nn.compact
    def __call__(self, t, training:bool):
        mapnet = MapNet(self.mapnet_layers, self.cnn_latent_shape)
        x = mapnet(t)
        x = jnp.reshape(x, x.shape[:-1] + self.cnn_latent_shape)
        x = x[...,None] # add features dimension
        x = Decoder(self.features, self.momentum, self.levels, out_features=self.out_features)(x, training)
        return x 

# ------------------- Generators --------------------------------------- # 

def helix_generator(nframes, total_cycles):
    ts = jnp.linspace(0,total_cycles, nframes, endpoint=False)
    helix = jnp.stack([jnp.cos(2 * jnp.pi * ts), jnp.sin(2 * jnp.pi *ts), ts/total_cycles], axis=-1)
    return helix

# ------------------- TD-DIP Net --------------------------------------- # 

class TimeDependant_DIP_Net:
    """
    Implementación de la red Time-Dependent Deep Image Prior (tDIP).

    Este modelo genera imágenes dinámicas a partir de una representación latente
    que evoluciona con el tiempo. El tiempo se mapea a un espacio latente usando
    un generador (por ejemplo, una hélice sinusoidal), y luego se decodifica a 
    imágenes complejas mediante la red tDIP (MapNet + Decoder).

    Parameters
    ----------
    nframes : int
        Número total de frames temporales.
    total_cycles : int
        Número de ciclos en la parametrización temporal (ej. frecuencia de la hélice).
    latent_generator : Callable[[int, int], np.ndarray]
        Función que genera la representación latente. Debe devolver un array de
        forma (nframes, N), donde N es el número de features latentes.
    imshape : Tuple[int, int]
        Forma espacial de las imágenes de salida (nx, ny).
    mapnet_layers : Sequence[int]
        Número de neuronas en cada capa oculta de la red MapNet.
    cnn_latent_shape : Tuple[int, int], default=(8,8)
        Forma (alto, ancho) de la grilla latente que se pasa al Decoder.
    features : int, default=128
        Número de mapas de características en el Decoder.
    momentum : float, default=0.99
        Parámetro de momentum para normalización.
    levels : int, default=3
        Número de niveles jerárquicos en el Decoder.
    out_images : int, default=1
        Número de imágenes de salida. La red produce `out_images * 2` canales
        para representar valores complejos (parte real e imaginaria).
    """

    def __init__(self, nframes: int,
                 total_cycles: int,
                 latent_generator,
                 imshape: Tuple[int, int],
                 mapnet_layers: Sequence[int],
                 cnn_latent_shape: Tuple[int, int] = (8, 8),
                 features: int = 128,
                 momentum: float = 0.99,
                 levels: int = 3,
                 out_images: int = 1):

        self.nframes = nframes
        self.total_cycles = total_cycles
        self.imshape = imshape

        # Representación latente generada por el generador elegido (ej. hélice)
        self.latent = latent_generator(nframes, total_cycles)

        # Salidas complejas: real e imaginaria
        self.out_images = out_images

        # Definición de la red base tDIP (MapNet + Decoder)
        self.net = tDIP(
            mapnet_layers,
            cnn_latent_shape,
            features,
            momentum,
            levels,
            out_features=out_images * 2
        )

    # -------------------------------------------------------------------

    def init_params(self, key):
        """
        Inicializa los parámetros de la red.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Semilla aleatoria de JAX.

        Returns
        -------
        params : PyTree
            Parámetros inicializados de la red.
        """
        params = self.net.init(key, self.latent[:1], training=False)
        return params

    # -------------------------------------------------------------------

    def get_latent(self, t_index):
        """
        Obtiene la representación latente asociada a un frame.

        Parameters
        ----------
        t_index : int or None
            Índice temporal. Si es None, devuelve la latente completa.

        Returns
        -------
        latent : np.ndarray
            Representación latente de forma (N,) o (nframes, N).
        """
        if t_index is None:
            return self.latent
        return self.latent[t_index, :]

    # -------------------------------------------------------------------

    def train_forward_pass(self, params, key, t_index):
        """
        Realiza un paso forward en modo entrenamiento.

        Parameters
        ----------
        params : PyTree
            Parámetros de la red.
        key : jax.random.PRNGKey
            Semilla aleatoria para operaciones estocásticas (ej. dropout).
        t_index : int
            Índice temporal del frame.

        Returns
        -------
        y : jnp.ndarray
            Imagen compleja reconstruida en el frame indicado.
            Forma: (nx, ny, out_images).
        batch_stats : dict
            Estadísticas de normalización actualizadas.
        """
        latent = self.latent[t_index, :]
        y, updates = self.net.apply(
            params,
            latent,
            training=True,
            rngs={'dropout': key},
            mutable=['batch_stats']
        )
        y = to_complex(y)

        nx, ny = self.imshape
        y = y[..., :nx, :ny, :]
        return y, updates['batch_stats']

    # -------------------------------------------------------------------

    def lax_cine(self, params, key, t_idx, batch: int):
        """
        Reconstruye una secuencia cine (dinámica) evaluando el modelo en lotes temporales.

        Los índices de tiempo se agrupan en lotes (`batch`). Si el número de tiempos
        no es divisible entre `batch`, se completan con repeticiones del último frame
        y luego se recortan al final.

        Parameters
        ----------
        params : PyTree
            Parámetros de la red.
        key : jax.random.PRNGKey
            Semilla aleatoria para operaciones estocásticas.
        t_idx : np.ndarray
            Índices temporales a evaluar. Forma: (Nt,).
        batch : int
            Número de tiempos a procesar por lote.

        Returns
        -------
        predim : jnp.ndarray
            Reconstrucción dinámica compleja de forma (nx, ny, Nt).
        """
        Nt = t_idx.shape[0]
        remainder = Nt % batch
        pad = (batch - remainder) if remainder > 0 else 0

        # Si sobra, repetimos el último índice para completar el batch
        if pad > 0:
            pad_indices = jnp.full((pad,), t_idx[-1])
            padded_t_idx = jnp.concatenate([t_idx, pad_indices], axis=0)
        else:
            padded_t_idx = t_idx

        stacked_tidxs = padded_t_idx.reshape(-1, batch)

        # Evaluación en lotes con laxmap
        predim = laxmap(lambda t: self.train_forward_pass(params, key, t)[0], stacked_tidxs)

        nx, ny = self.imshape
        predim = jnp.reshape(predim, (-1, nx, ny))  # (Nt + pad, nx, ny)
        predim = predim[:Nt]                        # quitar el padding extra
        predim = jnp.moveaxis(predim, 0, -1)        # (nx, ny, Nt)
        return predim


def ts_from_ecg(ecg):
    """
    Generates a continuous pseudo-time axis for a sequence of radial MRI spokes 
    grouped by cardiac cycles, using ECG trigger indices.

    Parameters
    ----------
    ecg : array-like (1D)
        Indices indicating the start of each cardiac cycle (typically output
        from `get_cycle_start_indices`). Must contain at least two entries to
        define one cycle.

    Returns
    -------
    ts : numpy.ndarray
        1D array of pseudo-time values, linearly scaled within each cardiac
        cycle. Each cycle spans a time interval of 1.0 units, so the result
        increases smoothly across cycles (e.g.,[0.0, ..., 0.95, 1.0, ..., 1.95,
        2.0, ...]).

    Notes
    -----
    The function assigns each spoke a fractional time value according to its relative
    position within its cardiac cycle.
    """
    times = []
    total_cycles = len(ecg) - 1

    for i in range(total_cycles):
        start = ecg[i]
        end = ecg[i + 1]
        n_spokes = end - start

        # Create a local time axis from i to i+1 (excluding endpoint)
        ts = i + jnp.linspace(0, 1, n_spokes, endpoint=False)
        times.append(ts)

    return jnp.concatenate(times, axis=0)

def helix_from_ecg_generator(ecg):
    """
    Generates a 3D helical trajectory from ECG-like trigger indices to model
    periodic time. Each point corresponds to a spoke and lies on a helix winding
    around the z-axis.

    Parameters
    ----------
    ecg : array-like (1D)
        Indices marking the start of each cardiac cycle (e.g., output of
        `data.radial_cine.get_cycle_start_indices`).

    Returns
    -------
    make_helix : Callable(n,t)-> numpy.ndarray
        Array of shape (N, 3), where N is the total number of spokes. Each row
        contains (x, y, z) coordinates of a point on the helix:
            - x = cos(2πt)
            - y = sin(2πt)
            - z = normalized time (t / total_cycles)

    Notes
    -----
    This is useful for pseudo-periodic behavior, such as cardiac motion, where
    similar phases should lie close to each other in the helix.
    """
    def make_helix(nframes, total_cycles):
        ts = ts_from_ecg(ecg)
        total_cycles = len(ecg) - 1

        helix = jnp.stack([
            jnp.cos(2 * jnp.pi * ts),
            jnp.sin(2 * jnp.pi * ts),
            ts / total_cycles
        ], axis=-1)
        return helix

    return make_helix

def tddip_net_circle_loader(nframes, **kwargs):
    """
    - `kwargs`: argumentos de `dip.TimeDependant_DIP_Net`
    ## Ejemplo 

    CONFIG_NET = {
    'mapnet_layers':[64,64],
    'cnn_latent_shape':(8,8),
    'levels':4,
    'features':64, 
    'imshape':IMSHAPE
    }
    nframes = 25 
    cine_net = tddip_net_circle_loader(nframes, **CONFIG_NET)
    """
    pass 

    def circle_generator(nframes, total_cycles):
        ts = jnp.linspace(0,1,nframes, endpoint=False)
        circ = jnp.stack([jnp.cos(2 * jnp.pi * ts), jnp.sin(2 * jnp.pi *ts)], axis=-1)
        return circ 

    cine_net = TimeDependant_DIP_Net(
        nframes=nframes, 
        total_cycles=1,
        latent_generator=circle_generator,
        **kwargs
        )
    
    return cine_net 