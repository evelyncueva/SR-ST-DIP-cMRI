from typing import List, Optional, Tuple, Sequence
import jax
import jax.numpy as jnp
from flax import nnx
from inrmri.radon import _radon_points
from jax import jit, vmap
from jax.scipy.interpolate import RegularGridInterpolator as RGI
from jax.numpy.fft import fft,fftshift, ifftshift

class MapNetNNX(nnx.Module):
    """
    Mapping network implementado en Flax NNX.

    Construye una red feed-forward con varias capas ocultas `Linear` seguidas de
    funciones de activación ReLU, y una capa de salida lineal.

    Parameters
    ----------
    in_features : int
        Número de características de entrada (dimensión de la primera capa).
        Por ejemplo, 3 si se pasa un vector (x, y, z).
    hidden_sizes : Sequence[int]
        Lista con el número de neuronas en cada capa oculta.
        Ejemplo: (8, 8) crea dos capas ocultas de 8 unidades cada una.
    out_features : int
        Número de neuronas de la capa de salida. Por ejemplo, 64 si se requiere
        un vector latente de tamaño 64.
    rngs : nnx.Rngs
        Generador de números aleatorios de NNX para inicializar los parámetros.

    Example
    -------
    >>> rngs = nnx.Rngs(0)
    >>> model = MapNetNNX(in_features=3, hidden_sizes=(8, 8), out_features=64, rngs=rngs)
    >>> x = jnp.ones((1, 3))     # batch=1, input=(x,y,z)
    >>> y = model(x)             # salida de shape (1, 64)
    >>> y.shape
    (1, 64)
    """

    def __init__(self,
                 in_features: int,
                 hidden_sizes: Sequence[int],
                 out_features: int,
                 rngs: nnx.Rngs):
        # Construcción de las capas ocultas
        self.hidden: list[nnx.Linear] = []
        prev = in_features
        for i, h in enumerate(hidden_sizes):
            # Capa densa oculta de tamaño h
            self.hidden.append(nnx.Linear(prev, h, rngs=rngs))
            prev = h

        # Capa de salida que proyecta a out_features
        self.out = nnx.Linear(prev, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Ejecuta el forward pass de la red.

        Parameters
        ----------
        x : jax.Array
            Entrada de forma (batch, in_features) o (..., in_features).

        Returns
        -------
        jax.Array
            Salida de forma (..., out_features).
        """
        # Pasar por cada capa oculta + ReLU
        for layer in self.hidden:
            x = layer(x)
            x = jax.nn.relu(x)

        # Proyección final sin activación
        x = self.out(x)
        return x

# ---------- util ----------
def upsample_2d(x: jax.Array, scale: int = 2, method: str = "nearest") -> jax.Array:
    if scale == 1:
        return x
    # NHWC, nearest
    x = jnp.repeat(x, scale, axis=1)
    x = jnp.repeat(x, scale, axis=2)
    return x

# ---------- bloques ----------
class ConvBNReLU(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int,int]=(3,3),
        strides: Tuple[int,int]=(1,1),
        momentum: float = 0.99,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.conv = nnx.Conv(
            in_channels, out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding="SAME",
            rngs=rngs,
        )
        self.bn = nnx.BatchNorm(
            num_features=out_channels,
            momentum=momentum,
            epsilon=1e-5,
            axis=-1, 
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        x = self.conv(x)
        x = self.bn(x, use_running_average=not training)
        return jax.nn.relu(x)

class DecoderNNX(nnx.Module):
    """
    Decoder 2D NHWC sin lazy-build.
    Estructura: 2x(ConvBNReLU) -> Upsample -> [levels * (2x(ConvBNReLU) -> Upsample)] -> 2x(ConvBNReLU) -> Conv final
    """
    def __init__(
        self,
        in_channels: int,            # 1 si tu MapNet->reshape añade un solo canal
        features: int = 128,
        momentum: float = 0.99,
        levels: int = 3,
        out_features: int = 2,
        upsampling_method: str = "nearest",
        upsampling_factor: int = 2,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.features = features
        self.levels = levels
        self.out_features = out_features
        self.upsampling_method = upsampling_method
        self.upsampling_factor = upsampling_factor
        self.rngs = rngs if rngs is not None else nnx.Rngs(0)

        # Head
        self.head: List[ConvBNReLU] = [
            ConvBNReLU(in_channels, features, momentum=momentum, rngs=self.rngs),
            ConvBNReLU(features,    features, momentum=momentum, rngs=self.rngs),
        ]

        # Body
        self.body: List[List[ConvBNReLU]] = []
        for _ in range(levels):
            self.body.append([
                ConvBNReLU(features, features, momentum=momentum, rngs=self.rngs),
                ConvBNReLU(features, features, momentum=momentum, rngs=self.rngs),
            ])

        # Tail
        self.tail: List[ConvBNReLU] = [
            ConvBNReLU(features, features, momentum=momentum, rngs=self.rngs),
            ConvBNReLU(features, features, momentum=momentum, rngs=self.rngs),
        ]

        # Conv final
        self.final_conv = nnx.Conv(
            features, out_features,
            kernel_size=(3,3),
            strides=(1,1),
            padding="SAME",
            rngs=self.rngs,
        )

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        for blk in self.head:
            x = blk(x, training=training)

        x = upsample_2d(x, scale=self.upsampling_factor, method=self.upsampling_method)

        for level_blocks in self.body:
            for blk in level_blocks:
                x = blk(x, training=training)
            x = upsample_2d(x, scale=self.upsampling_factor, method=self.upsampling_method)

        for blk in self.tail:
            x = blk(x, training=training)

        return self.final_conv(x)

def to_complex(an_array): 
   real, imag = jnp.split(an_array, 2, axis=-1)
   return real + 1j * imag 
    
class TDIPNNX(nnx.Module):
    """
    tDIP en NNX: MapNet -> reshape (px, py) -> añadir canal -> Decoder.

    map_in_features: dimensión de t
    map_hidden_sizes: capas de MapNet
    cnn_latent_shape: (px, py) que produce MapNet (antes de añadir canal)
    decoder_features: # filtros internos del decoder
    momentum: BN momentum dentro del decoder
    levels: # de niveles [2xConvBNReLU -> upsample] del decoder
    out_features: canales de salida
    upsampling_method: 'nearest' o 'bilinear'
    upsampling_factor: factor de upsample por nivel
    rngs: RNGs compartidos
    """
    def __init__(
        self,
        map_in_features: int,
        map_hidden_sizes: Sequence[int],
        cnn_latent_shape: Tuple[int, int],
        decoder_features: int,
        momentum: float,
        levels: int,
        out_features: int = 2,
        upsampling_method: str = "nearest",
        upsampling_factor: int = 2,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.px, self.py = cnn_latent_shape
        self.rngs = rngs if rngs is not None else nnx.Rngs(0)

        # 1) MapNet: (N, map_in_features) -> (N, px*py)
        self.mapnet = MapNetNNX(
            in_features=map_in_features,
            hidden_sizes=map_hidden_sizes,
            out_features=self.px * self.py,
            rngs=self.rngs,
        )

        # 2) Decoder: entrada NHWC con C=1 (después del reshape)
        self.decoder = DecoderNNX(
            in_channels=1,                  
            features=decoder_features,
            momentum=momentum,
            levels=levels,
            out_features=out_features,
            upsampling_method=upsampling_method,
            upsampling_factor=upsampling_factor,
            rngs=self.rngs,
        )

    def __call__(self, t: jax.Array, training: bool) -> jax.Array:
        """
        t: (N, map_in_features)  ->  y: (N, H_out, W_out, out_features)
        """
        # MapNet
        x = self.mapnet(t)                               # (N, px*py)

        # Reshape + canal
        x = jnp.reshape(x, (x.shape[0], self.px, self.py, 1))  # (N, px, py, 1)

        # Decoder
        y = self.decoder(x, training=training)           # (N, H_out, W_out, out_features)
        return y
    
def radon_integration(img, ds, axis=-1):
    return jnp.sum(img, axis=axis) * ds

def fft1d_shifted(x):
    return jnp.fft.fftshift(jnp.fft.fft(jnp.fft.ifftshift(x, axes=-1), axis=-1), axes=-1)


def interpolate_points_to_grid(points, csmap): 
  """
  - points: narray [...,2]
  - csmap: array 2d (px,py)
  """
  Nx, Ny = csmap.shape[:2]
  x, y = jnp.linspace(-1,1,Nx, endpoint = False), jnp.linspace(-1,1,Ny, endpoint = False)
  interpolator = RGI((x,y), csmap, bounds_error = False, fill_value = 0.)
  return interpolator(points)


def radon_points(alpha, N):
    """
        grilla rotada de valores en [0,1]^2
    - alpha () angulo de rotacion
    - N int 
    """
    ss, radonxy, perpendicularxy = _radon_points(alpha, 1., N)
    radonpoints = (radonxy[:,None,:] + perpendicularxy[None,:,:])
    return radonpoints, ss

def rotate(im, alpha):
    """
    - im: shape (frame, px, py). px=py

    rota en las dimensiones de px, py
    
    """
    N = im.shape[-1]
    radonpoints, _ = radon_points(alpha, N)
    vmapped = interpolate_points_to_grid 
    for _ in range(im.ndim-2):
        vmapped = vmap(vmapped, in_axes=(None,0))
    interp_im = vmapped(radonpoints, im)
    return interp_im 

@jit
def fastshiftfourier(A):
  print(A.shape[-1])
  return fftshift(fft(ifftshift(A), norm = 'forward'))

class ForwardRadonOperatorJAX:
    """
    csmap: (ncoils, N, N) complejo o real (si real, se convierte a complejo)
    """
    def __init__(self, csmap: jnp.ndarray, spclim: float = 0.5, ds: float = 1.0):
        csmap = jnp.asarray(csmap)
        assert csmap.ndim == 3 and csmap.shape[1] == csmap.shape[2], "csmap debe ser (ncoils,N,N)"
        self.csmap = csmap.astype(jnp.complex64) if csmap.dtype.kind != 'c' else csmap
        self.N = csmap.shape[1]
        self.ds = ds
        self.spclim = spclim

    def rotated_csweighted_ims(self, im, alphas):
        """
        Aplica ponderación por los mapas de sensibilidad y rota cada imagen
        según un ángulo diferente.

        Parameters
        ----------
        im : np.ndarray
            Lote de imágenes a procesar. Shape: (batch, px, py).
        alphas : np.ndarray
            Ángulos de rotación en radianes. Shape: (batch,).

        Returns
        -------
        rotated_im : np.ndarray
            Imágenes ponderadas y rotadas. Shape: (batch, ncoils, px, py).
        """
        # Reordenar: batch → última dimensión
        im = jnp.moveaxis(im, 0, -1)                     # (px, py, batch)

        # Ponderar por mapas de sensibilidad de coils
        im = im[None, :, :, :] * self.csmap[:, :, :, None]  # (ncoils, px, py, batch)

        # Reordenar: batch → eje 1
        im = jnp.moveaxis(im, -1, 1)                     # (ncoils, batch, px, py)

        # Rotar cada imagen con su ángulo correspondiente
        rotated_im = vmap(rotate, in_axes=(1, 0))(im, alphas)  # (batch, ncoils, px, py)

        return rotated_im

    def radon_transform(self, rotated_im):
        """
        Calcula la transformada de Radon para un lote de imágenes rotadas.

        Parameters
        ----------
        rotated_im : np.ndarray
            Imágenes ya ponderadas y rotadas. Shape: (batch, ncoils, px, py).

        Returns
        -------
        radon_kspace : np.ndarray
            Datos de k-space tras la integración de Radon. Shape: (batch, ncoils, px).
        """
        # Integración a lo largo de un eje espacial
        im = radon_integration(rotated_im, self.ds, axis=-1)   # (batch, ncoils, px)

        # Transformada rápida de Fourier con centrado
        radon_kspace = fastshiftfourier(im * self.spclim * 2)

        return radon_kspace
    
# ================== Targets en k-space (robusto a formatos) ==================
def as_complex_kspace(Y: jnp.ndarray) -> jnp.ndarray:
    """
    Convierte Y a complejo shape (B, C, N)
    Admite:
      - (B,C,N) complejo
      - (B,C,N,1) complejo  -> squeeze
      - (B,C,N,2) real      -> Re/Im
    """
    Y = jnp.asarray(Y)
    if Y.ndim == 3 and Y.dtype.kind == 'c':
        return Y
    if Y.ndim == 4 and Y.shape[-1] == 1 and Y.dtype.kind == 'c':
        return jnp.squeeze(Y, axis=-1)
    if Y.ndim == 4 and Y.shape[-1] == 2 and Y.dtype.kind != 'c':
        return Y[..., 0] + 1j * Y[..., 1]
    raise ValueError(f"Formato de Y no soportado: shape={Y.shape}, dtype={Y.dtype}")

# ================== Pérdida compleja ponderada ==================
def complex_mse_weighted(pred: jnp.ndarray, targ: jnp.ndarray, w_11N: jnp.ndarray) -> jnp.ndarray:
    """
    pred, targ: (B, C, N) complejo
    w_11N: (1,1,N)
    """
    dre = (pred.real - targ.real)
    dim = (pred.imag - targ.imag)
    se = dre*dre + dim*dim
    return jnp.mean(se * w_11N)

def weighted_loss(X,Y,W): 
  return jnp.mean((jnp.abs(X - Y) * W)**2)

