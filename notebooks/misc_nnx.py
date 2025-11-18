from typing import List, Optional, Tuple, Sequence
import jax
import jax.numpy as jnp
from flax import nnx
from inrmri.radon import _radon_points
from jax import jit, vmap
from jax.scipy.interpolate import RegularGridInterpolator as RGI
from jax.numpy.fft import fft,fftshift, ifftshift, fftfreq
from jax import lax


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
        t: (B, map_in_features)  ->  y: (B, H_out, W_out, out_features)
        B: batch size
        """
        # MapNet
        x = self.mapnet(t)                               # (B, px*py)

        # Reshape + canal
        x = jnp.reshape(x, (x.shape[0], self.px, self.py, 1))  # (B, px, py, 1)

        # Decoder
        y = self.decoder(x, training=training)           # (B, H_out, W_out, out_features)
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

class ForwardRadonOperator:
    """
    Operador de Radon hacia adelante con ponderación de sensibilidad de coils.

    Este operador toma imágenes reconstruidas, aplica la ponderación por los
    mapas de sensibilidad de los coils (csmap), las rota según ángulos dados,
    y luego aplica la integración de Radon para obtener el k-space.

    Parameters
    ----------
    csmap : np.ndarray
        Mapas de sensibilidad de los coils. Shape: (ncoils, px, px).
    spclim : float, default=0.5
        Factor de escalamiento en k-space.
    """

    def __init__(self, csmap, spclim: float = 0.5):
        assert csmap.ndim == 3, "csmap debe tener 3 dimensiones (ncoils, px, px)"
        assert csmap.shape[1] == csmap.shape[2], "csmap debe ser cuadrado en las dos últimas dims"

        self.csmap = csmap
        self.N = csmap.shape[1]
        self.ds = get_radon_ds_from_N(self.N)  # tamaño de paso para integración de Radon
        self.spclim = spclim

    # -------------------------------------------------------------------

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

    # -------------------------------------------------------------------

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

def multi_slice_circle_generator(nframes, num_slices, key, addConst, radius=1.0, z_min=-1.0, z_max=1.0):
    """
    Generate a set of circular trajectories across multiple slices along the z-axis.

    This function creates `num_slices` circles distributed linearly between `z_min` and `z_max`. 
    Each circle is parameterized by `nframes` points along its circumference, 
    representing a uniform sampling of the unit circle scaled by `radius`.

    Optionally, a constant random value can be appended to each point as an additional dimension.

    Parameters
    ----------
    nframes : int
        Number of points (frames) sampled along the circumference of each circle.
    num_slices : int
        Number of slices (planes along the z-axis) in which circles will be generated.
    key : jax.random.PRNGKey
        Random key used to generate the optional constant value.
    addConst : bool
        If True, appends a random constant dimension to each point.
    radius : float, optional (default=1.0)
        Radius of the circles.
    z_min : float, optional (default=-1.0)
        Minimum z-coordinate (lower bound of the slices).
    z_max : float, optional (default=1.0)
        Maximum z-coordinate (upper bound of the slices).

    Returns
    -------
    jax.numpy.ndarray
        A stacked array of shape `(num_slices, nframes, 3)` if `addConst=False`, 
        or `(num_slices, nframes, 4)` if `addConst=True`.

        - First dimension: slice index
        - Second dimension: frame index along the circle
        - Last dimension: coordinates `[x, y, z]` or `[x, y, z, const]`

    Examples
    --------
    >>> import jax
    >>> key = jax.random.key(0)
    >>> arr = multi_slice_circle_generator(5, 2, key, addConst=False, radius=2.0)
    >>> arr.shape
    (2, 5, 3)

    >>> arr = multi_slice_circle_generator(5, 2, key, addConst=True)
    >>> arr.shape
    (2, 5, 4)
    """

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

def make_forward_radon_operator(csmap, spclim=0.5):
    N = csmap.shape[1]
    ds = get_radon_ds_from_N(N)

    def radon_transform(im, alphas):
        """
        - im: (batch, px, py)
        - alphas: (batch,)
        """
        im = jnp.moveaxis(im, 0, -1)  # (px, py, batch)
        im = im[None, :, :, :] * csmap[:, :, :, None]  # (ncoils, px, py, batch)
        im = jnp.moveaxis(im, -1, 1)  # (ncoils, batch, px, py)
        im = jax.vmap(rotate, in_axes=(1, 0))(im, alphas)  # (batch, ncoils, px, py)
        im = radon_integration(im, ds, axis=-1)  # (batch, ncoils, px)
        im = fastshiftfourier(im * spclim * 2)
        return im

    return radon_transform

def get_radon_ds_from_N(N):
    ss = 2 * fftshift(fftfreq(N))
    return ss[1] - ss[0]

def to_complex(arr: jnp.ndarray) -> jnp.ndarray:
    """
    arr: (..., 256, 256, 2) con canales [real, imag]
    return: (..., 256, 256) complejo
    """
    real, imag = jnp.split(arr, 2, axis=-1)   # (..., 256, 256, 1) cada uno
    real = real.squeeze(-1)                   # (..., 256, 256)
    imag = imag.squeeze(-1)                   # (..., 256, 256)
    return lax.complex(real, imag)            # preserva dtype (evita upcast a c128)

def sample_from_groups(index_groups: List[jnp.ndarray], n_keep: int, key):
    keys = jax.random.split(key, len(index_groups))
    sampled = []

    for group, k in zip(index_groups, keys):
        group = jnp.asarray(group)
        n = group.shape[0]

        if n_keep <= n:
            # Sample without replacement
            selected = jax.random.choice(k, group, shape=(n_keep,), replace=False)
        else:
            # Repeat and shuffle to ensure enough samples
            repeats = (n_keep + n - 1) // n  # ceiling division
            extended_group = jnp.tile(group, repeats)
            shuffled = jax.random.permutation(k, extended_group)
            selected = shuffled[:n_keep]

        sampled.append(selected)

    return jnp.array(sampled)

# ===============================================================
# U-Net DIP for 2D images (input/output same size, e.g. 256×256)
# ===============================================================

# --- simple, memory-safe upsample (bilinear) ---
def upsample_2d(x: jax.Array, scale: int = 2, method: str = "linear") -> jax.Array:
    if scale == 1:
        return x
    new_shape = (x.shape[0], x.shape[1]*scale, x.shape[2]*scale, x.shape[3])
    return jax.image.resize(x, new_shape, method=method)

# --- basic conv block ---
class ConvReLU(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, k: Tuple[int,int]=(3,3),
                 s: Tuple[int,int]=(1,1), rngs: Optional[nnx.Rngs]=None):
        self.conv = nnx.Conv(in_ch, out_ch, kernel_size=k,
                             strides=s, padding="SAME", rngs=rngs)
    def __call__(self, x: jax.Array, training: bool=False) -> jax.Array:
        return jax.nn.relu(self.conv(x))

# --- downsampling encoder block ---
class DownBlock(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, rngs: Optional[nnx.Rngs]=None):
        self.c1 = ConvReLU(in_ch,  out_ch, rngs=rngs)
        self.c2 = ConvReLU(out_ch, out_ch, rngs=rngs)
        self.down = nnx.Conv(out_ch, out_ch, kernel_size=(3,3),
                             strides=(2,2), padding="SAME", rngs=rngs)
    def __call__(self, x: jax.Array, training: bool=False):
        x = self.c1(x, training=training)
        x = self.c2(x, training=training)
        skip = x
        x = jax.nn.relu(self.down(x))
        return x, skip

class UpBlock(nnx.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, rngs=None):
        self.c1 = ConvReLU(in_ch + skip_ch, out_ch, rngs=rngs)
        self.c2 = ConvReLU(out_ch, out_ch, rngs=rngs)
    def __call__(self, x, skip, training=False):
        x = upsample_2d(x, scale=2, method="linear")
        h = jnp.minimum(x.shape[1], skip.shape[1])
        w = jnp.minimum(x.shape[2], skip.shape[2])
        x, skip = x[:, :h, :w, :], skip[:, :h, :w, :]
        x = jnp.concatenate([x, skip], axis=-1)
        x = self.c1(x, training=training)
        x = self.c2(x, training=training)
        return x

# --- full U-Net DIP model ---
class UNetDIP(nnx.Module):
    """
    Classic U-Net-like Deep Image Prior.
    Input/Output: (B, H, W, C)
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Tuple[int,int,int,int] = (32, 64, 128, 256),
        rngs: Optional[nnx.Rngs] = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        f1, f2, f3, f4 = features

        # Encoder
        self.down1 = DownBlock(in_channels, f1, rngs=rngs)  # 256 -> 128
        self.down2 = DownBlock(f1, f2, rngs=rngs)           # 128 -> 64
        self.down3 = DownBlock(f2, f3, rngs=rngs)           # 64  -> 32
        self.down4 = DownBlock(f3, f4, rngs=rngs)           # 32  -> 16

        # Bottleneck
        self.bot1 = ConvReLU(f4, f4, rngs=rngs)
        self.bot2 = ConvReLU(f4, f4, rngs=rngs)

        # Decoder (correct channel config)
        self.up3 = UpBlock(in_ch=f4, skip_ch=f4, out_ch=f3, rngs=rngs)  # 256 + 256 -> 128
        self.up2 = UpBlock(in_ch=f3, skip_ch=f3, out_ch=f2, rngs=rngs)  # 128 + 128 -> 64
        self.up1 = UpBlock(in_ch=f2, skip_ch=f2, out_ch=f1, rngs=rngs)  # 64  + 64  -> 32

        # Final fusion with first skip (s1): concat(d1_up, s1) => channels = f1 + f1
        self.up0_conv = ConvReLU(2 * f1, f1, rngs=rngs)  # was (f1 + in_channels, f1)
        self.out_conv = nnx.Conv(f1, out_channels, kernel_size=(1,1),
                                 strides=(1,1), padding="SAME", rngs=rngs)

    def __call__(self, x: jax.Array, training: bool=False) -> jax.Array:
        # Encoder
        x1, s1 = self.down1(x, training=training)  # 256x256 -> skip s1
        x2, s2 = self.down2(x1, training=training) # 128x128 -> skip s2
        x3, s3 = self.down3(x2, training=training) # 64x64  -> skip s3
        x4, s4 = self.down4(x3, training=training) # 32x32  -> skip s4

        # Bottleneck
        b = self.bot1(x4, training=training)
        b = self.bot2(b,  training=training)

        # Decoder
        d3 = self.up3(b,  s4, training=training)  # 32->64
        d2 = self.up2(d3, s3, training=training)  # 64->128
        d1 = self.up1(d2, s2, training=training)  # 128->256 (after upsample below)

        # Final fusion with first skip (s1)
        d1_up = upsample_2d(d1, scale=2, method="linear")  # 128->256
        h = jnp.minimum(d1_up.shape[1], s1.shape[1])
        w = jnp.minimum(d1_up.shape[2], s1.shape[2])
        d1_up = d1_up[:, :h, :w, :]
        s1 = s1[:, :h, :w, :]
        x = jnp.concatenate([d1_up, s1], axis=-1)

        # Output
        x = self.up0_conv(x, training=training)
        y = self.out_conv(x)

        return y


# Fourier transform utilities
def fft2c(x):  # centered 2D FFT
    return jnp.fft.fftshift(
        jnp.fft.fft2(jnp.fft.ifftshift(x), norm="ortho")
    )

def ifft2c(k):
    return jnp.fft.fftshift(
        jnp.fft.ifft2(jnp.fft.ifftshift(k), norm="ortho")
    )


def downsample_sinc_image(x_hr, s=2):
    """
    Downsample ideal (sinc) de una imagen 2D compleja.
    """
    # Acepta tanto (N,N)
    if x_hr.ndim == 4:
        x_hr = x_hr[0, :, :, 0]
    elif x_hr.ndim == 3:
        x_hr = x_hr[0, :, :]
        
    N = x_hr.shape[0]
    N_LR = N // s

    k = fft2c(x_hr)
    c = N // 2
    h = N_LR // 2
    k_lr = k[c - h : c + h, c - h : c + h]
    x_lr = ifft2c(k_lr)
    return x_lr[None, :, :], k_lr



def apply_radial_mask_image(x_hr, mask):
    """
    Simulate MRI acquisition by applying a radial sampling mask to k-space.
    Args:
        x_hr: (1,H,W) (HR sizes)
        mask: (h, w) binary sampling mask {0,1} (LR sizes)
    Returns:
        x_reco: reconstructed (zero-filled) image
        k_masked: masked k-space
    """
    if x_hr.ndim == 3:
        x_hr = x_hr[0, :, :] # (N,N)

    _, k_lr = downsample_sinc_image(x_hr)
    k_masked = k_lr * mask
    return k_masked

def exponential_kspace_weight(N, c=0.8):
    """
    Compute exponential weighting map for k-space.
    Args:
        N: image size (NxN)
        c: control parameter (0.2-1.0 typical)
    Returns:
        W_exp: (N, N) array with exponential weights
    """
    ky, kx = jnp.meshgrid(jnp.arange(-N//2, N//2),
                          jnp.arange(-N//2, N//2),
                          indexing='ij')
    radius = jnp.sqrt(kx**2 + ky**2)
    shell = radius / (N/2)  # normalized distance (0..1)
    W_exp = 2**(c * shell)  # exponential weighting
    return W_exp

def weighted_complex_mse(a, b, W):
    diff = a - b
    return jnp.mean(W * jnp.abs(diff)**2)

def charbonnier_loss(I1,I2, epsilon=1e-3):
    return jnp.mean(jnp.sqrt(jnp.abs(I1-I2) + epsilon**2))

def radial_mask(N, n_spokes=20, radius_ratio=1.0):
    """
    Create a radial k-space sampling mask (centered).
    
    Args:
        N: image dimension (N x N)
        n_spokes: number of radial lines crossing center
        radius_ratio: proportion of radius to keep (0-1)
    Returns:
        mask: (N, N) array, binary {0,1}
    """
    yy, xx = jnp.meshgrid(jnp.linspace(-1, 1, N),
                          jnp.linspace(-1, 1, N),
                          indexing='ij')
    r = jnp.sqrt(xx**2 + yy**2)
    theta = jnp.arctan2(yy, xx)

    # radial sampling lines
    mask = jnp.zeros_like(r)
    for i in range(n_spokes):
        angle = i * jnp.pi / n_spokes
        mask = jnp.logical_or(mask, jnp.abs(jnp.sin(theta - angle)) < (1.5/N))

    # circular boundary
    mask = jnp.logical_and(mask, r <= radius_ratio)
    return mask.astype(float)
