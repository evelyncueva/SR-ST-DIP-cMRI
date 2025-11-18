# ===== Flax NNX versión =====
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Sequence, Tuple, Callable
from typing import List
import copy

# ---------- utils: resize 1D/2D (idéntico a Linen) ----------
def upsampling_1d(x: jax.Array, newshape, method: str):
    old_len, channels = x.shape[-2:]
    batch_shape = x.shape[:-2]
    # newshape es el shape del tensor a “igualar” (tú lo pasas desde self.upsample_shape)
    new_len = newshape[-2]
    return jax.image.resize(x, shape=batch_shape + (new_len, channels), method=method)

def upsampling_2d(x: jax.Array, newshape, method: str):
    old_h, old_w, channels = x.shape[-3:]
    batch_shape = x.shape[:-3]
    new_h, new_w = newshape[-3:-1]
    return jax.image.resize(x, shape=batch_shape + (new_h, new_w, channels), method=method)

def new_upsampled_shape(initialshape, upsampling_factor: int, dimensions: int):
    # Igual a función en Linen
    batch_shape = initialshape[:-(dimensions + 1)]
    convolved_shape = initialshape[-(dimensions + 1):-1]
    return batch_shape + tuple(n * upsampling_factor for n in convolved_shape) + (initialshape[-1],)

def split_last_dim(a):
    old = a.shape
    return a.reshape(old[:-1] + (old[-1] // 2, 2))

def to_complex(a):
    real, imag = jnp.split(a, 2, axis=-1)
    return real + 1j * imag


# ----------------- NNX submódulos -----------------
class ConvolutionalDIPBlock_NNX(nnx.Module):
    def __init__(self, features: int, kernel: int, stride: int, momentum: float, dimensions: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(features=features, kernel_size=(kernel,) * dimensions, strides=(stride,) * dimensions, padding='SAME', rngs=rngs)
        self.bn   = nnx.BatchNorm(momentum=momentum, use_running_average=False)

    def __call__(self, x: jax.Array, training: bool, rngs: nnx.Rngs) -> jax.Array:
        x = self.conv(x)
        x = self.bn(x, use_running_average=not training)
        x = jax.nn.relu(x)
        return x


class Decoder_NNX(nnx.Module):
    def __init__(self, features: int = 128, momentum: float = 0.99, levels: int = 3,
                 out_features: int = 2, dimensions: int = 2, upsampling_factor: int = 2,
                 upsampling_method: str = 'nearest', rngs: nnx.Rngs = nnx.Rngs(0)):
        self.features = features
        self.momentum = momentum
        self.levels = levels
        self.out_features = out_features
        self.dimensions = dimensions
        self.upsampling_factor = upsampling_factor
        self.upsampling_method = upsampling_method
        # bloques conv
        self.pre1 = ConvolutionalDIPBlock_NNX(features, 3, 1, momentum, dimensions, rngs)
        self.pre2 = ConvolutionalDIPBlock_NNX(features, 3, 1, momentum, dimensions, rngs)
        # pila intermedia
        self.blocks = [ConvolutionalDIPBlock_NNX(features, 3, 1, momentum, dimensions, rngs) for _ in range(2 * levels)]
        # pos
        self.post1 = ConvolutionalDIPBlock_NNX(features, 3, 1, momentum, dimensions, rngs)
        self.post2 = ConvolutionalDIPBlock_NNX(features, 3, 1, momentum, dimensions, rngs)
        self.head  = nnx.Conv(features=out_features, kernel_size=(3,) * dimensions, strides=(1,) * dimensions, padding='SAME', rngs=rngs)

    def _upsample(self, x, newshape):
        if self.dimensions == 1:
            return upsampling_1d(x, newshape, self.upsampling_method)
        else:
            return upsampling_2d(x, newshape, self.upsampling_method)

    def _upsample_shape(self, shape):
        return new_upsampled_shape(shape, self.upsampling_factor, self.dimensions)

    def __call__(self, x: jax.Array, training: bool, rngs: nnx.Rngs) -> jax.Array:
        x = self.pre1(x, training, rngs)
        x = self.pre2(x, training, rngs)
        x = self._upsample(x, self._upsample_shape(x.shape))
        # niveles: (conv, conv) + upsample
        it = iter(self.blocks)
        for _ in range(self.levels):
            x = next(it)(x, training, rngs)
            x = next(it)(x, training, rngs)
            x = self._upsample(x, self._upsample_shape(x.shape))
        x = self.post1(x, training, rngs)
        x = self.post2(x, training, rngs)
        x = self.head(x)
        return x


class MapNetNNX(nnx.Module):
    def __init__(self,
                 mapnet_layers: Sequence[int],
                 cnn_latent_shape: Tuple[int, int],
                 rngs: nnx.Rngs):
        # Guardamos configuración
        self._sizes: List[int] = list(copy.deepcopy(mapnet_layers))
        if self._sizes:  # solo si no está vacío agregamos px*py
            px, py = cnn_latent_shape
            self._sizes.append(px * py)

        self._layers: List[nnx.Linear] = []  # se construyen en la 1ª llamada
        self._built: bool = False
        self.rngs = rngs

    def _build(self, in_features: int):
        prev = in_features
        self._layers = []
        for i, out_features in enumerate(self._sizes):
            layer = nnx.Linear(prev, out_features, rngs=self.rngs, name=f"mapnet-{i}")
            self._layers.append(layer)
            prev = out_features
        self._built = True

    def __call__(self, t: jax.Array) -> jax.Array:
        # Construcción perezosa de las capas usando el tamaño del input
        if not self._built and self._sizes:
            self._build(in_features=int(t.shape[-1]))

        x = t
        for layer in self._layers:
            x = layer(x)           # (batch, features)
            x = jax.nn.relu(x)
        return x


class tDIP_NNX(nnx.Module):
    def __init__(self, mapnet_layers: Sequence[int], cnn_latent_shape: Tuple[int, int],
                 features: int, momentum: float, levels: int,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        self.mapnet = MapNet_NNX(mapnet_layers, cnn_latent_shape, rngs)
        self.cnn_latent_shape = cnn_latent_shape
        self.decoder = Decoder_NNX(features=features, momentum=momentum, levels=levels, dimensions=2, rngs=rngs)

    def __call__(self, t: jax.Array, training: bool, rngs: nnx.Rngs) -> jax.Array:
        # t: (..., D)
        print("[TRACING] Recompiling tDIP (NNX)...")
        x = self.mapnet(t, training, rngs)                           # (..., px*py)
        x = jnp.reshape(x, x.shape[:-1] + self.cnn_latent_shape)     # (..., px, py)
        x = x[..., None]                                             # (..., px, py, 1)
        x = self.decoder(x, training, rngs)                          # (..., px*, py*, 2)
        return x


# ---------------- wrapper multi-slice (NNX) ----------------
class MS_TD_DIP_Net_NNX(nnx.Module):
    def __init__(self,
                 nframes: int,
                 n_slices: int,
                 key_latent,
                 addConst: bool,
                 latent_generator: Callable[..., jax.Array],
                 radius: float,
                 z_min: float,
                 z_max: float,
                 imshape: Tuple[int, int],
                 mapnet_layers: Sequence[int],
                 cnn_latent_shape: Tuple[int, int] = (8, 8),
                 features: int = 128,
                 momentum: float = 0.99,
                 levels: int = 3,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        # hiperparámetros
        self.nframes = nframes
        self.n_slices = n_slices
        self.imshape = imshape

        # latente (n_slices, nframes, D)
        self.latent = latent_generator(nframes, n_slices, key_latent, addConst,
                                       radius=radius, z_min=z_min, z_max=z_max)

        # red
        self.net = tDIP_NNX(mapnet_layers, cnn_latent_shape, features, momentum, levels, rngs)

    def init_params(self, key: jax.Array):
        # NNX no usa dicts de params separados
        t0 = self.latent[0, :1]                 # (1, D)
        _ = self.net(t0, training=False, rngs=nnx.Rngs(key))
        # en NNX, los parámetros ya viven en self.net
        return None

    def train_forward_pass(self, key: jax.Array, t_index: int, slice_index: int):
        z = self.latent[slice_index, t_index, :]    # (D,)
        y = self.net(z, training=True, rngs=nnx.Rngs(key))  # (1?, 2, H, W) o (2, H, W) según shapes
        # asegurar batch eje si hace falta
        if y.ndim == 4:             # (B, 2, H, W)
            y2 = y
        elif y.ndim == 3:           # (2, H, W)
            y2 = y[None]
        else:
            raise ValueError(f"Unexpected output shape {y.shape}")
        # convertir a complejo y recortar
        real = y2[:, 0]             # (B, H, W)
        imag = y2[:, 1]
        yc = real + 1j * imag
        im = yc[0]
        nx, ny = self.imshape
        im = im[:nx, :ny]
        return im
