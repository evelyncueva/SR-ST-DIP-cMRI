# -------------------------------------------------------------------------------------# 
# new_radon.py 
# Functions to calculate the Radon Transform of an MRI image, including coil
# sensitivities. Es similar a src/radon_training.py, pero ese archivo es específico para
# entrenar Implicit Neural Representations (donde puedo evaluar en coordenadas
# arbitrarias), mientras que este sirve para imágenes (donde tengo que interpolar). 
# -------------------------------------------------------------------------------------# 
from inrmri.radon import _radon_points 
from inrmri.radon_training import interpolate_points_to_grid 
from inrmri.fourier import fastshiftfourier, get_freqs
from jax.numpy.fft import fftshift, fftfreq
from jax import vmap 
import jax.numpy as np 
import jax

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

def radon_integration(im, ds, axis=-1):
    return np.sum(im, axis=axis) * ds 

def get_radon_ds_from_N(N):
    ss = 2 * fftshift(fftfreq(N))
    return ss[1] - ss[0]

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
        im = np.moveaxis(im, 0, -1)                     # (px, py, batch)

        # Ponderar por mapas de sensibilidad de coils
        im = im[None, :, :, :] * self.csmap[:, :, :, None]  # (ncoils, px, py, batch)

        # Reordenar: batch → eje 1
        im = np.moveaxis(im, -1, 1)                     # (ncoils, batch, px, py)

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

    
# --------------------------------------------------------------------------------- #
# k-space filters 
# --------------------------------------------------------------------------------- #

def cosine_filter(N):
   filter_freqs = np.sin(np.linspace(0, np.pi, N, endpoint=False)) # cosine filter 
   return filter_freqs 

def ramp_filter(N):
   filter_freqs = np.ones(N)
   return filter_freqs 

def shepplogan_filter(N):
    filter_freqs = np.fft.fftshift(np.sinc(np.fft.fftfreq(N))) # shepp logan filter 
    return filter_freqs 

FILTERS = {
    'ramp'          : ramp_filter,
    'cosine'        : cosine_filter,
    'shepp-logan'   : shepplogan_filter
}

def get_weight_freqs(N:int, str_filter:str='ramp'):
    """
    Dar un mayor peso a las altas frecuencias 

    - N: int número de puntos en total 
    - str_filter: str en ['ramp', 'cosine' y 'shepp-logan']
    """
    freqs = np.abs(get_freqs(N))
    filter_freqs = FILTERS[str_filter](N)
    return filter_freqs * freqs

def make_forward_radon_operator(csmap, spclim=0.5):
    N = csmap.shape[1]
    ds = get_radon_ds_from_N(N)

    def radon_transform(im, alphas):
        """
        - im: (batch, px, py)
        - alphas: (batch,)
        """
        im = np.moveaxis(im, 0, -1)  # (px, py, batch)
        im = im[None, :, :, :] * csmap[:, :, :, None]  # (ncoils, px, py, batch)
        im = np.moveaxis(im, -1, 1)  # (ncoils, batch, px, py)
        im = jax.vmap(rotate, in_axes=(1, 0))(im, alphas)  # (batch, ncoils, px, py)
        im = radon_integration(im, ds, axis=-1)  # (batch, ncoils, px)
        im = fastshiftfourier(im * spclim * 2)
        return im

    return radon_transform