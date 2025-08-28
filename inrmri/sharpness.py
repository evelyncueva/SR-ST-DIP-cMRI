# -------------------------------------------------------------- # 
# sharpness.py 
# cálculo de la sharpness usada por [1] en base a [2]
# [1]: G. Cruz et al, “Single-heartbeat cardiac cine imaging via jointly regularized nonrigid motion-corrected reconstruction,” _NMR in Biomedicine_, vol. 36, no. 9, p. e4942, 2023, doi: [10.1002/nbm.4942](https://doi.org/10.1002/nbm.4942).
# [2]: S. M. Shea et al., “Coronary artery imaging: 3D segmented k-space data acquisition with multiple breath-holds and real-time slab following,” _Journal of Magnetic Resonance Imaging_, vol. 13, no. 2, pp. 301–307, 2001, doi: [10.1002/1522-2586(200102)13:2<301::AID-JMRI1043>3.0.CO;2-8](https://doi.org/10.1002/1522-2586\(200102\)13:2<301::AID-JMRI1043>3.0.CO;2-8).
# -------------------------------------------------------------- # 
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def get_line_points_from_angle(center, angle, shape):
    """
    Devuelve puntos a lo largo de una línea que pasa por `center` en dirección `angle`,
    manteniéndose dentro de una imagen de tamaño `shape = (Nx, Ny)`.
    
    Parameters:
    - center: array_like (x0, y0)
    - angle: float (radianes)
    - shape: tuple (Nx, Ny)

    Returns:
    - points: array de shape (npoints, 2) con coordenadas (x, y)
    """
    cx, cy = center
    dx, dy = np.cos(angle), np.sin(angle)
    Nx, Ny = shape

    # t en que la línea intersecta los bordes de la imagen
    t_vals = []

    if dx != 0:
        t_vals += [(0 - cx) / dx, (Nx - 1 - cx) / dx]
    if dy != 0:
        t_vals += [(0 - cy) / dy, (Ny - 1 - cy) / dy]

    t_vals = np.array(t_vals)
    # calcular puntos candidatos
    points = center + t_vals[:, None] * np.array([dx, dy])
    # filtrar solo los que caen dentro de los límites de la imagen
    valid_mask = np.logical_and.reduce([
        points[:, 0] >= 0, points[:, 0] <= Nx - 1,
        points[:, 1] >= 0, points[:, 1] <= Ny - 1
    ])
    points = points[valid_mask]

    if points.shape[0] != 2:
        raise ValueError("No se pudieron encontrar dos extremos válidos dentro de la imagen.")

    p1, p2 = points
    line_len = np.linalg.norm(p2 - p1)
    npoints = int(np.round(line_len))
    lmbda = np.linspace(0, 1, npoints)[:, None]
    line_points = (1 - lmbda) * p1 + lmbda * p2
    idx_center = np.argmin(np.linalg.norm(line_points - np.array([cx, cy])[None,:], axis=-1))

    return line_points, idx_center 

def extract_line_at_points(im, points):
    N, M = im.shape[:2]
    interp = RegularGridInterpolator([np.arange(N), np.arange(M)], np.abs(im))
    return interp(points)


def find_convex_coeff(x, y, xp):
    """
    Compute the convex combination coefficient lambda such that:
        xp = (1 - lambda) * x + lambda * y

    Parameters
    ----------
    x : float or array-like
        Lower bound value(s).
    y : float or array-like
        Upper bound value(s), must be different from x.
    xp : float or array-like
        Intermediate value(s) between x and y.

    Returns
    -------
    lambda : float or array-like
        Convex coefficient(s) in [0,1] such that xp = (1 - lambda) * x + lambda * y.

    Raises
    ------
    ValueError
        If x and y are equal at any position (division by zero).
    """
    denom = y - x
    if np.any(denom == 0):
        raise ValueError("x and y must be different to compute a convex coefficient.")
    return (xp - x) / denom

def find_percentile_idx(sorted_profile, percentile):
    """
    - sorted profile: arr en orden creciente 
    - percentile en [0,100]
    """
    a = np.abs(sorted_profile).min()
    b = np.abs(sorted_profile).max()

    perc_value = a + (b - a) * percentile/100
    idx        = np.searchsorted(sorted_profile, perc_value) # indices por la derecha 
    lmbda      = find_convex_coeff(sorted_profile[idx-1], sorted_profile[idx], perc_value)
    # pseudo_idx   = (1-lmbda) * (idx-1) + lmbda * (idx)
    if lmbda <= 0.5: 
        idx = idx-1
    return idx, perc_value

def get_distance(points):
    """
    - points: (n, 2)
    """
    return np.mean(np.linalg.norm(np.diff(points, axis=0), axis=1)) # distancia entre los puntos


def distance_20_80_from_profile(profile):
    """
    Compute the pixel distance between the 20th and 80th range in a profile.

    Parameters
    ----------
    profile : ndarray
        1D array of intensity values.

    Returns
    -------
    d : int
        Absolute distance (in pixels) between the points associated with 20th
        and 80th range.
    """
    idx_sorting = np.argsort(profile)

    idx_sorted_20, _ = find_percentile_idx(profile[idx_sorting], 20) 
    idx_sorted_80, _ = find_percentile_idx(profile[idx_sorting], 80) 

    corrected_idxs = np.arange(profile.shape[0])[idx_sorting]

    idx_20 = corrected_idxs[idx_sorted_20]
    idx_80 = corrected_idxs[idx_sorted_80]

    d = np.abs(idx_20 - idx_80)
    return d 

def distance_20_80_from_im_and_points(im, points): 
    """
    Compute the 20-80 distance across a sampled line in the image.

    Parameters
    ----------
    im : ndarray
        2D array (image) of shape (nx, ny).
    points : ndarray
        Array of shape (N, 2) containing (x, y) points along which to sample the image.

    Returns
    -------
    d_mm : float
        Distance in physical units (pixels scaled by spacing between sampled points)
        between the points of 20th and 80th range of the sampled intensity profile.
    """

    line_profile = extract_line_at_points(im, points)
    dx = get_distance(points)
    d = distance_20_80_from_profile(line_profile)
    return d * dx

def radial_sharpness(im, center, angle:float, radius:int):
    """
    Estimate radial sharpness at a given angle by measuring the 20–80 distance.

    Parameters
    ----------
    im : ndarray[float]
        2D input image array of shape (nx, ny).
    center : array[int] 
        1D array oh shape (2,). Center coordinates (x, y) for radial measurement.
    angle : float
        Angle in radians along which to sample the radial line.
    radius : int
        Number of pixels to extend in each direction from the center along the line.

    Returns
    -------
    sharpness : float
        Inverse of the averaged 20–80 distance (higher means sharper).
    """
    points, idx_center = get_line_points_from_angle(center, angle, im.shape)
    points_left  = points[idx_center-radius:idx_center,:]
    points_right = points[idx_center:idx_center+radius,:]

    d_left  = distance_20_80_from_im_and_points(im, points_left)
    d_right = distance_20_80_from_im_and_points(im, points_right)

    d = 0.5 * (d_right + d_left)
    sharpness = 1/d 
    return sharpness 