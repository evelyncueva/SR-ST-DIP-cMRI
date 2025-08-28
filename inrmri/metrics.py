import numpy as onp 
import jax.numpy as np 
from skimage.metrics import structural_similarity as ssim

from image_similarity_measures.quality_metrics import fsim as ism_fsim
from image_similarity_measures.quality_metrics import issm as ism_issm
from image_similarity_measures.quality_metrics import psnr as ism_psnr
from image_similarity_measures.quality_metrics import rmse as ism_rmse

from inrmri.basic_nn import psnr

def get_range(array): 
    return onp.max(array) - onp.min(array)

def ssim_metric(im1, im2, data_range = None):
    data_range = data_range or get_range(onp.abs(im2))
    return ssim(onp.abs(im1), onp.abs(im2), data_range=data_range)

def mean_ssim(gtim, im, data_range=None):
    ssim = onp.mean([ssim_metric(gtim[...,i], im[...,i], data_range=data_range) for i in range(im.shape[-1])])
    return ssim

def hist_2d(im1, im2, bins:int):
    hist = np.histogram2d(im1.flatten(), im2.flatten(), bins=bins)[0]
    return hist

def log_at_non_zero(arr):
    non_zeros = arr != 0
    arr = arr.at[non_zeros].set(np.log(arr[non_zeros]))
    return arr

def mutual_information(hgram):

    """ Mutual information for joint histogram

    """
    # Convert bins counts to probability values
    pxy = hgram / np.array(np.sum(hgram), float)
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def mutual_information_metric(im1, im2, bins=64):
    return mutual_information(hist_2d(im1, im2, bins))

METRIC_FUNCTIONS = {
    'ssim_3D':ssim_metric, 
    'mean_ssim_2D':mean_ssim, 
    'psnr': psnr, 
    'ism_fsim': ism_fsim, 
    'ism_issm': ism_issm, 
    'ism_psnr': ism_psnr, 
    'ism_rmse': ism_rmse,
    'nmi': mutual_information_metric
}