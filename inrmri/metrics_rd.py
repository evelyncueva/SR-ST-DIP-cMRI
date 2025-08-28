import jax
import jax.numpy as jnp
from jax import jit, vmap

# ---------------------------
# Artifact Power
# ---------------------------

@jit
def artifact_power(img_pred, img_gt):
    error = img_pred - img_gt
    power_error = jnp.mean(error ** 2)
    power_gt = jnp.mean(img_gt ** 2)
    return power_error / (power_gt + 1e-8)

@jit
def mean_artifact_power(images_pred, images_gt):
    compute = vmap(artifact_power, in_axes=(2, 2))
    results = compute(images_pred, images_gt)
    return jnp.mean(results)

# ---------------------------
# PSNR
# ---------------------------

@jit
def psnr(img_pred, img_gt, max_val=1.0):
    mse = jnp.mean((img_pred - img_gt) ** 2)
    return 20 * jnp.log10(max_val) - 10 * jnp.log10(mse + 1e-8)

@jit
def mean_psnr(images_pred, images_gt, max_val=1.0):
    compute = vmap(lambda x, y: psnr(x, y, max_val), in_axes=(2, 2))
    results = compute(images_pred, images_gt)
    return jnp.mean(results)

# ---------------------------
# SSIM
# ---------------------------

@jit
def ssim(img_pred, img_gt, max_val=1.0, k1=0.01, k2=0.03):
    C1 = (k1 * max_val) ** 2
    C2 = (k2 * max_val) ** 2
    
    mu_pred = jnp.mean(img_pred)
    mu_gt = jnp.mean(img_gt)
    sigma_pred = jnp.var(img_pred)
    sigma_gt = jnp.var(img_gt)
    sigma_pred_gt = jnp.mean((img_pred - mu_pred) * (img_gt - mu_gt))
    
    numerator = (2 * mu_pred * mu_gt + C1) * (2 * sigma_pred_gt + C2)
    denominator = (mu_pred**2 + mu_gt**2 + C1) * (sigma_pred + sigma_gt + C2)
    
    return numerator / (denominator + 1e-8)

@jit
def mean_ssim(images_pred, images_gt, max_val=1.0):
    compute = vmap(lambda x, y: ssim(x, y, max_val), in_axes=(2, 2))
    results = compute(images_pred, images_gt)
    return jnp.mean(results)
