#!/usr/bin/env python3

import os
import argparse
import jax
import jax.numpy as np
import matplotlib.pyplot as plt

# Your imports
from inrmri.image_processor import reduce_FOV
from inrmri.basic_plotting import full_halph_FOV_space_time
from inrmri.radial_acquisitions import RadialAcquisitions

from misc_nnx import (
    downsample_sinc_image,
    radial_mask,
    exponential_kspace_weight,
    charbonnier_loss,
    weighted_complex_mse,
    apply_radial_mask_image,
    fft2c,
    UNetDIP,
)

from flax import nnx
import optax

print("reading file")
# =====================================================================
#        TOTAL VARIATION
# =====================================================================
def tv(I, eps=1e-6, normalize=True):
    """Isotropic TV for complex 2D images."""
    dx = I[:, 1:] - I[:, :-1]
    dy = I[1:, :] - I[:-1, :]

    dx = dx[:-1, :]
    dy = dy[:, :-1]

    g2 = np.abs(dx) ** 2 + np.abs(dy) ** 2
    tv_val = np.sum(np.sqrt(g2 + eps))

    if normalize:
        tv_val = tv_val / (I.shape[0] * I.shape[1])
    return tv_val


# =====================================================================
#        LOSS FUNCTION
# =====================================================================
def build_loss_fn(x_lr_true, k_lr_true, mask_lr, W_exp):

    def loss_fn(model, beta=0.001, lambda_tv=1e-2):
        y_hr = model(z_hr, training=True)
        I_hr_pred = y_hr[..., 0] + 1j * y_hr[..., 1]

        k_lr_pred = apply_radial_mask_image(I_hr_pred, mask_lr)

        y_lr = model(z_lr, training=True)
        I_lr_pred2 = y_lr[..., 0] + 1j * y_lr[..., 1]
        k_lr_pred2 = fft2c(I_lr_pred2[0]) * mask_lr

        # Unsupervised branch
        I_lr_pred, _ = downsample_sinc_image(I_hr_pred[0], 2)
        L_img1 = charbonnier_loss(I_lr_pred, x_lr_true)
        L_data1 = weighted_complex_mse(k_lr_pred * mask_lr,
                                       k_lr_true * mask_lr,
                                       W_exp)
        L_US = L_img1 + L_data1

        # Supervised branch
        L_img2 = charbonnier_loss(I_lr_pred2, x_lr_true)
        L_data2 = weighted_complex_mse(k_lr_pred2 * mask_lr,
                                       k_lr_true * mask_lr,
                                       W_exp)
        L_SS = beta * (L_img2 + L_data2)

        # TV
        L_TV = lambda_tv * tv(I_hr_pred[0])

        return L_US + L_SS + L_TV

    return loss_fn


# =====================================================================
#        MAIN SCRIPT
# =====================================================================
def main(args):

    # ---------------------------------------------------
    # 1. Create output directories
    # ---------------------------------------------------
    os.makedirs(args.output, exist_ok=True)
    img_dir = os.path.join(args.output, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"\n Saving outputs to: {args.output}")

    # ---------------------------------------------------
    # 2. Load data
    # ---------------------------------------------------
    print("Loading data...")

    folder = args.data_folder

    data = np.load(folder + 'sl5-data.npy')
    trajs = np.load(folder + 'sl5-traj.npy')
    radial_acquisition = RadialAcquisitions(trajs, data)

    csmap = np.load(folder + 'sl5-csmap.npy')
    reco_sense = np.load(folder + 'sl5-reco-sense.npy')
    reco_grasp = np.load(folder + 'sl5-reco-grasp.npy')

    print("Data loaded!")
    print(radial_acquisition)

    # ---------------------------------------------------
    # 3. Prepare frame
    # ---------------------------------------------------
    frame = args.frame
    N = args.N

    x_temp = reco_grasp[:, :, frame]
    c = 256 // 2
    h = N // 2
    x_true = x_temp[c - h : c + h, c - h : c + h]

    s = 2
    N_LR = N // s
    x_lr_true, k_lr_true = downsample_sinc_image(x_true, s)

    # ---------------------------------------------------
    # 4. Noise
    # ---------------------------------------------------
    key = jax.random.key(args.seed)
    global z_hr, z_lr
    z_hr = jax.random.normal(key, (1, N, N, 1))

    z_lr, _ = downsample_sinc_image(z_hr, s=2)
    z_lr = z_lr[..., None]

    # ---------------------------------------------------
    # 5. Sampling masks + weights
    # ---------------------------------------------------
    mask_lr = np.ones((N_LR, N_LR))
    W_exp = exponential_kspace_weight(N_LR, c=0.8)

    # ---------------------------------------------------
    # 6. Build model + optimizer
    # ---------------------------------------------------
    print("Building model...")

    rng = nnx.Rngs(jax.random.key(args.seed))
    model = UNetDIP(
        in_channels=1,
        out_channels=2,
        features=(32, 64, 128, 256),
        rngs=rng
    )

    optimizer = nnx.Optimizer(
        model=model,
        tx=optax.adam(args.lr),
        wrt=nnx.Param
    )

    # ---------------------------------------------------
    # 7. Build loss function
    # ---------------------------------------------------
    loss_fn = build_loss_fn(x_lr_true * 10, k_lr_true, mask_lr, W_exp)

    # ---------------------------------------------------
    # 8. Training loop
    # ---------------------------------------------------
    print("Starting training...\n")
    losses = []

    for step in range(args.steps):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        losses.append(loss)

        if step % args.print_every == 0:
            print(f"Step {step:5d} | Loss = {float(loss):.6e}")

        if step % args.save_every == 0:
            y = model(z_hr, training=False)
            pred = np.abs(y[..., 0] + 1j * y[..., 1])[0]

            plt.figure(figsize=(5, 5))
            plt.imshow(pred, cmap="gray")
            plt.title(f"Step {step}")
            plt.axis("off")

            out_file = os.path.join(img_dir, f"pred_step_{step:05d}.png")
            plt.savefig(out_file, bbox_inches="tight")
            plt.close()

    # ---------------------------------------------------
    # Final save
    # ---------------------------------------------------
    print("\nFinished training.")
    np.save(os.path.join(args.output, "losses.npy"), np.array(losses))
    print(f"Saved losses to {args.output}/losses.npy\n")


