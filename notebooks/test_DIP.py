import numpy as np
import jax.numpy as jnp
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import matplotlib.pyplot as plt
import optax
import matplotlib.pyplot as plt

# 1. Generate Shepp–Logan phantom
N = 128
phantom = shepp_logan_phantom()
phantom = resize(phantom, (N, N), anti_aliasing=True)
x_true = jnp.array(phantom, dtype=jnp.float32)

# 2. Compute full k-space (complex)
k_true = jnp.fft.fft2(x_true)

# 3. Generate radial mask (spokes)
def radial_points_mask(N, n_lines=16, n_points=None):
    if n_points is None:
        n_points = N
    center = (N - 1) / 2.0
    radius = N / 2.0
    mask = np.zeros((N, N), dtype=np.float32)
    angles = np.linspace(0, np.pi, n_lines, endpoint=False)
    r = np.linspace(-radius, radius, n_points)
    for a in angles:
        x = center + r * np.cos(a)
        y = center + r * np.sin(a)
        x_idx = np.clip(np.round(x).astype(int), 0, N - 1)
        y_idx = np.clip(np.round(y).astype(int), 0, N - 1)
        mask[y_idx, x_idx] = 1.0
    return jnp.array(mask)

M = radial_points_mask(N, n_lines=128)
# M = jnp.ones((N,N))

# 4. Apply mask to simulate undersampled k-space
y = M * k_true


def loss_fn(model):
    # Imagen predicha y FFT
    x_pred = model(z, training=True)[0, :, :, 0]
    k_pred = jnp.fft.fft2(x_pred)

    # Coordenadas de frecuencia normalizadas
    ky, kx = jnp.meshgrid(jnp.fft.fftfreq(N), jnp.fft.fftfreq(N))
    r = jnp.sqrt(kx**2 + ky**2)

    # Peso: más grande en altas frecuencias
    # Usa una forma suave para evitar discontinuidades
    freq_weight = 1.0 + 5.0 * r  # o r**gamma, gamma≈0.5-1 según pruebas

    # Diferencia ponderada en posiciones muestreadas
    diff = M * (k_pred - y)
    weighted_diff = freq_weight * diff

    # Pérdida final
    loss = jnp.sum(jnp.abs(weighted_diff)**2) / (jnp.sum(M) + 1e-8)
    return loss

import jax
key = jax.random.key(0)
from flax import nnx
from misc_nnx import DecoderNNX
import matplotlib.pyplot as plt

z = jax.random.normal(key, (1, 8, 8, 1))

model = DecoderNNX(
    in_channels=1,
    features=64,
    levels=3,       # 8→16→32→64→128
    out_features=1,
    rngs=nnx.Rngs(key)
)
x_init = model(z, training=False)[0, :, :, 0]



# Optimizer
optimizer = nnx.Optimizer(
    model=model,
    tx=optax.adam(1e-3),
    wrt=nnx.Param
)

n_steps = 10_000
losses = []

for step in range(n_steps):
    # Compute loss and grads
    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # Update model parameters
    optimizer.update(model, grads)

    losses.append(loss)

    # Logging
    if step % 10 == 0:
        print(f"Step {step:5d}, Loss: {loss:.4e}")

    # Visualization (use inference mode)
    if step % 100 == 0:
        x_reco = model(z, training=False)[0, :, :, 0]
        plt.figure(figsize=(4,4))
        plt.imshow(x_reco, cmap='gray')
        plt.title(f"Step {step}")
        plt.axis('off')
        plt.show()