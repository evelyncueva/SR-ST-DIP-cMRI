import jax.numpy as np
from jax import random, grad, jit, vmap
from jax.numpy.fft import fftshift, fftfreq
from functools import partial 

@jit
def normal(alpha): 
  return np.array([np.cos(alpha), np.sin(alpha)])

@jit
def orthogonal(v): 
  vx, vy = v 
  return np.array([-vy, vx])

@jit
def points_in_vector_direction(distances, direction): 
  return distances[:,None] * direction

@jit
def get_ds(ss): 
  return ss[1] - ss[0]

@jit
def linear_reg_coeffs(x,y):
  """
  Return coefficients of linear regression, 
  p[0] + p[1] * x =(approx) y 
  """
  A = np.c_[np.ones(len(x)), x]
  p = np.matmul(np.linalg.pinv(A), y)
  return p 

def print_angle(name, theta):
  print(f"{name}: {theta:.2f} rad, {theta * 180 / np.pi}°")

def calculate_angle(spoke):
  x0, y0 = spoke[0]
  theta = np.arctan2(y0,x0)
  # alpha = np.mod(theta + np.pi / 2, 2 * np.pi)
  alpha = np.mod(theta, 2 * np.pi)
  # x, y = split(spoke) 
  # _, alpha = distance_normal_parametrization_from_coeff(linear_reg_coeffs(x, y)) 
  return alpha 

@partial(jit, static_argnums=[2])
def _radon_points(alpha, lims, N): 
  n = normal(alpha)
  # ss = np.linspace(*lims, N)
  ss = 2 * fftshift(fftfreq(N))
  radonxy = points_in_vector_direction(ss, -n)
  evalradonxy = points_in_vector_direction(ss, orthogonal(n))
  return ss, radonxy, evalradonxy
