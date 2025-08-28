from jax import numpy as np 
from jax import lax 
# %%
def gram_schmidt(vectors):
  """Implementation of the modified Gram-Schmidt orthonormalization algorithm.

  Obtained from: https://github.com/tensorflow/tensorflow/issues/62828#issuecomment-2379030651
  Modified to work also in complex matrixs.

  We assume here that the vectors are linearly independent. Zero vectors will be
  left unchanged, but will also consume an iteration against `num_vectors`.

  From [1]: "MGS is numerically equivalent to Householder QR factorization
  applied to the matrix A augmented with a square matrix of zero elements on
  top."

  Historical note, see [1]: "modified" Gram-Schmidt was derived by Laplace [2],
  for elimination and not as an orthogonalization algorithm. "Classical"
  Gram-Schmidt actually came later [2]. Classical Gram-Schmidt has a sometimes
  catastrophic loss of orthogonality for badly conditioned matrices, which is
  discussed further in [1].

  #### References

  [1] Bjorck, A. (1994). Numerics of gram-schmidt orthogonalization. Linear
      Algebra and Its Applications, 197, 297-316.

  [2] P. S. Laplace, Thiorie Analytique des Probabilites. Premier Supple'ment,
      Mme. Courtier, Paris, 1816.

  [3] E. Schmidt, über die Auflosung linearer Gleichungen mit unendlich vielen
      Unbekannten, Rend. Circ. Mat. Pulermo (1) 25:53-77 (1908).

  Args:
    vectors: A Tensor of shape `[d, n]` of `d`-dim column vectors to
      orthonormalize.

  Returns:
    A Tensor of shape `[d, n]` corresponding to the orthonormalization.
  """
  num_vectors = vectors.shape[-1]

  def body_fn(vecs, i):
    # Slice out the vector w.r.t. which we're orthogonalizing the rest.
    u = np.nan_to_num(vecs[:,i]/np.linalg.norm(vecs[:,i]))
    # Find weights by dotting the d x 1 against the d x n.
    weights = u@np.conjugate(vecs)
    # Project out vector `u` from the trailing vectors.
    masked_weights = np.where(np.arange(num_vectors) > i, weights, 0.)
    vecs = vecs - np.outer(u,np.conjugate(masked_weights))
    return vecs, None

  vectors, _ = lax.scan(body_fn, vectors, np.arange(num_vectors - 1))
  vec_norm = np.linalg.norm(vectors, axis=0, keepdims=True)
  return np.nan_to_num(vectors/vec_norm)
