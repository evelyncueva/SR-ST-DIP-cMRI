import jax.numpy as np 
from jax import vmap 

def wrapped_vmap(func, in_axes, out_axes, nlevels=1):
    vmapped_func = func
    for _ in range(nlevels):
        vmapped_func = vmap(vmapped_func, in_axes=in_axes, out_axes=out_axes)
    return vmapped_func

def calculate_centered_linspace(Nt):
    """
    ## Ejemplos 
    
    ```python
    >>> N = 5 
    >>> np.linspace(0,1,N) # linspace normal 
    Array([0.  , 0.25, 0.5 , 0.75, 1.  ], dtype=float32)
    >>> np.linspace(0,1,N, endpoint=False) # linspace sin punto final 
    Array([0. , 0.2, 0.4, 0.6, 0.8], dtype=float32)
    >>> calculate_centered_linspace(N)
    Array([0.1       , 0.3       , 0.5       , 0.70000005, 0.90000004],      dtype=float32)
    ```
    """
    ts = np.linspace(0,1,Nt, endpoint=False)
    ts = ts + np.diff(ts)[0]/2
    return ts

class Frame1DInterpolator:

    def __init__(self, frame_list):
        """
        frame_list: np.array, shape (nframes, ...)
        """
        assert frame_list.ndim >= 1
        self.frame_list = frame_list
        self._vmap_levels = frame_list.ndim - 1
        self.nframes = frame_list.shape[0]
        self.ts = calculate_centered_linspace(self.nframes)

    def eval_t(self, t):
        vmapped_func = wrapped_vmap(np.interp, in_axes=(None, None, -1), out_axes=-1, nlevels=self._vmap_levels)
        out = vmapped_func(t, self.ts, self.frame_list)
        return out
    
def bilinear_interp(corners, x, y): 
    """
    ## Ejemplos 
    - corners: (4, ...)
    - output: (...)    
    ```
    >>> corners = Q00, Q01, Q10, Q11
    >>> motion_bilinear_interp(corners, 0.0, 1.0) == Q01 
    True 
    >>> motion_bilinear_interp(corners, 1.0, 0.0) == Q10 
    True 
    ```
    """
    Q00, Q01, Q10, Q11 = corners
    xdif = (1-x, x)
    ydif = (1-y, y)
    a0 = Q00 * ydif[0] + Q01*ydif[1]
    a1 = Q10 * ydif[0] + Q11*ydif[1]
    return xdif[0] * a0 + xdif[1] * a1 
