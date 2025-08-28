import jax.numpy as np 

def TV_space(im, epsilon_smooth_tv = 1e-9): 
    """
    - im: (px,py,...)
    - epsilon_smooth_tv: small positiva value to make a smooth tv 
    """
    dfx = np.diff(im, axis=0)
    dfy = np.diff(im, axis=1)
    tv_val = np.mean(np.abs(dfx)) + np.mean(np.abs(dfy)) + epsilon_smooth_tv
    return tv_val 

def TV_one_axis(im, axis=0, epsilon_smooth_tv = 1e-9): 
    """
    - im: array 
    - axis: axis to calculate tv 
    - epsilon_smooth_tv: small positiva value to make a smooth tv 
    """
    df = np.diff(im, axis=axis)
    tv_val = np.mean(np.abs(df)) + epsilon_smooth_tv
    return tv_val 
