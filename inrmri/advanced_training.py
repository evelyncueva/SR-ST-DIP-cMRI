import optax 
from tqdm import tqdm 
import jax 
import jax.numpy as jnp 
from flax.core import FrozenDict
import jax.numpy as np
import time
from typing import List
from jax import jit
from inrmri.metrics_rd import mean_psnr, mean_ssim, mean_artifact_power
import gc


class OptimizerWithExtraState: 
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer 
    
    def init(self, params, extra): 
        self.extra = extra 
        return self.optimizer.init(params)
    
    def update(self, grads, opt_state, params):
        return self.optimizer.update(grads, opt_state, params)

# optimizer = OptimizerWithExtraState(optax.adam(1e-3))

# opt_state = optimizer.init(params['params'], params['batch_stats']) #parameters initial 

# la perdida necesita un key 
# la perdida produce un elemento que debe ser actualizado en la funcion 


def train_with_updates(
    loss,
    X,
    Y,
    fullparams_,
    optimizer,
    key,
    nIter: int = 5000,
    batch_size: int = 5000,
    save_at: int = 1000,
):
    """
    Entrena una red neuronal en JAX usando un optimizador con estado extra.

    Parameters
    ----------
    loss : Callable
        Función de pérdida, debe devolver (loss_value, updates).
    X : np.ndarray
        Dataset de entrada, shape (n_samples, ...).
    Y : np.ndarray
        Etiquetas / datos esperados, shape (n_samples, ...).
    fullparams_ : PyTree
        Parámetros iniciales del modelo (incluye 'params' y extras).
    optimizer : OptimizerWithExtraState
        Optimizador de Optax extendido con estado extra.
    key : jax.random.PRNGKey
        Clave aleatoria de JAX.
    nIter : int, default=5000
        Número total de iteraciones de entrenamiento.
    batch_size : int, default=5000
        Número de ejemplos a muestrear en cada iteración.
    save_at : int, default=1000
        Cada cuántas iteraciones guardar los parámetros en el historial.

    Returns
    -------
    results : dict
        Diccionario con:
        - 'last_param' : últimos parámetros entrenados.
        - 'best_param' : mejores parámetros encontrados (según loss).
        - 'train_loss' : lista de pérdidas registradas cada 100 iteraciones.
        - 'best_loss'  : mejor valor de pérdida alcanzado.
        - 'iterations' : lista de iteraciones guardadas.
        - 'param_history' : historial de parámetros guardados cada `save_at`.
    """
    # Separar parámetros iniciales en 'params' y 'extras'
    fullparams_ = FrozenDict(fullparams_)
    extras, params = fullparams_.pop("params")

    # Inicializar estado del optimizador
    opt_state = optimizer.init(params, extras)

    # -----------------------------------------------------------
    # Paso de entrenamiento (1 iteración)
    # -----------------------------------------------------------
    def step(loss, params, opt_state, X, Y, key, niter):
        updated_key = jax.random.fold_in(key, niter)

        # Definir pérdida auxiliar para cálculo de gradientes
        def aux_loss(params):
            full_params = {"params": params, **optimizer.extra}
            return loss(full_params, X, Y, updated_key)

        # Calcular pérdida y gradientes
        (loss_value, updates_from_loss), grads = jax.value_and_grad(
            aux_loss, has_aux=True
        )(params)

        # Actualizar parámetros
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Guardar actualizaciones extra en el optimizador
        optimizer.extra = updates_from_loss
        return params, opt_state, loss_value

    # -----------------------------------------------------------
    # Loop de entrenamiento
    # -----------------------------------------------------------
    train_loss = []
    best_loss = jnp.inf
    best_param = None
    iterations = []
    param_history = {}

    for it in tqdm(range(nIter), desc="train iter", leave=True):
        key, key_batch, key_loss = jax.random.split(key, 3)

        # Selección aleatoria de un batch
        idx_batch = jax.random.choice(
            key_batch, X.shape[0], shape=(batch_size,)
        )
        params, opt_state, train_loss_value = step(
            loss, params, opt_state, X[idx_batch], Y[idx_batch], key_loss, it
        )

        # Guardar mejor pérdida
        if train_loss_value < best_loss:
            best_loss = train_loss_value
            best_param = params

        # Guardar métricas periódicamente
        if it % 100 == 0:
            train_loss.append(train_loss_value)
            print(f" it {it}, train loss = {train_loss_value:.6e}")

            if it % save_at == 0:
                param_history[f"param-{it // save_at}"] = {
                    "params": params,
                    **optimizer.extra,
                }
            iterations.append(it)

    # -----------------------------------------------------------
    # Resultados finales
    # -----------------------------------------------------------
    results = {
        "last_param": params,
        "best_param": best_param,
        "train_loss": train_loss,
        "best_loss": best_loss,
        "iterations": iterations,
        "param_history": param_history,
    }
    return results

### Nuevas funciones de Rafael

@jit
def compute_scalar_variance(volume_list, unbiased=True):
    """
    Compute scalar variance across a list of 3D volumes.
    """
    stack = np.stack(volume_list, axis=0)  # shape (T, H, W, D)
    mean = np.mean(stack, axis=0)
    var = np.mean((stack - mean)**2, axis=0)
    if unbiased and len(volume_list) > 1:
        var *= len(volume_list) / (len(volume_list) - 1)
    return np.mean(var)  # single scalar value

def sample_from_groups(index_groups: List[np.ndarray], n_keep: int, key):
    keys = jax.random.split(key, len(index_groups))
    sampled = []

    for group, k in zip(index_groups, keys):
        group = np.asarray(group)
        n = group.shape[0]

        if n_keep <= n:
            # Sample without replacement
            selected = jax.random.choice(k, group, shape=(n_keep,), replace=False)
        else:
            # Repeat and shuffle to ensure enough samples
            repeats = (n_keep + n - 1) // n  # ceiling division
            extended_group = np.tile(group, repeats)
            shuffled = jax.random.permutation(k, extended_group)
            selected = shuffled[:n_keep]

        sampled.append(selected)

    return np.array(sampled)


def to_python_scalar(x):
    # Converts JAX or NumPy scalar to Python float
    return float(x) if hasattr(x, "item") else x


def train_with_updates_ms_nspokeswise_select(
    loss,
    X_list, Y_list,
    fullparams_, optimizer, key, h_params,
    *,
    recon_cine=None,                    # required iff select_by == 'mean_var'
    hollow_mask_array=None,             # required iff select_by == 'mean_var'
    val_slices=     np.array([0], dtype=np.int32),
    reference_list=None,                # required iff select_by == 'mean_var'
    debug=False,
):
    """
    Unified trainer with selectable model criterion via h_params['select_by'] ∈ {'loss','mean_var'}.

    - 'loss': tracks best-by-loss, does NOT perform recon/variance window (fast).
    - 'mean_var': runs moving window over reconstructions and selects params at the
      center of the window with the lowest mean variance across val_slices.

    Returns:
        {
          'time_str': ...,
          'time_s': ...,
          'best': { 'it', 'params', 'time', ('loss' or 'mean_var'), 'waiting' },
          'log_queue': {...}  # lightweight if 'loss', full (incl. mean_var + slice IQMs) if 'mean_var'
        }
    """

    select_by = h_params.get('select_by', 'loss')
    if select_by not in ('loss', 'mean_var'):
        raise ValueError("h_params['select_by'] must be 'loss' or 'mean_var'")

    use_mv = (select_by == 'mean_var')

    # ----- Hyperparams
    epsilon      = 1e-4
    nIter        = h_params['iter'] + 1
    batch_size   = h_params['bs']
    metric_step  = h_params['metric_step']
    nframes      = h_params['NFRAMES']
    nspokes      = h_params['nspokes']

    if use_mv:
        # extra params needed only for moving-variance selection
        window_size  = h_params['window_size']
        HW_idx       = window_size // 2
        val_frames   = h_params['val_frames']
        # sanity checks for required call-time args
        if (recon_cine is None) or (hollow_mask_array is None) or (reference_list is None):
            raise ValueError("When select_by == 'mean_var', you must pass recon_cine, hollow_mask_array, reference_list")
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd so the center is well-defined.")
        
    
    # ----- Opt state
    start = time.time()
    fullparams_ = FrozenDict(fullparams_)
    params = fullparams_['params']
    extras = {k: v for k, v in fullparams_.items() if k != 'params'}
    opt_state = optimizer.init(params, extras)

    def step(params, X, Y, index_frames, key, niter):
        updated_key = jax.random.fold_in(key=key, data=niter)
        def aux_loss(p):
            full_p = {'params': p, **optimizer.extra}
            return loss(full_p, X, Y, index_frames, updated_key)
        (loss_value, updates_from_loss), grads = jax.value_and_grad(aux_loss, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        optimizer.extra = updates_from_loss
        return new_params, new_opt_state, loss_value

    # ----- Best tracker (only for selected criterion)
    if select_by == 'loss':
        best = {'it': None, 'params': None, 'time': None, 'loss': np.inf, 'waiting': 0}
    else:
        best = {'it': None, 'params': None, 'time': None, 'mean_var': np.inf, 'waiting': 0}

    # ----- Logs
    log_queue = {'it': [], 'loss': [], 'time': []}
    if use_mv:
        log_queue['mean_var'] = []
        for s_idx in val_slices:
            log_queue[s_idx] = {'var': [], 'ssim': [], 'psnr': [], 'ap': [], 'loss': [], 'it': [], 'time': []}
        windowed_queue = {'it': [], 'params': [], 'loss': [], 'time': [], 'nitems': 0}
        for s_idx in val_slices:
            windowed_queue[s_idx] = {'var': [], 'ssim': [], 'psnr': [], 'ap': [], 'time': []}

    # ----- Training loop
    for it in tqdm(range(nIter), desc='train iter', leave=True):
        key, key_batch, key_loss = jax.random.split(key, 3)

        # sample frames
        idx_batch_latent = jax.random.choice(key_batch, nframes, shape=(batch_size,), replace=False)
        times_batch      = idx_batch_latent / nframes
        index_frames     = np.int32(times_batch * nframes)

        # build nspokes-wise, per-slice batches
        Xb, Yb = [], []
        for item in range(len(X_list)):
            X = X_list[item]; Y = Y_list[item]
            idx_groups = [np.where(np.abs(X[:, 1] - t) < epsilon)[0] for t in times_batch]
            group_inds = sample_from_groups(idx_groups, nspokes, key)
            Xb.append(np.stack([X[g] for g in group_inds]))
            Yb.append(np.stack([Y[g] for g in group_inds]))
        X_batch_array = np.stack(Xb); Y_batch_array = np.stack(Yb)

        params, opt_state, train_loss_value = step(params, X_batch_array, Y_batch_array, index_frames, key_loss, it)

        # ---- lightweight logs always
        if it % metric_step == 0:
            tnow = time.time() - start
            log_queue['it'].append(it)
            log_queue['loss'].append(to_python_scalar(train_loss_value))
            log_queue['time'].append(tnow)

        # ---- selection by LOSS (fast path)
        if not use_mv:
            if train_loss_value < best['loss']:
                best.update({
                    'it': it,
                    'params': {'params': params, **optimizer.extra},
                    'time': time.time() - start,
                    'loss': to_python_scalar(train_loss_value),
                    'waiting': 0
                })
            else:
                best['waiting'] += 1
            continue  # next iteration

        # ---- selection by MOVING VARIANCE (windowed path)
        if it % metric_step == 0:
            current_params = {'params': params, **optimizer.extra}

            # push into window
            windowed_queue['time'].append(tnow)
            windowed_queue['it'].append(it)
            windowed_queue['params'].append(current_params)
            windowed_queue['loss'].append(to_python_scalar(train_loss_value))
            windowed_queue['nitems'] += 1

            # compute slice metrics for this point
            for s_idx in val_slices:
                recon_images = recon_cine(current_params, val_frames, s_idx, hollow_mask_array[s_idx], key_loss)
                windowed_queue[s_idx]['var'].append(compute_scalar_variance(recon_images))
                windowed_queue[s_idx]['ssim'].append(mean_ssim(recon_images, reference_list[s_idx]))
                windowed_queue[s_idx]['psnr'].append(mean_psnr(recon_images, reference_list[s_idx]))
                windowed_queue[s_idx]['ap'].append(mean_artifact_power(recon_images, reference_list[s_idx]))
                windowed_queue[s_idx]['time'].append(tnow)
                del recon_images
                gc.collect()

            # evaluate once window is full
            if windowed_queue['nitems'] >= window_size:
                # center sample is the representative snapshot
                c             = HW_idx
                center_it     = windowed_queue['it'][c]
                center_time   = windowed_queue['time'][c]
                center_params = windowed_queue['params'][c]

                # variance across window, then mean across slices
                var_list = []
                for s_idx in val_slices:
                    var_list.append(windowed_queue[s_idx]['var'][c])
                mean_var_now = float(np.mean(np.array(var_list)))
                log_queue['mean_var'].append(mean_var_now)

                # per-slice logs at the center
                for s_idx in val_slices:
                    # compute metrics
                    current_var = to_python_scalar( windowed_queue[s_idx]['var'][c] )
                    current_ssim = to_python_scalar(windowed_queue[s_idx]['ssim'][c])
                    current_psnr = to_python_scalar(windowed_queue[s_idx]['psnr'][c])
                    current_ap   = to_python_scalar(windowed_queue[s_idx]['ap'][c])

                    # assign metrics
                    log_queue[s_idx]['var'].append(   current_var   )
                    log_queue[s_idx]['ssim'].append(  current_ssim  )
                    log_queue[s_idx]['psnr'].append(  current_psnr  )
                    log_queue[s_idx]['ap'].append(    current_ap    )
                    log_queue[s_idx]['loss'].append(windowed_queue['loss'][c])
                    log_queue[s_idx]['it'].append(center_it)
                    log_queue[s_idx]['time'].append(center_time)


                # update best by mean variance
                if mean_var_now < best['mean_var']:
                    best.update({
                        'it': center_it,
                        'params': center_params,
                        'time': center_time,
                        'mean_var': mean_var_now,
                        'waiting': 0
                    })
                    for s_idx in val_slices:
                        best[s_idx] = {
                            'var': to_python_scalar( windowed_queue[s_idx]['var'][c] ),
                            'ssim': to_python_scalar( windowed_queue[s_idx]['ssim'][c] ),
                            'psnr': to_python_scalar( windowed_queue[s_idx]['psnr'][c] ),
                            'ap': to_python_scalar( windowed_queue[s_idx]['ap'][c] ),
                        }

                else:
                    best['waiting'] += 1

                # pop window head
                for k in ['it', 'params', 'loss', 'time']:
                    windowed_queue[k].pop(0)
                for s_idx in val_slices:
                    for k in ['var', 'ssim', 'psnr', 'ap', 'time']:
                        windowed_queue[s_idx][k].pop(0)
                windowed_queue['nitems'] -= 1

    # ----- wrap up
    elapsed = time.time() - start
    hours, minutes = int(elapsed // 3600), int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    # For compatibility with your previous fast logger, also mirror per-slice keys when not using variance
    if not use_mv:
        for s_idx in range(len(X_list)):
            log_queue[s_idx] = {
                'it':   log_queue['it'],
                'loss': log_queue['loss'],
                'time': log_queue['time'],
            }

    if debug:
        if select_by == 'loss':
            print(f"[End] Best loss {best['loss']:.6f} @ iter {best['it']}")
        else:
            print(f"[End] Best mean_var {best['mean_var']:.6f} @ iter {best['it']}")

    return {
        'time_str': f"{hours}h {minutes}m {seconds:.2f}s",
        'time_s': elapsed,
        'best': best,
        'log_queue': log_queue,
    }


