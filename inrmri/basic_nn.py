from inrmri.loggers import ExperimentLogger, LocalLogger
import jax.numpy as np
from jax import random, grad, jit, vmap
from jax.nn import relu, sigmoid
from tqdm import tqdm # mostrar barras de progreso 
from jax.tree_util import tree_leaves
from jax.flatten_util import ravel_pytree
from typing import Dict, Callable 

#----------------------------------------------------------------#
# Basic Multilayer Perceptron 
#----------------------------------------------------------------#

def init_params(layers, key):
  Ws = []
  bs = []
  for i in range(len(layers) - 1):
    std_glorot = np.sqrt(2/(layers[i] + layers[i + 1]))
    key, subkey = random.split(key)
    Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    bs.append(np.zeros(layers[i + 1]))
  return (Ws, bs)

@jit
def forward_pass(params, H):
  Ws, bs = params
  N_layers = len(Ws)
  for i in range(N_layers - 1):
    H = np.matmul(H, Ws[i]) + bs[i]
    H = relu(H)
  Y = np.matmul(H, Ws[-1]) + bs[-1]
  return Y

#----------------------------------------------------------------#
# Training 
#----------------------------------------------------------------#


# @partial(jit, static_argnums=(0,))
def step(loss, i, opt_state, X_batch, Y_batch, get_params, opt_update):
    params = get_params(opt_state)
    g = grad(loss)(params, X_batch, Y_batch)
    return opt_update(i, g, opt_state)


def train(loss, metrics, image_generators, train, test, optimizer, params, key, nIter = 10000, batch_size = 10, logger = None, **logger_kwargs):
    """
    loss: (params, X, Y)
    metrics: diccionario de la forma {"metric_name": metric_func}, donde 
      metric_func es una función de (params).
    image_generators: diccionario de la forma {"generator_name": generator_func}, 
      donde generator_func es una función de (params).
    logger: ExperimentLogger
    **logger_kwargs: argumentos extra para logger.init()
    """
    # best_loss = np.inf
    # best_param = get_params(opt_state)

    opt_init, opt_update, get_params = optimizer
    opt_state = opt_init(params)

    logger = logger or LocalLogger('./')
    logger.init(**logger_kwargs)
    logger.print_backend_info()

    X, Y = train
    testX, testY = test 

    for it in tqdm(range(nIter), desc='train iter', leave=True):
        key, subkey = random.split(key)
        idx_batch = random.choice(subkey, X.shape[0], shape = (batch_size,), replace = False)
        
        idx_test_batch = random.choice(random.split(key)[1], testX.shape[0], shape = (np.minimum(batch_size, testX.shape[0]),), replace = False)
        opt_state = step(loss, it, opt_state, X[idx_batch], Y[idx_batch], get_params, opt_update)
                  
        if it % 100 == 0:
            logger.set_step(it)

            params = get_params(opt_state)
            train_loss_value = loss(params, X[idx_batch], Y[idx_batch])
            logger.log_train_loss(train_loss_value)
            
            metric_msg = "metric: "
            for metric_name, metric_func in metrics.items(): 
              metric_value = metric_func(params)
              
              logger.log_metric(metric_name, metric_value)
              metric_msg += f"{metric_name} = {metric_value:.3f}, "
              

            test_loss_value = loss(params, testX[idx_test_batch], testY[idx_test_batch])
            logger.log_test_loss(test_loss_value)

            to_print = "it %i, train loss = %e, test loss = %e, %s" % (it, train_loss_value, test_loss_value, metric_msg)
            logger.log_info(to_print)

            if it % 1000 == 0: 
              for gen_name, gen_func in image_generators.items():
                logger.add_artifact(f'img{gen_name}-{it}', gen_func(params))

              logger.add_artifact(f'param-{it}', params)
    
    params = get_params(opt_state)

    for metric_name, metric_func in metrics.items(): 
      metric_value = metric_func(params)
      logger.log_summary(metric_name, metric_value)
    
    for gen_name, gen_func in image_generators.items():
      logger.log_summary('img' + gen_name, gen_func(params))

    logger.add_artifact('last_param', params)
    logger.finish()      

def simple_loss(measures_net_X, measures_Y):
  # return np.mean((np.abs(measures_net_X) - np.abs(measures_Y))**2)
  return np.mean(np.abs(measures_net_X - measures_Y)**2)

def weighted_loss(X,Y,W): 
  return np.mean((np.abs(X - Y) * W)**2)

@jit
def mse(X,Y): 
  return np.mean(np.power(np.abs(X - Y), 2))

@jit
def psnr(X,Y): 
  return -10 * np.log10(mse(X,Y))

import optax 
import jax 

def simple_train(loss, X, Y, params, optimizer, key, batch_size = 5000, nIter = 5000):
    opt_state = optimizer.init(params)

    #@partial(jax.jit, static_argnums=(3,)) # yo diria que 0 
    def step(loss, params, opt_state, batchX, batchY):
        loss_value, grads = jax.value_and_grad(loss)(params, batchX, batchY)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    train_loss = []
    best_loss = np.inf
    iterations = []
    for it in tqdm(range(nIter), desc='train iter', leave=True):
        key, subkey = random.split(key)
        idx_batch = random.choice(subkey, X.shape[0], shape = (batch_size,))#, replace = False)# 
        params, opt_state, train_loss_value = step(loss, params, opt_state, X[idx_batch], Y[idx_batch])
        if train_loss_value < best_loss: 
            best_loss = train_loss_value
            best_param = params           
        if it % 100 == 0:
            train_loss.append(train_loss_value)     
            to_print = " it %i, train loss = %e" % (it, train_loss_value)
            print(to_print)

            iterations.append(it)
    results = {
      'last_param': params, 
      'best_param': best_param,
      'train_loss': train_loss,
      'best_loss': best_loss,
      'iterations': iterations
    }
    return results 

def count_params(params, magnitude_term =1e6): 
    nparams = sum([arr.size for arr in tree_leaves(params)])
    return nparams / magnitude_term

def mean_abs(tree):
    return np.mean(np.abs(ravel_pytree(tree)[0]))

class LossEvaluator:
    """
    
    ## Ejemplo de uso 

    Supongamos que tenemos previamente definidas las funciones de pérdida 
    `loss_registration(params, im1, im2)` y `loss_residual(params, key)`.
    Además hemos entrenado una red y hemos obtenido una evolución de parámetros,
    guardados en `results['param_history']`, bajo los nombres 'param-0', 'param-1', ....
    Podemos usar el evaluador de la siguiente forma: 

    ```
    evaluator = LossEvaluator({
        'registration': loss_registration,
        'residual': loss_residual
        })

    k_params = [results['param_history'][f'param-{it}'] for it in range(3)]
    arg_dic = {
            'registration': (im1, im2),
            'residual': (key,)
        }

    l_vals = evaluator.evaluate_losses(k_params, arg_dic)
    g_vals = evaluator.grad_evo(k_params, arg_dic)
    ```
    """

    def __init__(self, loss_dict:Dict[str, Callable]):
        """
        loss_dict: 
            {'loss_name': loss_func(params, *args)}
        """
        self.loss_dict = loss_dict

    def evaluate_losses(self, param_list, args_losses):
        loss_vals = {
            loss_name: [loss_func(params, *args_losses[loss_name]) for params in param_list] for loss_name, loss_func in self.loss_dict.items()
        }
        return loss_vals

    def grad_evo(self, param_list, args_losses): 
        evo_loss = {
            loss_name: [mean_abs(jax.value_and_grad(loss_func)(params, *args_losses[loss_name])[1]) for params in param_list] for loss_name, loss_func in self.loss_dict.items()
        }
        return evo_loss