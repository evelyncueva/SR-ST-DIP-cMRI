from abc import ABC, abstractmethod, abstractproperty
import jax.numpy as np
from jax import random, grad, jit, vmap

from inrmri.basic_nn import init_params, forward_pass
from inrmri.utils import meshgrid_from_subdiv_autolims
from inrmri.utils import meshgrid_from_subdiv, to_complex
from functools import partial
from dataclasses import dataclass 
from argparse import ArgumentParser
from jax import Array as PRNGKeyArray

def concat_last_axis(arraylist): 
  return np.concatenate(arraylist, axis = -1)


def time_mapping(t, m):
  """
  - `m`: si m = 3, entonces usa los periodos
  2pi, 4pi, 6pi, es decir usa los periodos (2npi), para n = 1,...,m
  """
  periods = np.arange(m)+1
  return concat_last_axis([np.sin(2 * np.pi * periods * t), np.cos(2 * np.pi * periods * t)])

def B_matriz(shape, key): 
  return random.normal(key, shape=shape)

@partial(jit, static_argnums=(2,))
def positional_encoding(x, sigma, m): 
  x_proj = 2 * np.pi * x * sigma ** (np.arange(m)/m)
  return concat_last_axis([np.sin(x_proj), np.cos(x_proj)])

def input_mapping(x, B): 
  """
  Aplica fourier features al vector de entrada x con matriz B
  """
  x_proj = (2.*np.pi*x) @ B.T
  return concat_last_axis([np.sin(x_proj), np.cos(x_proj)])

def time_even_periods_mapping(t, m):
  """
  - `m`: si m = 3, entonces usa los periodos
  2pi, 4pi, 6pi, es decir usa los periodos (2npi), para n = 1,...,m
  """
  periods = np.arange(m)+1
  return concat_last_axis([np.sin(2 * np.pi * periods * t), np.cos(2 * np.pi * periods * t)])

def process_coordinates_and_even_periods_time(xt, B, m):
  """
  Preprocesamiento, aplica fourier features en espacio y un mapeo de varios periodos
  en tiempo. Mapea xt[2] (tiempo) en [0,1].
  - xt: vector de largo 3, de la forma (x,y,t)
  - B: matriz de fourier features de 2 columnas 
  - m: cuántos periodos. el caso m = 1 corresponde a mapear el tiempo a un circulo.
  """
  x = xt[:-1]
  t = xt[-1]
  return concat_last_axis([input_mapping(x,B), time_even_periods_mapping(t, m)])

def spatio_temporal_mapping(xt, B):
  """
  Aplica Fourier Features en espacio, multiplicando ademas por informacion temporal.
  - xt: vector de largo 3, de la forma (x,y,t)
  - B: matriz de fourier faetures espacial, de 2 columnas 
  """
  x = xt[:-1]
  t = xt[-1]
  x_proj = (2.*np.pi*x) @ B.T
  t_proj = 2 * np.pi * t 
  return concat_last_axis([np.sin(x_proj)*np.sin(t_proj),
                         np.sin(x_proj)*np.cos(t_proj),
                         np.cos(x_proj)*np.sin(t_proj), 
                         np.cos(x_proj)*np.cos(t_proj)])

def static_mixed_mapping(xt, B_static, B_mixed): 
  """
  Aplica Fourier Features en espacio, además de hacer un spatio_temporal_mapping,
  concatenando todo en un vector.
  - xt: vector de largo 3, de la forma (x,y,t)
  - `B_static`: matriz de fourier features espacial, de 2 columnas 
  - `B_mixed`: matriz de fourier features espacio-temporal, de 2 columnas 
  """
  ff_static = input_mapping(xt[:2], B_static)
  ff_mixed = spatio_temporal_mapping(xt, B_mixed)

  return concat_last_axis([ff_static, ff_mixed])

# def FFnet_forward_pass(params, x, B):
#   """
#   Red MLP con entrada preprocesada con fourier features 
#   """
#   return forward_pass(params, input_mapping(x, B))

# def image_prediction(params, grid, B): 
#   return FFnet_forward_pass(params, grid, B)[:,:,0]

# squeezednet = lambda p, x, B: squeeze(FFnet_forward_pass(p, x, B))
# 
# DxFFnet = np.vectorize(grad(squeezednet, 1), signature='(2),(n,2)->(2)', excluded = [0])

#------------------------------------------------------------------------------# 
# FF en espacio y tiempo periódica 
#------------------------------------------------------------------------------# 

@partial(np.vectorize, signature='(3),(n,2)->(p)')
def process_coordinates(xt, B):
  """
  Preprocesamiento, aplica fourier features en espacio y un mapping de periodo 1
  en tiempo. Mapea xt[2] (tiempo) en [0,1] al un circulo.
  - xt: vector de largo 3, de la forma (x,y,t)
  - B: matriz de fourier features de 2 columnas 
  """
  x = xt[:-1]
  t = xt[-1]
  return np.concatenate([input_mapping(x,B), np.array([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])], axis = -1)

# @partial(np.vectorize, signature='(3),(n,2),(),()->(p)')
def process_coordinates_and_PEtime(xt, B, sigmaPE, mPE):
  """
  Preprocesamiento, aplica fourier features en espacio y positional encoding en 
  tiempo. Mapea xt[2] (tiempo) en [0,1].
  - xt: vector de largo 3, de la forma (x,y,t)
  - B: matriz de fourier features de 2 columnas 
  """
  x = xt[:-1]
  t = xt[-1]
  return np.concatenate([input_mapping(x,B), positional_encoding(t, sigmaPE, mPE)], axis = -1)


# @partial(np.vectorize, signature='(3),(n,2)->()', excluded = [0])
# def FF_periodtime_net_forward_pass(params, xt, B):
#   """
#   Red MLP con entrada preprocesada con fourier features 
#   """
#   return squeeze(forward_pass(params, process_coordinates(xt, B)))

# Dxnet = np.vectorize(grad(FF_periodtime_net_forward_pass, 1), signature='(3),(n,2)->(3)', excluded = [0])

@partial(np.vectorize, signature='(2),()->(3)')
def concat(X, t): 
  return np.r_[X, t]

# def image_xt_prediction(params, grid_X, t, B): 
#   return FF_periodtime_net_forward_pass(params, concat(grid_X, t), B)

#------------------------------------------------------------------------------# 
# Otros 
#------------------------------------------------------------------------------# 

# def FF_periodtime_net_forward_pass_not_sqeezed(params, xt, B):
#   """
#   Red MLP con entrada preprocesada con fourier features 
#   """
#   return forward_pass(params, process_coordinates(xt, B))

# def complex_img_xt_prediction(params, grid_X, t, B):
#   return to_complex(FF_periodtime_net_forward_pass_not_sqeezed(params, concat(grid_X,t), B))[:,:,0]

#------------------------------------------------------------------------------# 
# FF en espacio y FF tiempo indep 
#------------------------------------------------------------------------------# 

@partial(np.vectorize, signature='(3),(n,2),(m,1)->(p)')
def process_coordinates_and_time(xt, Bx, Bt):
  """
  Preprocesamiento, aplica fourier features en espacio y en tiempo de manera
  independiente, tiempo no periódico.
  - xt: vector de largo 3, de la forma (x,y,t)
  - Bx, Bt: matrices de fourier features de 2 columnas 
  """
  x = xt[:-1]
  t = xt[-1:]
  return np.concatenate([input_mapping(x,Bx), input_mapping(t,Bt)], axis = -1)

# def FFx_FFt_net_forward_pass(params, xt, Bx, Bt):
#   """
#   Red MLP con entrada preprocesada con fourier features 
#   """
#   return squeeze(forward_pass(params, process_coordinates_and_time(xt, Bx, Bt)))

# def image_FFx_FFt_prediction(params, grid_X, t, Bx, Bt): 
#   return FFx_FFt_net_forward_pass(params, concat(grid_X, t), Bx, Bt)
#------------------------------------------------------------------------------# 
# FF en espacio y FF en tiempo periódico 
#------------------------------------------------------------------------------# 

@partial(np.vectorize, signature='(3),(n,2),(m,2)->(p)')
def process_coordinates_and_periodtime(xt, Bx, Bt):
  """
  Preprocesamiento, aplica fourier features en espacio y en tiempo de manera
  independientes.
  - xt: vector de largo 3, de la forma (x,y,t)
  - Bx, Bt: matrices de fourier features de 2 columnas 
  """
  x = xt[:-1]
  t = xt[-1]
  periodt = np.array([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])
  return np.concatenate([input_mapping(x,Bx), input_mapping(periodt,Bt)], axis = -1)


# def FFx_FFpt_net_forward_pass(params, xt, Bx, Bt):
#   """
#   Red MLP con entrada preprocesada con fourier features 
#   """
#   return squeeze(forward_pass(params, process_coordinates_and_periodtime(xt, Bx, Bt)))

# def image_FFx_FFpt_prediction(params, grid_X, t, Bx, Bt): 
#   return FFx_FFpt_net_forward_pass(params, concat(grid_X, t), Bx, Bt)


@dataclass 
class FFnetParser(): 
    shortcut: str 
    nargs: int 
    helpstr: str 


class FFnet_2Dspace_time(ABC):
  """
    Redes con Fourier Features para aprender vídeos    

  Interfaz para redes con entrada x,y,t, entrenadas para aprender vídeos de la 
  forma (px,py,t). 

  Las redes consisten de un `FFBox`, una caja negra que transforma las
  coordenadas `(x,y,t)` mediante alguna técnica de Fourier Features (positional
  enconding, etc); un `MLP`, un multilayer perceptron que toma las coordenadas 
  procesadas por el `FFBox` y entrega una salida real o compleja.
  """
  
  def init_params(self, inner_layers, key): 
    """
      Parámetros para el `MLP`.
    ## Argumentos 
    - `inner_layers`: list(int)
    - `key`: seed para cosas random 
      Solo los layers interiores, no los de salida y entrada.
    """ 
    output = 2 if self.is_complex() else 1
    layers = [self.ff_outsize()] + inner_layers + [output]
    return init_params(layers, key)

  @abstractmethod
  def ff_type(self): # -> str 
    """
      Nombre de la `FFBox` usada.
    """
    pass

  @abstractmethod
  def ff_outsize(self): # -> int 
    """
      Tamaño de salida de `FFBox`.
    
    Corresponde a `self.useFFBox(x).shape[-1]`, y es el tamaño de la entrada al 
    `MLP`.
    """
    pass

  @abstractmethod
  def useFFBox(self, x): 
    """
      Transforma coordenadas de entrada `(x,y,t)` con Fourier Features.     
    
    ## Argumentos 
    - `x`: array que cumpla `x.shape[-1] == 3`
      La última dimensión debe corresponder a las coordendas `x,y,t`
    ## Returns 
    - `gamma`: array, `gamma.shape[-1] == ff_outsize`
    
    """
    pass 
  
  def useMLP(self, params, gamma):
    """
      Aplica un `MLP` a las coordenadas transformadas por `FFBox`. 
    """
    return forward_pass(params, gamma)
    

  @abstractmethod
  def is_complex(self): # -> bool 
    """
      La imagen generada es real o compleja.

    Si es real, el `MLP` tendrá salida con `shape[-1] = 1`; y si es complejo,
    salida con `shape[-1] = 2`. Para cada clase con `is_complex` es `True`, es 
    necesario implementar un método `to_complex` que transforme la salida a un 
    valor complejo.
    """
    pass

  @abstractmethod
  def to_complex(self, out): 
    """
      Transforma la salida en un valor complejo
    
    Solo es necesario si `is_complex = `True. Si es `False` debería retornar `out`
    sin modificaciones.
    """
    pass 
  
  def eval_xt(self, params, xt): 
    """
      Pasa las coordenadas de entrada `(x,y,t)` por un MLP, luego de haber aplicado FF.
    
    ## Argumentos 
    - `xt`: array que cumpla `x.shape[-1] == 3`
      La última dimensión debe corresponder a las coordendas `x,y,t`
    ## Returns 
    - `out`: array, `out.shape[-1] == 1`
      Su tipo es `complex` si `is_complex == True`, si no es `float`. 
    """
    val = self.useMLP(params, self.useFFBox(xt))
    #print("NN pre complex shape: ",val.shape )
    out = self.to_complex(val)
    return out 

  def eval_x_and_t(self, params, x, t): 
    """
      Pasa las coordenadas de entrada `(x,y),(t)` por un MLP, luego de haber aplicado FF.
    
    ## Argumentos 
    - `x`: array que cumpla `x.shape[-1] == 2`
      La última dimensión debe corresponder a las coordendas `x,y`
    - `t`: tiempo (float)
    ## Returns 
    - `out`: array, `out.shape[-1] == 1`
      Su tipo es `complex` si `is_complex == True`, si no es `float`. 
    """
    out = self.eval_xt(params, concat(x,t))
    return out 

  @abstractmethod
  def image_prediction_at_timeframe(self, params, t):
    """
      Genera una imagen en el tframe t.
    """
    pass

  @abstractmethod
  def image_prediction_full(self,params):
    """
      Genera un video
    
    ## Return 
    - `out`: array, shape (px, py, pt)
    """
    pass 

  @classmethod
  @abstractmethod
  def get_parser() -> FFnetParser: 
    """
      Genera un objeto con info suficiente para hacer un parser adecuado para la clase.
    """
    pass 

  def gradients_xt_per_output_component(self, params, xt, component): 
    """
    - `component`: numero en 0, ..., out.shape[-1] - 1
    """
    # squeezednet = lambda p, x, B: squeeze(FFnet_forward_pass(p, x, B))
    neteval = lambda xt: self.useMLP(params, self.useFFBox(xt))[...,component] # esto da un arreglo con demasiadas cosas en la ultima dim 
    # necesito las derivadas para cada una 
    DxFFnet = np.vectorize(grad(neteval), signature='(3)->(3)')
    return DxFFnet(xt)
  
  def gradients_xt(self, params, xt): 
    total_components = 2 if self.is_complex() else 1 
    return np.stack([self.gradients_xt_per_output_component(params, xt, component) for component in range(total_components)])

  @classmethod
  @abstractmethod
  def calculate_mapsize(arglist)-> int: 
    pass 

def find_correct_class(subclass_map, adict): 
    for shortcutkey, theclass in subclass_map.items(): 
        try: 
            if adict[shortcutkey] is not None: 
                return shortcutkey, theclass 
        except KeyError: 
            print(f"config no tiene {shortcutkey}")


def add_FFgroup_to_parser(parser:ArgumentParser): 
    """

    Agregar un grupo mutuamente excluyente y requerido de todas las subclases 
    de `FFnet_2Dspace_time`. Pensado para usarse en conjunto con la función
    `create_FFnet_from_parser`. 

    ## Argumentos 
    - `parser: ArgumentParser` de `argparse`

    ## Ejemplo 

    ```
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description="Train from data stored in npy file", formatter_class=ArgumentDefaultsHelpFormatter)

    add_FFgroup_to_parser(parser) 

    args = parser.parse_args("--evenpt 50 2".split())
    ```
    de esta forma 
    ```
    >>> vars(args)
    {'evenpt': ['50', '2'],
    'mixed': None,
    'static_mixed': None,
    'frac_static_mixed': None}
    ```
    """
    group = parser.add_mutually_exclusive_group(required=True)
    for subclass in FFnet_2Dspace_time.__subclasses__(): 
        parserinfo = subclass.get_parser()
        group.add_argument(f"--{parserinfo.shortcut}", help = parserinfo.helpstr, nargs = parserinfo.nargs)

def get_FF_subclass_map(): 
  return  {subclass.get_parser().shortcut: subclass for subclass in FFnet_2Dspace_time.__subclasses__()}

def create_FFnet_from_parser(adict, im): 
    """
    ## Argumentos 
    - `adict`: diccionario cuyas llaves incluyen el atributo 
      `shortcut` del `FFnetParser` de cada una de las subclases
      de `FFnet_2Dspace_time`. Todas estas llaves salvo la de la
      clase elegida deben estar asociadas a valores `None`.
      La llave de la clase elegida debe tener una lista del largo
      dado por el atributo `nargs` del `FFnetParser` de la clase
      elegida. Este es el tipo de diccionario que se genera por un 
      argparser de python usando exclusión mutua y _nargs_. Puede 
      obtenerse mediante la función `add_FFgroup_to_parser`. 
      El diccionario debe también incluir los argumentos comunes 
      a todos los métodos (`'sigma'`, `'radon_seed'`). 
    - `im`: array, shape (px, py, pt). Imagen usada por las FFnet
      para definir el tamaño de la grilla y si usar 2 canales de
      salida (imagen compleja) o 1 (imagen real).

    ## Ejemplo 

    Podemos obtener un diccionario adecuado usando `argparse` y la función 
    `add_FFgroup_to_parser`.

    ```
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description="Train from data stored in npy file", formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--sigma", type = float, default=1.5)
    parser.add_argument("--radon_seed", type = int, default=0)
    
    add_FFgroup_to_parser(parser) 

    args = vars(parser.parse_args("--frac_static_mixed 0.6 50".split()))
    ```

    De esta forma obtenemos el diccionario `args` que cumple las condiciones requeridas.

    ```
    >>> args
    {'sigma': 1.5,
    'radon_seed': 0,
    'evenpt': None,
    'mixed': None,
    'static_mixed': None,
    'frac_static_mixed': ['0.6', '50']}
    ```
    
    Finalmente, podemos usar la factory `create_FFnet_from_parser` para obtener una `FF_fraction_static_mixed_net`.
    
    ```
    >>> FFnet = create_FFnet_from_parser(args, np.zeros((100,100,20)) + 0j)
    >>> FFnet 
    <inrmri.fourier_features.FF_fraction_static_mixed_net at 0x000000000000>
    ```
    """
    subclass_map = get_FF_subclass_map()
    shortcut, subclass = find_correct_class(subclass_map, adict)
    paramslist = adict[shortcut]
    instance = super(FFnet_2Dspace_time, subclass).__new__(subclass)
    instance.__init__(*paramslist, sigma = adict["sigma"], gtimg = im, key = random.PRNGKey(adict["radon_seed"])) 
    return instance

class FFx_evenpt_net(FFnet_2Dspace_time):
  """
    Red clásica, con FourierFeatures en espacio y tiempo periódico. 
  """

  def __init__(self, mapping_size, m, sigma, gtimg, key): 

    mapping_size = int(mapping_size)
    self._m = int(m)


    self._B = sigma * random.normal(key, shape=(mapping_size,2))
    self._gtimg = gtimg / np.max(np.abs(gtimg))

    self._gridX = meshgrid_from_subdiv(gtimg.shape[:2], (-1,1))
    self._gridt = np.linspace(0,1,gtimg.shape[2], endpoint = False)

    # conociendo gtimg -> sabes si es real o complejo 
    self._is_complex = np.iscomplexobj(gtimg)
    # la clase sabe sus parametros de salida y entrada     
    self._inputsize = self.useFFBox(np.zeros(3)).shape[0]

  def ff_type(self):
    return "FFx_evenpt_net"

  def ff_outsize(self): 
    return self._inputsize

  def useFFBox(self, x): 
    partial_funct = partial(process_coordinates_and_even_periods_time, B = self._B, m = self._m)
    return np.vectorize(partial_funct, signature='(3)->(p)')(x)

  def is_complex(self):
    return self._is_complex
  
  def to_complex(self, out): 
    return to_complex(out)

  def image_prediction_at_timeframe(self,params, t): 
    return np.squeeze(self.eval_x_and_t(params, self._gridX, t), axis = -1)
  
  def image_prediction_full(self,params): 
    out = vmap(lambda t: self.image_prediction_at_timeframe(params, t), out_axes = -1)(self._gridt)
    return out
  
  def get_parser():
        return FFnetParser('evenpt', 2, "args: M: mapping size, m: time periods")   

  def calculate_mapsize(arglist): 
    return int(arglist[0])

class FFx_mixedff_space_time_net(FFnet_2Dspace_time):
  """
    Red clásica, con FourierFeatures en espacio y tiempo periódico. 
  """

  def __init__(self, mapping_size, sigma, gtimg, key): 
    mapping_size = int(mapping_size)

    self._B = sigma * random.normal(key, shape=(mapping_size,2))
    self._gtimg = gtimg / np.max(np.abs(gtimg))

    self._gridX = meshgrid_from_subdiv(gtimg.shape[:2], (-1,1))
    self._gridt = np.linspace(0,1,gtimg.shape[2], endpoint = False)

    # conociendo gtimg -> sabes si es real o complejo 
    self._is_complex = np.iscomplexobj(gtimg)
    # la clase sabe sus parametros de salida y entrada     
    self._inputsize = self.useFFBox(np.zeros(3)).shape[0]

  def ff_type(self):
    return "FFx_mixedff_space_time_net"

  def ff_outsize(self): 
    return self._inputsize

  def useFFBox(self, x): 
    partial_funct = partial(spatio_temporal_mapping, B = self._B)
    return np.vectorize(partial_funct, signature='(3)->(p)')(x)

  def is_complex(self):
    return self._is_complex
  
  def to_complex(self, out): 
    return to_complex(out)

  def image_prediction_at_timeframe(self,params, t): 
    return np.squeeze(self.eval_x_and_t(params, self._gridX, t), axis = -1)
  
  def image_prediction_full(self,params): 
    out = vmap(lambda t: self.image_prediction_at_timeframe(params, t), out_axes = -1)(self._gridt)
    return out
    
  def get_parser():
    return FFnetParser('mixed', 1, "args: M: mapping size")   

  def calculate_mapsize(arglist): 
    return int(arglist[0])

class FF_static_mixed_net(FFnet_2Dspace_time):
  """
    Red clásica, con FourierFeatures en espacio y tiempo periódico. 
  """

  def __init__(self, mapping_size, sigma, gtimg, key): 

    mapping_size = int(mapping_size)
    self._B = sigma * B_matriz((mapping_size, 2), key)
    self._gtimg = gtimg / np.max(np.abs(gtimg))

    self._gridX = meshgrid_from_subdiv(gtimg.shape[:2], (-1,1))
    self._gridt = np.linspace(0,1,gtimg.shape[2], endpoint = False)

    # conociendo gtimg -> sabes si es real o complejo 
    self._is_complex = np.iscomplexobj(gtimg)
    # la clase sabe sus parametros de salida y entrada     
    self._inputsize = self.useFFBox(np.zeros(3)).shape[0]

  def ff_type(self):
    return "FF_static_mixed_net"

  def ff_outsize(self): 
    return self._inputsize

  def useFFBox(self, x): 
    partial_funct = partial(static_mixed_mapping, B_static = self._B, B_mixed = self._B)
    return np.vectorize(partial_funct, signature='(3)->(p)')(x)

  def is_complex(self):
    return self._is_complex
  
  def to_complex(self, out): 
    if self.is_complex():
      return to_complex(out)
    else: 
      return out 

  def image_prediction_at_timeframe(self,params, t): 
    return np.squeeze(self.eval_x_and_t(params, self._gridX, t), axis = -1)
  
  def image_prediction_full(self,params): 
    out = vmap(lambda t: self.image_prediction_at_timeframe(params, t), out_axes = -1)(self._gridt)
    return out
  
  def get_parser():
      return FFnetParser('static_mixed', 1, "args: M: mapping size")
  
  def calculate_mapsize(arglist): 
    return int(arglist[0])

class FF_fraction_static_mixed_net(FFnet_2Dspace_time):
  """
    Red clásica, con FourierFeatures en espacio y tiempo periódico. 
  """

  def __init__(self, static_fraction:float, desired_ffvector_len:int, sigma:float, imshape:tuple, complex_output:bool, key:PRNGKeyArray): 
    """
      INR con Fourier Features de tipo STiFF

    Los fourier features de tipo STiFF se describen en [este paper](https://arxiv.org/pdf/2307.14363.pdf), eq (4).
    
    ## Argumentos 
    - `static_fraction:float`. valor en [0,1], que fracción del vector de fourier features es estático.
    - `desired_ffvector_len:int`. largo deseado del vector STiFF de fourier features (L)
    - `sigma:float`. desviación estándar usada para generar las entradas de las matrices de direcciones.
    - `imshape:tuple`. Tupĺa con el tamaño de la imagen a reconstruir, en formato `(px, py, nframes)`.
    - `complex_output:bool`. Si `True`, la red tiene 2 canales de salida, que son usados como parte real e imaginaria. Si `False`, la salida tiene 1 canal.
    - `key:PRNGKeyArray`. Para ra reproducibilidad de las matrices.

    ## Comentarios 

    - No es seguro que el largo del vector STiFF generado sea de tamaño `desired_ffvector_len`.
    """
    static_fraction = float(static_fraction)
    desired_ffvector_len = int(desired_ffvector_len)
    M_static = round(static_fraction * desired_ffvector_len / 2)
    M_mixed = round((1-static_fraction) * desired_ffvector_len / 4)

    key_static, key_mixed = random.split(key)
    self._B_static = sigma * B_matriz((M_static, 2), key_static)
    self._B_mixed = sigma * B_matriz((M_mixed, 2), key_mixed)

    self._gridX = meshgrid_from_subdiv(imshape[:2], (-1,1))
    self._gridt = np.linspace(0,1,imshape[2], endpoint = False)

    # conociendo gtimg -> sabes si es real o complejo 
    self._is_complex = complex_output
    # la clase sabe sus parametros de salida y entrada     
    self._inputsize = self.useFFBox(np.zeros(3)).shape[0]

  def ff_type(self):
    return "FF_fraction_static_mixed_net"

  def ff_outsize(self): 
    return self._inputsize

  def useFFBox(self, x): 
    partial_funct = partial(static_mixed_mapping, B_static = self._B_static, B_mixed = self._B_mixed)
    return np.vectorize(partial_funct, signature='(3)->(p)')(x)

  def is_complex(self):
    return self._is_complex
  
  def to_complex(self, out): 
    if self.is_complex():
      return to_complex(out)
    else: 
      return out 

  def image_prediction_at_timeframe(self,params, t): 
    return np.squeeze(self.eval_x_and_t(params, self._gridX, t), axis = -1)
  
  def image_prediction_full(self,params): 
    out = vmap(lambda t: self.image_prediction_at_timeframe(params, t), out_axes = -1)(self._gridt)
    return out
    
  def get_parser():
    return FFnetParser('frac_static_mixed', 2, "args: static_fraction (float), desired_ffvector_len (int)")   

  def calculate_mapsize(arglist): 
    return int(arglist[1])


class FFxt_net(FFnet_2Dspace_time):
  """
    Red clásica, trata espacio y tiempo equitativamente pero cambia el dominio espacial 
  """

  def __init__(self, mapping_size, sigma, gtimg, key): 

    mapping_size = int(mapping_size)

    self._B = sigma * random.normal(key, shape=(mapping_size,3))
    self._gtimg = gtimg / np.max(np.abs(gtimg))

    self._gridXt = meshgrid_from_subdiv_autolims(gtimg.shape)
    self._gridX = meshgrid_from_subdiv_autolims(gtimg.shape[:2])
    self._gridt = np.linspace(-1,1,gtimg.shape[2], endpoint = False)

    # conociendo gtimg -> sabes si es real o complejo 
    self._is_complex = np.iscomplexobj(gtimg)
    # la clase sabe sus parametros de salida y entrada     
    self._inputsize = self.useFFBox(np.zeros(3)).shape[0]

  def ff_type(self):
    return "FFxt_net"

  def ff_outsize(self): 
    return self._inputsize

  def useFFBox(self, x): 
    partial_funct = partial(input_mapping, B=self._B)
    return np.vectorize(partial_funct, signature='(3)->(p)')(x)

  def is_complex(self):
    return self._is_complex
  
  def to_complex(self, out): 
    return to_complex(out)

  def image_prediction_at_timeframe(self,params, t): 
    return np.squeeze(self.eval_x_and_t(params, self._gridX, t), axis = -1)
  
  def image_prediction_full(self,params): 
    out = vmap(lambda t: self.image_prediction_at_timeframe(params, t), out_axes = -1)(self._gridt)
    return out
  
  def get_parser():
        return FFnetParser('ff3d', 1, "args: M: mapping size")   

  def calculate_mapsize(arglist): 
    return int(arglist[0])


# TODO: hay mucho obverlap entre esta clase y FFnet_2Dspace_time, algún día debería comprimirlas
class CoordinateEncodedNetwork(ABC):
  """
    Redes con Fourier Features para aprender vídeos

  Interfaz para redes con entrada x,y,t, entrenadas para aprender vídeos de la
  forma (px,py,t).

  Las redes consisten de un FFBox, una caja negra que transforma las
  coordenadas (x,y,t) mediante alguna técnica de Fourier Features (positional
  enconding, etc); un MLP, un multilayer perceptron que toma las coordenadas
  procesadas por el FFBox y entrega una salida real o compleja.
  """

  def init_params(self, key, inner_layers):
    """
      Parámetros para el MLP.
    ## Argumentos
    - inner_layers: list(int)
    - key: seed para cosas random
      Solo los layers interiores, no los de salida y entrada.
    """
    layers = [self.ff_vector_size()] + inner_layers + [self.output_dimension()]
    print(layers)
    return init_params(layers, key)

  @abstractmethod
  def input_dimension(self)->int: # -> int
    """
      Tamaño de entrada de FFBox.
    """
    pass

  @abstractmethod
  def output_dimension(self)->int: # -> int
    """
      Número de canales de salida de la red.
    """
    pass

  def ff_vector_size(self)->int: # -> int
    """
      Tamaño de salida de FFBox.
    """
    return self.useFFBox(np.zeros(self.input_dimension())).shape[-1]
  
  @abstractmethod
  def useFFBox(self, x):
    """
      Transforma coordenadas de entrada (x,y,t) con Fourier Features.

    ## Argumentos
    - x: array que cumpla x.shape[-1] == 3
      La última dimensión debe corresponder a las coordendas x,y,t
    ## Returns
    - gamma: array, gamma.shape[-1] == ff_outsize

    """
    pass

  def useMLP(self, params, gamma):
    """
      Aplica un MLP a las coordenadas transformadas por FFBox.
    """
    return forward_pass(params, gamma)

  def eval_coordinates(self, params, X):
    """
      Pasa las coordenadas de entrada (x,y,t) por un MLP, luego de haber aplicado FF.

    ## Argumentos
    - xt: array que cumpla x.shape[-1] == 3
      La última dimensión debe corresponder a las coordendas x,y,t
    ## Returns
    - out: array, out.shape[-1] == 1
      Su tipo es complex si is_complex == True, si no es float.
    """
    val = self.useMLP(params, self.useFFBox(X))
    return val 
  

def B_matrix(key, input_dims:int, sigma:float, mapping_size:int):
  """
    Fourier Features matriz
  - key: generada con random.PNRGKey(0)
  - input_dims: int, número de coordenadas de entrada que recibe
  - sigma: float > 0
  - mapping_size: number of directions
  """
  B = sigma * random.normal(key, shape=(mapping_size,input_dims))
  return B

class BasicSpatialFourierFeaturesNet(CoordinateEncodedNetwork):
  """
    Red clásica, con Fourier Features en espacio.

  - input_size: int numero de coordenadas espaciales de entrada (2 para red 2D, etc)
  - input_size, mapping_size, sigma: parametros de B_matriz.
  - output_size: int, numero de canales de salida de la red. Normalmente 1 para
    imagenes en escala de grises, 2 para complejas y 3 para imágenes a color.
  """

  def __init__(self, input_size:int, mapping_size:int, sigma:float, output_size:int, key_B):

    self._B = B_matrix(key_B, input_size, sigma, mapping_size)
    # assert len(imshape) == input_size, f"imshape debe ser una n-tupla o lista con tantos elementos como dimensiones de entrada de la red (input_size={input_size})"

    # self._gridX = meshgrid_from_subdiv(imshape, (-1,1))

    # la clase sabe sus parametros de salida y entrada
    self.input_size = input_size
    self.output_size = output_size
    self._ff_vector_size = 2 * self._B.shape[0]

  def input_dimension(self):
    return self.input_size

  def output_dimension(self):
    return self.output_size

  def useFFBox(self, x):
    partial_funct = partial(input_mapping, B = self._B)
    return np.vectorize(partial_funct, signature=f'({self.input_size})->({self._ff_vector_size})')(x)
