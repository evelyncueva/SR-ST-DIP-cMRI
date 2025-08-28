"""
Loggers para guardar resultados de entrenamientos.
Están pensados para ser usados con la función `train`
de `inr.basic_nn`
"""

from abc import ABC, abstractmethod

import random as pyrandom
from wonderwords import RandomWord
from pathlib import Path
import logging 
import pickle # guardar resultados 
import os 

class ExperimentLogger(ABC): 
  """
    Una interfaz común para varios loggers.
  
  Está pensada para ser usado durante el entrenamiento, de la forma: 
  ```python 
  logger = anExperimentLogger()
  logger.init(project = projectName, group = groupName, config = configdict)
  for it in range(nIters):
    if it%100 == 0: 
      logger.set_step(it) # debe hacerse antes de guardar los argumentos de esa iteracion.
      ... # calcular métricas y pérdida
      logger.log_metric(metricname1, metricvalue1)
      logger.log_metric(metricname2, metricvalue2)
      logger.log_train_loss(trainlossvalue)
      logger.log_test_loss(testlossvalue)
  logger.finish()     
  ```

  Atención: `ExperimentLogger` supone que luego de usar `set_step`, se loggea 
  un y solo un valor para cada métrica y loss usada. Estos ejemplos serían
  inválidos: 
  
  **Dos valores de la misma métrica**

  ```python
  logger.set_step(it) 
  logger.log_metric(metricname, metricvalue1)
  logger.log_metric(metricname, metricvalue2)
  ```

  **Un valor no loggeado consistentemente**
  Cada valor que se loggea debe hacerse en cada iteración,  y solo una vez.
  ```python
  logger.set_step(it) 
  logger.log_metric(metricname1, metricvalue1)
  logger.log_metric(metricname2, metricvalue2)
  
  # next iteration 
  logger.set_step(it+1) 
  logger.log_metric(metricname2, metricvalue2)

  # next iteration
  logger.set_step(it+2) 
  ```

  Adicionalmente es posible guardar artefactos, lo que es útil, por ejemplo, 
  para guardar parámetros. Pensado para hacerlo una vez, no en cada iteración.
  ```python
  logger.set_step(it) 
  logger.add_artifact("last_param", params)
  ```
  """
  @abstractmethod
  def init(self, project = None, group = None, config = None):
    """
      Inicia un nuevo experimento
    # Argumentos 
    - `project`: str 
    - `group`: str, nesting inside of project 
    - `config`: dict of hyperparameters 
    """
    pass 

  @abstractmethod
  def set_step(self, step):
    """
      Fija el paso de la iteración actual.
    Debe hacerse en cada iteración, antes de loggear las pérdidas y métricas de 
    esa iteración. 

    # Argumentos 
    - `step`: int 
    """
    pass 

  @abstractmethod
  def log_metric(self, metricname, metricvalue):
    """
      Loggea una metrica durante el entrenamiento 
    # Argumentos 
    - `metricname`: str, nombre identificador 
    - `metricvalue`: num, valor 
    """
    pass

  @abstractmethod
  def log_summary(self, name, value):
    """
      Loggea un resumen del entrenamiento 
    # Argumentos 
    - `name`: str, nombre identificador del item resumen  
    - `value`: num, valor 
    """
    pass

  @abstractmethod
  def log_image(self, imgname, img):
    """
      Loggea una imagen durante el entrenamiento 
    # Argumentos 
    - `imgname`: str, nombre identificador 
    - `img`: array
    """
    pass

  @abstractmethod
  def log_train_loss(self, value):
    """
      Loggea la funcion de perdida del entrenamiento 
    # Argumentos 
    - `value`: num, valor 
    """
    pass

  @abstractmethod
  def log_test_loss(self, value):
    """
      Loggea la funcion de perdida del entrenamiento 
    # Argumentos 
    - `value`: num, valor 
    """
    pass

  @abstractmethod
  def add_artifact(self, name, value):
    """
      Agrega un artefacto. Útil para guardar parámetros entrenados del modelo o datos.
    # Argumentos 
    - `name`: str, nombre
    - `value`: num, valor 
    """
    pass

  @abstractmethod
  def print_backend_info(self):
    """
      Imprimer información del backend del ExperimentLogger
    Actualmente las opciones son: 
    - Weights & Biases
    - Comet ML 
    - Local 
    """
    pass

  @abstractmethod
  def log_info(self, strinfo):
    """
      Toda la información se muestra como un archivo.
    Util para mostrar progreso del entrenamiento.
    - `strinfo`: str, 
    """
    pass

  @abstractmethod
  def finish(self):
    """
      Finaliza el experimento.
    """
    pass
 
# class WandBLogger(ExperimentLogger):
#   """
#   Wrapper alrededor del Logger de Weight and Bias.
#   """
#   # overriding abstract method
#   def __init__(self):
#     pass
#   def set_step(self, step): 
#     ver: https://docs.wandb.ai/guides/track/log#stepwise-and-incremental-logging
#     pass  
# log_dict = {"train/loss": train_loss_value}
# log_dict[f"train/metric/{metric_name}"] = metric_value
# log_dict[f"train/sample/{gen_name}"] = wandb.Image(onp.array(gen_func(params)))
# log_dict["test/loss"] = test_loss_value
# wandb.log(log_dict)
# wandb.run.summary["final_accuracy"] = metric_value
# wandb.run.finish()

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger

class LocalLogger(ExperimentLogger):
  """
  Permite emular el comportamiendo de [Weight and Bias](https://wandb.ai/site)
  de forma local.

  ## Ejemplo 
  
  Para usarlo se debe crear un ``LocalLogger` antes
  del entrenamiento con la función `inrmri.basic_nn.train`.
  La carpeta (en este caso `"./results"`) debe haber sido
  creada previamente.

  ```python
  logger = LocalLogger("./results")
  ```
  
  Se necesitan además varios parámetros para inicializar 
  el `Logger`, los que serán proporcionados a la función
  `train` mediante el diccionario `logger_init_params`.
  `"project"` y `"group"` permiten organizar los resultados
  mediantes la creación/uso de subcarpetas dentro de la
  carpeta principal `"./results"`. El diccionario
  `config_dict` contiene todos los parámetros que uno
  quiera guardar para poder replicar el experimento
  posteriormente (tipo de red, número de capas, otros
  hyperparámetros, parámetros del procesamiento de datos,
  información del entrenamiento como tamaño del _batch_,
  _learning rate_ o número de iteraciones, etc).
  
  ```python
  logger_init_params = {
    "project" : "radon-recos",
    "group"   : "tests-with-small-sigma",
    "config"  : config_dict
  }
  ```
  
  Se creará una carpeta `radon-recos` dentro de `"./results"`, 
  y una carpeta `tests-with-small-sigma` dentro de `radon-recos`.
  En caso de que las carpetas ya existan se usarán esas en lugar 
  de crear carpetas nuevas.

  Al final del entrenamiento se han generado tres archivos:
  - `config_RANDOM-RUN-NUMBER.pickle`: contiene el diccionario 
    `config_dict`.
  - `log_RANDOM-RUN-NUMBER.log`: contiene el historial de logs del
  entrenamiento (avances de la función de pérdida, métricas, etc.)
  - `results_RANDOM-RUN-NUMBER.pickle`: contiene un diccionario
  con los resultados entregados por el entrenamiento (parámetros
  finales, parámetros intermedios, historial de la _loss_, métricas,
  etc).

  `RANDOM-RUN-NUMBER` es un adjetivo, un sustantivo y un número de tres
  dígitos aleatorio que se la da a la respectiva corrida. Esto permite
  volver a correr varias veces el mismo experimento sin reescribir
  los resultados.
  
  > La razón tras ese comportamiento es que antes generaba el nombre
  del experimento a partir de los parámetros que había usado para correrlo.
  Por ejemplo, tendría un nombre `learning-rate-0-5_batch10_iters5000`. 
  Este método fallaba cada vez que necesitaba variar un parámetro
  que no había considerado en la generación del nombre (agregar por ejemplo
  el tamaño de la red `red-512-512-512`), generando situaciones donde
  sobreescribía los resultados previamente obtenidos.
  
  Para el caso en que `RANDOM-RUN-NUMBER` es `legal-leaf-661`, la
  estructura de archivos es de la forma:

  ```
  /results
  ├── radon-recos
  │   ├── tests-with-small-sigma
  │   │   ├── config_legal-leaf-661.pickle
  │   │   ├── log_legal-leaf-661.log
  │   │   ├── results_legal-leaf-661.pickle
  ```
  
  Los parámetros `"project"` y `"group"` pueden ser definidos
  como `""` (un string vacío). En este caso no se usará la
  respectiva subcarpeta y los resultados serán guardados
  directamente en la carpeta principal.  
  """
  def __init__(self, folder):
    """
    - `folder`: path donde guardar experimentos 
    """
    if Path(folder).is_dir():
      self._folder = folder
    else: 
      print(f'{folder} is not directory')
    
  def init(self, project = "", group = "", config = None):
    self._experimentname   = self._get_random_experiment_name()
    self._experimentfolder = os.path.join(self._folder, project, group)
    
    Path(self._experimentfolder).mkdir(parents=True, exist_ok=True)
    with open(self._get_config_path(), 'wb') as handle: # save config 
      pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    self._results = {}
    self._steps = []

    # logging.basicConfig(filename=self._get_log_path(), encoding='utf-8', level=logging.INFO)
    # first file logger
    self._info_logger = setup_logger(self._experimentname + '_logger', self._get_log_path())
    
  def set_step(self, step): 
    self._save_results()
    self._steps.append(step)
    
  def log_metric(self, metricname, metricvalue): 
    self._add_values_under_subkey_to_results("metrics", metricname, metricvalue)

  def log_train_loss(self, value): 
    self._add_values_under_subkey_to_results("loss", "train", value)
  
  def log_test_loss(self, value): 
    self._add_values_under_subkey_to_results("loss", "test", value)

  def log_image(self, imgname, img): 
    self._add_values_under_subkey_to_results("samples_imgs", imgname, img)

  def log_summary(self, name, value):
    self._add_values_under_subkey_to_results("summary", name, value)

  def add_artifact(self, name, value): 
    self._add_values_under_subkey_to_results("artifacts", name, value)    

  def log_info(self, strinfo):
    self._info_logger.info(strinfo)

  def finish(self):
    self._save_results()

  def _save_results(self): 
    with open(self._get_results_path(), 'wb') as handle: 
      pickle.dump(self._results, handle, protocol=pickle.HIGHEST_PROTOCOL)

  def _add_values_under_subkey_to_results(self, superkey, subkey, value): 
    if superkey not in self._results:
      self._results[superkey] = {}

    if subkey not in self._results[superkey]:
      self._results[superkey][subkey] = []

    self._results[superkey][subkey].append(value)

  def print_backend_info(self):
    print("Local logger:")
    print(" - Experiment name:", self._experimentname)
    print(" - Experiment folder:", self._experimentfolder)

  def _get_config_path(self):
    return os.path.join(self._experimentfolder, 'config_' + self._experimentname + '.pickle')     

  def _get_results_path(self):
    return os.path.join(self._experimentfolder, 'results_' + self._experimentname + '.pickle')     

  def _get_log_path(self):
    return os.path.join(self._experimentfolder, 'log_' + self._experimentname + '.log')     

  def _get_random_experiment_name(self):
    """
      Genera un nombre al azar para un experimento
    El nombre es de la forma adjectivo-sustantivo-numero
    # Ejemplos: 
    - `"decorous-wrecker-863"`
    - `"tart-reminder-233"`
    - `"clammy-flick-223"`
    """
    
    r = RandomWord()
    adjective = r.word(include_parts_of_speech=["adjectives"], word_min_length=3, word_max_length=8)
    noun = r.word(include_parts_of_speech=["nouns"], word_min_length=3, word_max_length=8)
    number = pyrandom.randint(100,1000)

    name = f"{adjective}-{noun}-{number}"
    return name 
