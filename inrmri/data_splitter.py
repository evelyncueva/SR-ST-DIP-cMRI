from jax import numpy as np 
from jax import random 
from abc import ABC, abstractmethod

def random_choice(key, ntotal:int, nselected:int):
    idxs = random.choice(key, np.arange(ntotal), (nselected,))
    return idxs 

def train_validation_split_indexs(key, N, train_fraction): 
    train_idxs = random_choice(key, N, int(np.round(N * train_fraction)))
    val_idxs = np.array([idx for idx in np.arange(N) if idx not in train_idxs])
    return train_idxs, val_idxs

class DataLoader(ABC):
    """
        Clase para cargar datos 
    
    ## Modo de uso 

    Si `loader` es de tipo `DataLoader`, entonces se el dataset
    completo puede obtenerse haciendo.

    ```python 
    X_full, Y_full = loader.get_full_dataset()
    ```

    También es posible separar el dataset en un conjunto de
    entrenamiento y otro de validación de forma aleatoria.

    ```python 
    training_fraction = 0.7 
    key = random.PNGKey(0)

    X_train, Y_train, X_val, Y_val = loader.get_splitted_dataset(key, training_fraction)
    ``` 

    Los índices usados para separar entre training y validación 
    pueden obtenerse así: 

    ```python
    train_idxs, val_idxs = loader.get_split_indexs(key, training_fraction)
    ```
    
    ## Creación de un `DataLoader`

    Hay que instanciar la interfaz de `DataLoader` creando una
    subclase que tenga implementado el método `get_full_dataset`.

    """

    @abstractmethod
    def get_full_dataset(self): 
        """
        Devuelve una tupla X,Y con todos los pares datos disponibles 
        """
        pass 

    def get_number_of_samples(self): 
        return self.get_full_dataset()[0].shape[0]
    
    def get_split_indexs(self, key, train_fraction): 
        train_idxs, val_idxs = train_validation_split_indexs(key, self.get_number_of_samples(), train_fraction)
        return train_idxs, val_idxs

    def get_splitted_dataset(self, key, train_fraction):
        train_idxs, val_idxs = self.get_split_indexs(key, train_fraction)
        X_full, Y_full = self.get_full_dataset()
        X_train = X_full[train_idxs]
        Y_train = Y_full[train_idxs]

        X_val = X_full[val_idxs]
        Y_val = Y_full[val_idxs]
        return X_train, Y_train, X_val, Y_val

class SimpleDataLoader(DataLoader):

    def __init__(self, full_X, full_Y):
        assert full_X.shape[0] == full_Y.shape[0], "full_X y full_Y deben tener la misma cantidad de datos"
        self.X = full_X
        self.Y = full_Y

    def get_full_dataset(self):
        return self.X, self.Y 