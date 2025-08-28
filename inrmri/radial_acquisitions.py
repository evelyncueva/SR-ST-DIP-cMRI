"""
Radial acquisitions, guardadas como una trayectoria y un conjunto de datos.

Recibe una lista de angulos por tiempo, que transforma en una trayectoria. 

Es la principal interfaz con los datos que reciben la redes tipo INR,
mediante la funcion `generate_dataset` que entrega los datos de training X e Y. 
"""
from jax import numpy as np 
from inrmri.fourier import get_points_in_angle
from jax import vmap 
from inrmri.bart import bart_acquisition_from_arrays, BARTAcquisition

def radial_trajs(angles, nsamples):
    """
    # Argumentos 
    - `frame_angles`: array(float), shape (nframes, spokes_per_frame)

    # Salida: 
    - array(float), shape (nframes, spokes_per_frame, nsamples, 2) (2 por real e imaginario)
    
    """
    return vmap(vmap(lambda angle: get_points_in_angle(angle, nsamples)))(angles)

def add_times_to_framestrajxy(framestrajxy):
    """
    Agrega la información temporal a las trayectorias de muestreo radial.

    Esta función recibe las trayectorias de k-space organizadas por frame y spokes,
    y devuelve un arreglo aplanado donde cada spoke incluye un valor temporal que
    identifica a qué frame pertenece. El tiempo se normaliza en el intervalo [0, 1).

    Parameters
    ----------
    framestrajxy : np.ndarray of shape (n_frames, n_spokes_per_frame, n_samples, 2)
        Trayectorias de muestreo en k-space:
        - `n_frames`: número de frames.
        - `n_spokes_per_frame`: número de spokes en cada frame.
        - `n_samples`: número de muestras en cada spoke (readout).
        - `2`: coordenadas de cada punto en k-space (kx, ky).

    Returns
    -------
    np.ndarray of shape (n_frames * n_spokes_per_frame, n_samples + 1, 2)
        Arreglo de trayectorias:
        - Se combinan todos los spokes de todos los frames en una sola dimensión.
        - Se añade un primer "sample" en cada spoke, que contiene el tiempo
          asociado al frame (repetido en las dos coordenadas para mantener
          compatibilidad con la dimensión `2`).
        - El resto de samples corresponden a las coordenadas (kx, ky).

    Notes
    -----
    - `times` se genera con `np.linspace(0, 1, n_frames, endpoint=False)`,
      lo cual asigna tiempos uniformemente espaciados en [0,1).
    - La dimensión extra `+1` en `n_samples` corresponde al punto temporal añadido.

    Examples
    --------
    - framestrajxy.shape
      (2, 3, 5, 2)   # 2 frames, 3 spokes por frame, 5 samples por spoke
    - out = add_times_to_framestrajxy(framestrajxy)
    - out.shape
      (6, 6, 2)      # 2*3 spokes en total, 5+1 samples (incluye tiempo), 2 coords
    """
    nframes, nspokes_per_frame, nsamples, dims = framestrajxy.shape
    # Crea un array de tiempos (nframes * nspokes, 1)
    times = (np.linspace(0, 1, nframes, endpoint=False)[:, None]
             * np.ones(nspokes_per_frame)[None, :]).ravel()[:, None]

    # Duplica el tiempo en dos columnas para simular coordenadas (t, t)
    times = np.stack([times, times], axis=-1)

    # Une los tiempos con las trayectorias de k-space reordenadas
    return np.r_['1', times,
                 np.reshape(framestrajxy, (nframes * nspokes_per_frame, nsamples, dims))]



def kFOV_limit_from_spoke_traj(spoke):
    """
    Calcula un límite aproximado del campo de visión (FOV) en k-space a partir de un spoke.

    La estimación se obtiene midiendo la norma euclidiana del punto extremo del spoke
    y normalizándola por el número total de puntos en el readout.

    Parameters
    ----------
    spoke : np.ndarray
        Trayectoria de un spoke en k-space.
        Forma: (n_readout_points, 2).
        - `n_readout_points`: número de muestras en el readout.
        - `2`: coordenadas en k-space (kx, ky).

    Returns
    -------
    float
        Límite estimado del FOV en k-space para el spoke dado.

    Notes
    -----
    - Se asume que el spoke está centrado, por lo que el punto extremo (`spoke[0]`)
      corresponde a la máxima desviación en k-space desde el origen.
    - El valor retornado es proporcional al radio máximo en k-space dividido
      por la densidad de muestreo (`n_readout_points`).

    Examples
    --------
    >>> spoke.shape
    (256, 2)
    >>> kFOV_limit_from_spoke_traj(spoke)
    0.03125
    """
    assert spoke.ndim == 2
    assert spoke.shape[-1] == 2
    readout_points = spoke.shape[0]
    extreme_point = spoke[0]
    return np.linalg.norm(extreme_point) / readout_points


def radial_acquisition_from_angles(frame_angles, data):
    """
    # Argumentos 
    - `frame_angles`: array(float), shape (frames, spokes_per_frame)
    - `data`: array(complex), shape (ncoils, frames, spokes_per_frame, nsamples)
    """
    assert frame_angles.ndim == 2
    assert data.ndim == 4
    assert frame_angles.shape == data.shape[1:3]
    nsamples = data.shape[-1]
    trajs = radial_trajs(frame_angles, nsamples)
    return RadialAcquisitions(trajs, data)

class RadialAcquisitions:
    """
    Clase para almacenar y manipular trayectorias y datos de adquisiciones radiales en MRI.

    Esta clase es la interfaz principal entre las adquisiciones de datos,
    el entrenamiento de la red neuronal (Deep Image Prior u otros métodos)
    y la interacción con BART (Berkeley Advanced Reconstruction Toolbox).
    Normalmente, los objetos de esta clase son generados a partir de fuentes de datos
    que proveen las trayectorias de muestreo y los datos de k-space.

    Parameters
    ----------
    trajs : np.ndarray (float)
        Trayectorias de muestreo en k-space.
        Forma: (n_frames, n_spokes_per_frame, n_samples, 2),
        donde el último eje corresponde a las coordenadas (kx, ky).
    data : np.ndarray (complex)
        Datos de Fourier adquiridos por los coils.
        Forma: (n_coils, n_frames, n_spokes_per_frame, n_samples).

    Attributes
    ----------
    trajs : np.ndarray
        Trayectorias de muestreo radial en k-space.
    data : np.ndarray
        Datos de k-space adquiridos.

    Examples
    --------
    Crear una instancia con una adquisición radial y mostrar su resumen:
    radialdata = RadialAcquisitions(trajs, data)
    print(radialdata)
    Radial acquisition: 9 coils, 20 frames of 8 spokes each, 10 samples per spoke.

    Generar dataset para entrenar la red:
    dataX, dataY = radialdata.generate_dataset()
    """

    def __init__(self, trajs, data):
        """
        Inicializa el objeto con trayectorias y datos de k-space.

        Parameters
        ----------
        trajs : np.ndarray (float)
            Array con las trayectorias de muestreo en k-space.
            Forma: (n_frames, n_spokes_per_frame, n_samples, 2)
            
            - `n_frames`: número total de frames en la secuencia dinámica.
            - `n_spokes_per_frame`: número de spokes por frame.
            - `n_samples`: número de muestras por readout (longitud del spoke).
            - El último eje de tamaño 2 corresponde a las coordenadas (kx, ky).
            - Se espera que `trajs.max() ≈ n_samples // 2`.

        data : np.ndarray (complex)
            Datos de Fourier (k-space) adquiridos por los coils.
            Forma: (n_coils, n_frames, n_spokes_per_frame, n_samples)
            
            - `n_coils`: número de bobinas de recepción (coils).
            - `n_frames`: número de frames en la secuencia dinámica.
            - `n_spokes_per_frame`: número de spokes por frame.
            - `n_samples`: número de muestras por readout.
        """

        assert trajs.ndim == 4
        assert trajs.shape[-1] == 2
        assert data.ndim == 4
        assert trajs.shape[:3] == data.shape[1:4]
        self.trajs = trajs 
        self.data = data 
    
    def get_number_of_coils(self): 
        return self.data.shape[0]

    def get_total_frames(self):
        return self.data.shape[1]
    
    def get_spokes_per_frame(self): 
        return self.data.shape[2]
    
    def get_read_out_dim(self): 
        return self.data.shape[3]

    def generate_dataset(self):
        """
        Genera el dataset de entrenamiento a partir de las trayectorias y datos de k-space.

        Returns
        -------
        dataX : np.ndarray
            Trayectorias de muestreo con información temporal añadida.
            Forma: (n_frames * n_spokes_per_frame, n_samples + 1, 2).
            - Incluye coordenadas (kx, ky) y un valor temporal asociado al frame.
        dataY : np.ndarray
            Datos de Fourier adquiridos en k-space.
            Forma: (n_frames * n_spokes_per_frame, n_coils, n_samples, 1).
        """
        # Reorganiza los datos de k-space: combina frames y spokes en un solo eje.
        fourier_radial_data = self.data.reshape(
            self.get_number_of_coils(),
            self.get_total_frames() * self.get_spokes_per_frame(),
            self.get_read_out_dim()
        )

        # Mueve el eje de coils para que quede después del índice de spokes.
        # De (n_coils, total_spokes, nsamples) → (total_spokes, n_coils, nsamples)
        fourier_radial_data = np.moveaxis(fourier_radial_data, 0, 1)

        # Añade una dimensión extra al final → (total_spokes, n_coils, nsamples, 1)
        dataY = fourier_radial_data[..., None]

        # Prepara las trayectorias, agregando el tiempo del frame como primera muestra.
        dataX = add_times_to_framestrajxy(self.trajs)

        return dataX, dataY


    def __str__(self): 
        return f"Radial adquisition: {self.get_number_of_coils()} coils, {self.get_total_frames()} frames of {self.get_spokes_per_frame()} spokes each, {self.get_read_out_dim()} samples per spokes."
    
    def to_bart(self, datapath, name): 
        """
        Returns BARTAcquisition object 
        """
        return bart_acquisition_from_arrays(np.copy(self.trajs), np.copy(self.data), datapath, name) 

    def unbin_data(self):
        new_data_shape = (self.data.shape[0], -1, 1, self.data.shape[-1]) # preserve dims 0 and -1, put everthing in RO dim 
        new_traj_shape = (-1, 1, self.trajs.shape[-2], self.trajs.shape[-1])
        self.data = self.data.reshape(new_data_shape)
        self.trajs = self.trajs.reshape(new_traj_shape)

    def bin_data(self, new_spokes_per_frame):
        self.unbin_data()
        new_frames = self.data.shape[1] // new_spokes_per_frame
        new_total_spokes = new_frames * new_spokes_per_frame
        new_data_shape = (self.data.shape[0], new_frames, new_spokes_per_frame, self.data.shape[-1]) 
        self.data = self.data[:,:new_total_spokes, ...].reshape(new_data_shape)
        new_trajs_shape = (new_frames, new_spokes_per_frame, self.trajs.shape[-2], self.trajs.shape[-1]) 
        self.trajs  = self.trajs[:new_total_spokes, ...].reshape(new_trajs_shape)

def create_radial_acq_from_bart(bartacq:BARTAcquisition): 
    return radial_acquisition_from_angles(bartacq.calculate_frameangles(), bartacq.calculate_radial_acq_data())

def is_spoke_traj_centered(spoke) -> bool:
    """
    Verifica si una trayectoria radial (spoke) está centrada en k-space.

    El spoke se considera centrado si el punto en el índice central
    (según la convención de la FFT) coincide con el origen [0., 0.].

    Parameters
    ----------
    spoke : np.ndarray
        Trayectoria de un spoke individual en k-space.
        Forma: (n_readout_points, 2).
        - `n_readout_points`: número de muestras en el readout.
        - `2`: coordenadas en k-space (kx, ky).

    Returns
    -------
    bool
        `True` si el punto central es aproximadamente [0., 0.],
        `False` en caso contrario.

    Notes
    -----
    - El índice central se calcula como `n_readout_points // 2`, de acuerdo con
      la convención de la FFT (donde la frecuencia cero se ubica en el centro).
    - La comparación con [0., 0.] se realiza usando `np.isclose` para tolerar
      pequeñas imprecisiones numéricas.

    Examples
    --------
    >>> spoke.shape
    (256, 2)
    >>> is_spoke_traj_centered(spoke)
    True
    """
    assert spoke.ndim == 2
    assert spoke.shape[-1] == 2
    fft_center_index = spoke.shape[0] // 2
    center = spoke[fft_center_index]
    is_center_close_to_0 = np.all(np.isclose(center, np.array([0., 0.])))  # bool
    return is_center_close_to_0

def are_spoke_trajs_centered(spokes) -> bool:
    """
    Verifica si un conjunto de trayectorias radiales (spokes) está centrado en k-space.

    Esta función aplica la verificación de centrado (`is_spoke_traj_centered`)
    a cada spoke de forma vectorizada mediante `jax.vmap`, y devuelve un único
    booleano indicando si **todas** las trayectorias están correctamente centradas.

    Parameters
    ----------
    spokes : np.ndarray
        Trayectorias radiales de k-space.
        Forma: (n_spokes, n_readout_points, 2).
        - `n_spokes`: número total de spokes.
        - `n_readout_points`: número de puntos en cada readout.
        - `2`: coordenadas en k-space (kx, ky).

    Returns
    -------
    bool
        `True` si todas las trayectorias están centradas,
        `False` en caso contrario.

    Notes
    -----
    - Esta función llama a `is_spoke_traj_centered` para cada spoke individual.
    - El centrado implica que la frecuencia cero (origen de k-space)
      se encuentre en la posición esperada de cada readout.

    Examples
    --------
    >>> spokes.shape
    (128, 256, 2)   # 128 spokes, 256 puntos por readout
    >>> are_spoke_trajs_centered(spokes)
    True
    """
    are_spokes_centered = vmap(is_spoke_traj_centered)(spokes)  # array de bools
    return np.all(are_spokes_centered)

def check_centered_train_X(train_X):
    """
    Verifica si las trayectorias de muestreo en k-space están correctamente centradas.

    Esta función revisa que la frecuencia asociada al origen (0) se encuentre en la
    posición esperada dentro de `train_X`. El chequeo se hace sobre los spokes de las
    trayectorias, excluyendo el primer punto que contiene la marca temporal del frame.

    Parameters
    ----------
    train_X : np.ndarray
        Trayectorias de entrenamiento generadas con `RadialAcquisitions.generate_dataset()`.
        Forma: (n_frames * n_spokes_per_frame, n_samples + 1, 2).
        - La primera muestra (+1) contiene el tiempo del frame.
        - Las restantes corresponden a coordenadas de k-space (kx, ky).

    Returns
    -------
    bool
        `True` si las trayectorias están correctamente centradas,
        `False` en caso contrario.

    Notes
    -----
    - Esta función llama internamente a `are_spoke_trajs_centered`, que realiza
      la verificación de centrado en cada spoke.
    - Es útil como comprobación previa a entrenar la red o pasar los datos a BART.

    Examples
    --------
    dataX, dataY = radialdata.generate_dataset()
    check_centered_train_X(dataX)
    True
    """
    spokes = train_X[:, 1:, :]  # excluye la primera muestra (tiempo)
    centered_spokes = are_spoke_trajs_centered(spokes)
    return centered_spokes

def check_dataset_dims(train_X, train_Y):
    assert train_X.ndim == 3, f"train_X.ndim is {train_X.ndim}, should be 3: (batch, read out + 1, 2)"
    assert train_Y.ndim == 4, f"train_Y.ndim is {train_Y.ndim}, should be 4: (batch, ncoil, read out, 1)"
    assert train_Y.shape[-1] == 1, f"train_Y.shape[-1] is {train_Y.shape[-1]}, should be 1"
    assert train_X.shape[0] == train_Y.shape[0], f"train_X and train_Y have different batch size {train_X.shape[0]} and {train_Y.shape[0]}"
    assert train_X.shape[1] == train_Y.shape[2] + 1, f"train_X.shape[1] {train_X.shape[1]} and train_Y.shape[2]+1 ({train_Y.shape[2]+1}) should be the same"
    assert train_X.shape[-1] == 2, f"train_X.shape[-1] {train_X.shape[-1]} should be 2"

def check_correct_dataset(train_X):
    """
    Verifica que las trayectorias generadas en el dataset estén centradas.

    Esta función recibe el dataset `train_X`, normalmente producido por
    `RadialAcquisitions.generate_dataset()`, y utiliza la función auxiliar
    `check_centered_train_X` para comprobar si las trayectorias de muestreo
    en k-space están correctamente centradas alrededor del origen.

    Parameters
    ----------
    train_X : np.ndarray
        Trayectorias de muestreo en k-space con información temporal añadida.
        Forma: (n_frames * n_spokes_per_frame, n_samples + 1, 2).

    Returns
    -------
    bool
        `True` si las trayectorias están centradas, `False` en caso contrario.

    Notes
    -----
    - Imprime un mensaje indicando si las trayectorias están centradas o no.
    - Sirve como chequeo rápido de calidad antes de entrenar la red.

    Examples
    --------
    dataX, dataY = radialdata.generate_dataset()
    check_correct_dataset(dataX)
    The trajectories are centered
    True
    """
    centered_spokes = check_centered_train_X(train_X)
    str_centered_spokes = 'centered' if centered_spokes else 'not centered!'
    print(f"The trajectories are {str_centered_spokes}")
    return centered_spokes
