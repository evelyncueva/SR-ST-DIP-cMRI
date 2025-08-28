# ----------------------------------------------------------------------
# inrmri.data.radial_cine
# Funciones para trabajar con los datos del Rafa 
# ----------------------------------------------------------------------
import numpy as np 
from inrmri.radial_acquisitions import RadialAcquisitions
from scipy.io import loadmat 

def load_rafaels_data(path):
    """
    Loads radial k-space data and associated cardiac cycle information 
    from Rafael's `.mat` file format.

    Parameters
    ----------
    path : str
        Path to the `.mat` file containing radial MRI acquisition data.

    Returns
    -------
    ra : RadialAcquisitions
        An instance of `RadialAcquisitions` containing:
        - `ktraj`: k-space trajectory with shape (1, total_spokes, ro, 2)
        - `kdata`: k-space data with shape (ncoils, total_spokes, 1, ro)

    ecg : np.ndarray
        Indices of the spokes corresponding to the start of each cardiac cycle,
        estimated from `pmu_time` using `get_cycle_start_indices_from_pmu`.

    Notes
    -----
    The `.mat` file is expected to contain the following variables:

    - `data`: complex k-space data, shape (ro, total_spokes, ncoils)
    - `traj`: k-space trajectory, shape (ro, total_spokes, 2) [real, imag]
    - `pmu_time`: 1D array (1, total_spokes), containing PMU timestamps
    - `bin`: 1D array (1, total_spokes), maps each spoke to a cardiac bin
    - `cbin`: cell array (1, 1, nbins), where each element contains a list 
      of spoke indices associated with that bin (length may vary)

    Of these, only `data`, `traj` and `pmu_time` are required.
    """
    data = loadmat(path)
    kdata = data['data'] # (ro, total spoke, coils)
    kdata = np.transpose(kdata, (2, 1, 0)) # (coils, total spoke, ro)
    ktraj = data['traj'] #  (ro, total_spoke, 2 real-imag)
    ktraj = np.transpose(ktraj, (1, 0, 2)) # (total_spoke, ro, 2 real-imag) 
    ra = RadialAcquisitions(ktraj[:,None,:,:], kdata[:,:,None,:])
    ecg = get_cycle_start_indices_from_pmu(data['pmu_time'][0])
    return ra, ecg 

def get_cycle_start_indices(bins):
    """
    Identifies the indices corresponding to the start of each new cycle in a sequence
    of bins, typically used for detecting cardiac cycle starts from radial MRI acquisition data.

    Parameters
    ----------
    bins : array-like (1D)
        Sequence of bin labels (typically from `data['bins'][0]`) where each value indicates 
        the bin assigned to a spoke in a radial acquisition. The bins are expected to cycle 
        from a maximum value back to a minimum value at the beginning of each new cardiac cycle.

    Returns
    -------
    ecg : numpy.ndarray
        Array of indices indicating the start of each new bin cycle, interpreted as ECG trigger points.
        Includes index 0 if the first value corresponds to the minimum bin (i.e., the start of a cycle).
    
    Notes
    -----
    This function assumes that a transition from the maximum bin label to the minimum bin label 
    indicates the beginning of a new cardiac cycle (analogous to an R-wave trigger in ECG).
    It detects these transitions by checking for differences equal to (min_bin - max_bin).
    """
    ecg = np.array(bins, dtype=np.int16)  # Convert to int16 for safe subtraction
    frames = np.unique(ecg)
    min_frame = frames.min()
    max_frame = frames.max()

    add_idx_zero = ecg[0] == min_frame  # Check if first frame is a cycle start

    # Detect transitions from max_frame to min_frame (i.e., new cardiac cycles)
    ecg = np.where(np.diff(ecg) == (min_frame - max_frame))[0] + 1

    if add_idx_zero:
        # Include the first index if it corresponds to a cycle start
        ecg = np.concatenate((np.array([0]), ecg), axis=0)

    return ecg

def get_cycle_start_indices_from_pmu(pmu_times, threshold=100, start_threshold=15):
    """
    Detects cardiac cycle start indices from PMU (physiological monitoring unit)
    timestamps, based on large time jumps (e.g., due to ECG R-wave triggers).

    Parameters
    ----------
    pmu_times : array-like (1D)
        Sequence of time values (e.g., in milliseconds, tipically from
        `data['pmu_time'][0]`). These values are expected to increase within a
        cardiac cycle and reset cycle transitions.

    threshold : float, optional (default=100)
        Minimum difference between consecutive time values to be considered a cycle boundary.
        Any jump greater than or equal to this threshold is marked as a new cycle start.

    start_threshold : float, optional (default=15)
        If the first PMU time value is smaller than this threshold, index 0 is considered
        to be the start of the first cardiac cycle and is included explicitly.

    Returns
    -------
    ecg : numpy.ndarray
        Array of indices corresponding to the start of each cardiac cycle, based on time
        discontinuities in the PMU data and the optional inclusion of index 0.

    """
    pmu_times = np.array(pmu_times, dtype=np.int16) # for safe subtraction
    
    # Detect large time jumps indicating a new cardiac cycle
    ecg = np.where(np.diff(pmu_times) <= -threshold)[0] + 1

    # Optionally add index 0 if the first timestamp suggests the start of a new cycle
    if pmu_times[0] < start_threshold:
        ecg = np.concatenate(([0], ecg), axis=0)

    return ecg