from scipy.io import loadmat 
import numpy as onp
import jax.numpy as np
import matplotlib.pyplot as plt 
from jax import vmap
from inrmri.radon import calculate_angle 
from inrmri.radial_acquisitions import kFOV_limit_from_spoke_traj
from inrmri.radial_acquisitions import check_correct_dataset 

from inrmri.bart import bart_acquisition_from_arrays
from inrmri.data_harvard import get_csmaps_and_mask
import random
import optax

def seconds_to_min_sec_format(time_seconds):
    """
    Convert seconds to 'MM:SS' format.
    
    Parameters:
    - time_seconds (float or np.ndarray): Time in seconds.
    
    Returns:
    - str: Time formatted as MM:SS
    """
    time_seconds = float(time_seconds)
    minutes = int(time_seconds // 60)
    seconds = int(round(time_seconds % 60))
    return f"{minutes:02}:{seconds:02}"

def filter_and_get_columns(df, filter_params, target_columns):
    """
    Filters the DataFrame based on filter_params and returns target column values,
    including the original filter parameters in each result dictionary.
    """
    condition = np.ones(len(df), dtype=bool)
    for key, value in filter_params.items():
        condition = condition & (df[key] == value)

    filtered = df.loc[condition, target_columns]
    results = filtered.to_dict(orient='records')

    # Merge each result with the filter parameters
    return [{**filter_params, **res} for res in results]

def apply_transform(image, trans):
    """
    Applies flip and rotation to a 30-channel image based on dict_trans.

    Parameters:
    - image: np.ndarray of shape (H, W, 30)
    - dataset_name: str, e.g., 'DATA_1.5T'
    - volunteer_code: str, e.g., 'GC'
    - dict_trans: transformation dictionary

    Returns:
    - transformed image: np.ndarray of shape (H, W, 30)
    """

    if image.shape[-1] != 30:
        raise ValueError("Expected image with 30 channels in the last dimension.")

    # Apply horizontal flip
    if trans['flip_h']:
        image = np.flip(image, axis=1)

    # Apply vertical flip
    if trans['flip_v']:
        image = np.flip(image, axis=0)

    # Apply rotation (must be multiple of 90 degrees)
    rot = trans['rot']
    if rot not in [0, 90, 180, 270]:
        raise ValueError("Rotation must be 0, 90, 180, or 270 degrees.")
    k = rot // 90  # number of 90-degree rotations
    if k:
        # Rotate each channel independently
        image = np.stack([np.rot90(image[..., i], k=k) for i in range(30)], axis=-1)

    return image



def get_shedule(h_params):
    if h_params['lr_schedule'] == 'exponential_decay':
        schedule = optax.exponential_decay(
            init_value=h_params['lr_init_value'],        # initial learning rate
            transition_steps=h_params['lr_transition_steps'],  # how often to decay
            decay_rate=h_params['lr_decay_rate'],        # how much to decay each time
            staircase=h_params['lr_staircase']          # step-wise decay (set False for smooth decay)
        )
    elif h_params['lr_schedule'] == 'linear_schedule':
        schedule = optax.linear_schedule(
            init_value=h_params['lr_init_value'],      # start learning rate
            end_value=h_params['lr_end_value'],       # final learning rate
            transition_steps=h_params['lr_transition_steps']  # how many steps to decay over
        )
    elif h_params['lr_schedule'] == 'polynomial_schedule':
        schedule = optax.polynomial_schedule(
            init_value=h_params['lr_init_value'],      # start learning rate
            end_value=h_params['lr_end_value'],       # final learning rate
            transition_steps=h_params['lr_transition_steps'],  # how many steps to decay over
            power=h_params['lr_power'] 
        )
    elif h_params['lr_schedule'] == 'constant_schedule':
        schedule = optax.constant_schedule(
            value = h_params['lr_init_value']
        )
    else:
        print('fail :c')
        schedule = None
    return schedule

def plot_multi_axis(x, y_list, labels, figsize=(12, 4), xlim=None, save_path=None):
    """
    Plot multiple y curves sharing the same x, each with its own y-axis.

    Parameters:
    - x: array-like, common x values
    - y_list: list of array-like, each element is a y curve
    - labels: list of str, label for each y curve
    - figsize: tuple, size of the figure (width, height) in inches
    - xlim: tuple, (xmin, xmax) limits for x-axis
    - save_path: str, if given, path to save the figure (e.g., 'myplot.png')
    """
    x = np.array(x)
    # Basic strong colors
    base_colors = [
        'red', 'green', 'blue', 'orange', 'purple',
        'brown', 'cyan', 'magenta', 'black', 'pink'
    ]
    n_curves = len(y_list)
    colors = base_colors * (n_curves // len(base_colors) + 1)

    fig, ax_main = plt.subplots(figsize=figsize)
    axes = [ax_main]

    # Select visible x-range
    if xlim is not None:
        mask = (x >= xlim[0]) & (x <= xlim[1])
    else:
        mask = np.ones_like(x, dtype=bool)

    # Plot the first curve
    ax_main.plot(x, y_list[0], color=colors[0], label=labels[0])
    ax_main.set_ylabel(labels[0], color=colors[0])
    ax_main.tick_params(axis='y', colors=colors[0])
    if xlim:
        ax_main.set_xlim(xlim)
    visible_y = np.array(y_list[0])[mask]
    ax_main.set_ylim(np.min(visible_y), np.max(visible_y))

    # Plot remaining curves
    for idx in range(1, n_curves):
        ax_new = ax_main.twinx()
        ax_new.spines.right.set_position(("axes", 1 + 0.1 * idx))
        ax_new.plot(x, y_list[idx], color=colors[idx], label=labels[idx])
        ax_new.set_ylabel(labels[idx], color=colors[idx])
        ax_new.tick_params(axis='y', colors=colors[idx])

        visible_y = np.array(y_list[idx])[mask]
        ax_new.set_ylim(np.min(visible_y), np.max(visible_y))

        axes.append(ax_new)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_example_discrete_exp(a_list, b_list, n=10, type_comp = 2):
    x_values = onp.arange(-n, n, 1) 
    if type_comp==1: # combination
        for a in a_list:
            for b in b_list:
                y_values = (a * onp.exp(- onp.abs(b * x_values) )).astype(int)
                y_values[n] = 0
                plt.plot(x_values, y_values, 'o-', label='a=' + str(a) + ', b=' + str(b), alpha=0.5)
    elif type_comp==2: # independent
        for i in range(len(a_list)):
            a = a_list[i]
            b = b_list[i]
            y_values = (a * onp.exp(- onp.abs(b * x_values) )).astype(int)
            y_values[n] = 0
            plt.plot(x_values, y_values, 'o-', label='a=' + str(a) + ', b=' + str(b), alpha=0.5)

    plt.xlabel('Distance of other frames')
    plt.ylabel('Copied spokes from other frames')
    plt.title('Soft-binning (discrete exponential function)')
    plt.legend()
    plt.grid(True)
    plt.show()

def get_csm_format(ktraj, kdata, base_folder, dataset_name):
    '''
    Use bart to compute the coil sensitivity map (csm)
    
    Args:
    kdata               (array): coil data (readout, spokes, coils)
    ktraj               (array): kspace trajectory (spokes, readount, dim)
    base_folder           (str): directory of the data
    dataset_name          (str): selected dataset
    
    Results:
    csm                   (array): coil sensitivity map (csm) (coils, readout, readout)
    hollow_mask           (array): hollow_mask (readout, readout)
    bac_csm     (BARTAcquisition):  BARTAcquisition class used for the csm computation with bart
    '''
    dataset_name_csm = dataset_name + '_csm'
    ktraj_csm = onp.expand_dims(ktraj, axis=0)
    kdata_csm = onp.transpose(kdata, (2, 1, 0))
    kdata_csm = onp.expand_dims(kdata_csm, axis=1)
    #- traj: array(float), shape (frames, spokesperframe, readout, 2 o 3)
    #- data: array(complex), shape (coils, frames, spokesperframe, readout)
    bac_csm = bart_acquisition_from_arrays(ktraj_csm, kdata_csm, base_folder, dataset_name_csm)
    bac_csm.calculate_coil_sens()
    csm, hollow_mask = get_csmaps_and_mask(bac_csm)
    return csm, hollow_mask, bac_csm


def load_acq_data(folder_path):
    '''
    This function loads the information from a .mat file. 
    
    Args:
    folder_path (str): path to the matlab file
    
    Results:
    bins  (array): bin of each spoke
    cbin  (array): list of index of the spokes in each frame
    data (array): coil data (readout, spokes, coils)
    traj (array): kspace trajectory (spokes, readount, dim)
    '''
    n_dims = 2
    data = loadmat(folder_path)
    cbins = data['cbins']
    cbins = cbins.reshape((cbins.shape[-1]))
    # kspace data
    if 'kIN' in data:
        kdata = data['kIN']
    elif 'data' in data:
        kdata = data['data']
    # kspace trajectory
    n_readout, n_spokes, _ = kdata.shape
    if 'kpos' in data:
        ktraj = data['kpos'][0,0]
        ktraj = onp.reshape(ktraj, (n_spokes, n_readout, n_dims))
        ktraj = ktraj*n_readout
    elif 'traj' in data:
        ktraj = data['traj']
        ktraj = onp.transpose(ktraj, (1, 0, 2))
    # if attribute 'bin' doesn't exist create it
    if not 'bin' in data:
        n_spokes = ktraj.shape[0]
        bins = []
        for i in range(n_spokes) :
            counter = 0
            for j in range(len(cbins)):
                if i in cbins[j]:
                    bins.append(j)
                    counter +=1
            if counter==0:
                bins.append(None)
    else:
        bins = data['bins'][0,:]
    return bins, cbins, kdata, ktraj

def get_inf_about_data(ktraj, kdata, cbins):
    '''
    Gets variables from trajectory and coil arrays
    '''
    n_spokes = ktraj.shape[0]
    n_readout = ktraj.shape[1]
    n_coils = kdata.shape[2]
    n_bins = cbins.shape[0]
    n_dims = ktraj.shape[2]
    print('n_spokes = ', n_spokes)
    print('n_readout = ', n_readout)
    print('n_coils = ', n_coils)
    print('n_bins = ', n_bins)
    print('n_dims = ', n_dims)
    return n_spokes, n_readout, n_coils, n_bins, n_dims


def filtering_spokes(eliminated_spokes, bins, kdata, ktraj):
    '''
    This function loads the information from a .mat file. 
    
    Args:
    eliminated_spokes     (int): number of spokes to filter
    bins                (array): bin of each spoke
    kdata               (array): coil data (readout, spokes, coils)
    ktraj               (array): kspace trajectory (spokes, readount, dim)
    
    
    Results:
    bins_filtered         (array): bin of each spoke
    kdata_filtered        (array): coil data (readout, spokes, coils)
    ktraj_filtered        (array): kspace trajectory (spokes, readount, dim)
    hb_filtered           (array): hb of each spoke
    '''
    # check for None in bins list
    bins = [0 if x is None else x for x in bins]
    bins = onp.array(bins)
    # filter bins
    bins_filtered = bins.astype(float) - onp.min(bins)  # know will start from 0 
    bins_filtered = bins_filtered[eliminated_spokes:]
    # filter kdata and ktraj
    kdata_filtered = kdata[:,eliminated_spokes:,:]
    ktraj_filtered = ktraj[eliminated_spokes:, :,:]
    # heart bit estimation
    hb_filtered = []
    hb_count = 0
    hb_filtered.append(hb_count)
    for i in range(len(bins_filtered)-1):
        item = bins_filtered[i]
        nitem = bins_filtered[i+1]
        if nitem<item:
            hb_count += 1
        hb_filtered.append(hb_count) 
    return bins_filtered, kdata_filtered, ktraj_filtered, hb_filtered


def sort_data_by_bin(bins_filtered, kdata_filtered, ktraj_filtered, hb_filtered):
    '''
    Sort the spokes by the frame order. Before this function, the spokes are sorted by the time they were acquired 
    
    Args:
    kdata_filtered        (array): coil data (readout, spokes, coils)
    ktraj_filtered        (array): kspace trajectory (spokes, readount, dim)
    bins_filtered         (array): bin of each spoke
    hb_filtered           (array): hb of each spoke
    
    Results:
    
    kdata_sorted        (array): coil data (readout, spokes, coils)
    ktraj_sorted        (array): kspace trajectory (spokes, readount, dim)
    bins_sorted         (array): bin of each spoke
    hb_sorted           (array): hb of each spoke
    '''
    
    kdata_list = []
    ktraj_list = []
    bins_sorted = []
    hb_sorted = []
    
    for bin_value in onp.unique(bins_filtered):
        indexes = [i for i, x in enumerate(bins_filtered) if x == bin_value]
        kdata_list.append(kdata_filtered[:,indexes,:])
        ktraj_list.append(ktraj_filtered[indexes,:,:])
        bins_sorted += [bin_value] * len(indexes)
        hb_list = [hb_filtered[j] for j in indexes]
        hb_sorted += hb_list
        none_indices = [i for i, element in enumerate(hb_list) if element is None]
        for none_indice in none_indices:
            print(indexes[none_indice])
        
    kdata_sorted = onp.concatenate(kdata_list, axis=1)
    ktraj_sorted = onp.concatenate(ktraj_list, axis=0)
    return kdata_sorted, ktraj_sorted, bins_sorted, hb_sorted


def balance_spokes_duplicating(times, kdata_ordered, ktraj_ordered):
    print(kdata_ordered.shape, ktraj_ordered.shape)
    # statistics
    unique_values, counts = onp.unique(times, return_counts=True)
    n = onp.max(counts)
    n_frames = len(unique_values)
    # output
    ktraj_output = onp.random.random((n_frames, n, ktraj_ordered.shape[1], ktraj_ordered.shape[2]))
    kdata_output = onp.random.random((kdata_ordered.shape[2], n_frames, n, kdata_ordered.shape[0])) + 1j*onp.random.random((kdata_ordered.shape[2], n_frames, n, kdata_ordered.shape[0]))
    time_output = []
    # loop
    for i in range(len(unique_values)):
        unique_value = unique_values[i]
        time_output += [unique_value]*n
        index_list = onp.where(times == unique_value)[0]
        ktraj_aux = ktraj_ordered[index_list, :, :]
        kdata_aux = kdata_ordered[:, index_list, :]
        # k traj dimensions
        ktraj_aux = onp.expand_dims(ktraj_aux, axis=0)
        # k data dimensions
        kdata_aux = onp.transpose(kdata_aux, (2, 1, 0))
        kdata_aux = onp.expand_dims(kdata_aux, axis=1)
        # extend values
        current_spokes = kdata_aux.shape[2]
        sample_amount = n - current_spokes
        if sample_amount > 0:
            spokes_samples = onp.random.choice(current_spokes, size=sample_amount, replace=True)
            kdata_random_samples = kdata_aux[:, :, spokes_samples, :]
            ktraj_random_samples = ktraj_aux[:, spokes_samples, :, :]
            kdata_aux = onp.concatenate((kdata_aux, kdata_random_samples), axis=2)
            ktraj_aux = onp.concatenate((ktraj_aux, ktraj_random_samples), axis=1)
        ktraj_output[i,:,:,:] = ktraj_aux[0,:,:,:]
        kdata_output[:,i,:,:] = kdata_aux[:,0,:,:]
    return ktraj_output, kdata_output, time_output

def get_angles_dataset(ktraj_output_ordened, time_output):
    check_correct_dataset(ktraj_output_ordened)
    spclim = kFOV_limit_from_spoke_traj(ktraj_output_ordened[0,:,:])
    angles = vmap(calculate_angle)(ktraj_output_ordened)
    X_data = onp.stack([angles, time_output], axis=-1)
    return X_data, spclim

def reverse_order2standard(ktraj_output, kdata_output):
    ktraj_output_ordened = onp.transpose(ktraj_output, (0, 1, 2, 3))
    ktraj_output_ordened = onp.reshape(ktraj_output_ordened, (ktraj_output_ordened.shape[0]*ktraj_output_ordened.shape[1], ktraj_output_ordened.shape[2], ktraj_output_ordened.shape[3]))

    kdata_output_ordened = onp.transpose(kdata_output, (3, 1, 2, 0))
    kdata_output_ordened = onp.reshape(kdata_output_ordened, (kdata_output_ordened.shape[0], kdata_output_ordened.shape[1]*kdata_output_ordened.shape[2], kdata_output_ordened.shape[3]))

    return ktraj_output_ordened, kdata_output_ordened

def preprocess_ydata(kdata_ordered):
    Y_data = onp.transpose(kdata_ordered, (1, 2, 0))
    Y_data = onp.expand_dims(Y_data, axis=-1)
    Y_data = Y_data / onp.max ( onp.abs(Y_data) )
    return Y_data

def balance_spokes_from_neighbours(times, kdata_ordered, ktraj_ordered):
    # statistics
    unique_values, counts = onp.unique(times, return_counts=True)
    n = onp.max(counts)
    n_frames = len(unique_values)
    # outputs
    ktraj_output = onp.random.random((n_frames, n, ktraj_ordered.shape[1], ktraj_ordered.shape[2]))
    kdata_output = onp.random.random((kdata_ordered.shape[2], n_frames, n, kdata_ordered.shape[0])) + 1j*onp.random.random((kdata_ordered.shape[2], n_frames, n, kdata_ordered.shape[0]))
    times_output = []
    # loop
    for i in range(len(unique_values)):
        unique_value = unique_values[i]
        times_output += [unique_value] * n
        index_list = list(onp.where(times == unique_value)[0])
        elements_to_add = n - len(index_list)
        if elements_to_add > 0:
            elemenst_before = elements_to_add//2
            elemenst_after = elements_to_add - elemenst_before
            index_list_before = list(onp.where(times ==  unique_values[i-1])[0])
            index_list_after = list(onp.where(times ==  unique_values[(i+1)%len(unique_values)])[0])
            sample_before = random.sample(index_list_before, elemenst_before)
            sample_after = random.sample(index_list_after, elemenst_after)
            index_list = sample_before + index_list + sample_after
        ktraj_aux = ktraj_ordered[index_list, :, :]
        kdata_aux = kdata_ordered[:, index_list, :]
        # k traj dimensions
        ktraj_aux = onp.expand_dims(ktraj_aux, axis=0)
        # k data dimensions
        kdata_aux = onp.transpose(kdata_aux, (2, 1, 0))
        ktraj_output[i,:,:,:] = ktraj_aux
        kdata_output[:,i,:,:] = kdata_aux

    return ktraj_output, kdata_output, times_output

def balance_spokes(times, kdata_ordered, ktraj_ordered, algorithm_name, a=8, b=1.0):
    if algorithm_name == 'duplicating':
        ktraj_output, kdata_output, time_output = balance_spokes_duplicating(times, kdata_ordered, ktraj_ordered)
    elif algorithm_name == 'neighbours':
        ktraj_output, kdata_output, time_output = balance_spokes_from_neighbours(times, kdata_ordered, ktraj_ordered)
    elif algorithm_name == 'soft-binning':
        ktraj_output, kdata_output, time_output = soft_binning_descrite(times, kdata_ordered, ktraj_ordered, a, b)
    ktraj_output_ordened, kdata_output_ordened = reverse_order2standard(ktraj_output, kdata_output)
    return ktraj_output_ordened, kdata_output_ordened, time_output


#def preprocess_data(times, kdata_ordered, ktraj_ordered, algorithm_type, a=8, b=1.0):
#    # balancing
#    ktraj_output_ordened, kdata_output_ordened, time_output = balance_spokes(times, kdata_ordered, ktraj_ordered, algorithm_type, a=a, b=b)
#    # trajectory data
#    X_data, spclim = get_angles_dataset(ktraj_output_ordened, time_output)
#    # kspace data
#    Y_data = preprocess_ydata(kdata_output_ordened)
#    return Y_data, X_data, spclim

def preprocess_data(times, kdata, ktraj):
    # --- input ---
    # times:  Cardiac cycle percentaje, sorted  (spokes)
    # kdata:  Acquired data                     (readout, spokes, coils)
    # ktraj:  Kspace trajectory                 (spokes, readout, 2)
    # --- output ---
    # Y_data:  Cardiac cycle percentaje, sorted  (spokes)
    # X_data:  Acquired data                     (readout, spokes, coils)
    # spclim:  Kspace trajectory                 (spokes, readout, 2)
    # --------------------------------
    # ktraj, times --> X_data, spclim
    X_data, spclim = get_angles_dataset(ktraj, times)
    # kdata --> Y_data
    Y_data = preprocess_ydata(kdata)
    return Y_data, X_data, spclim

import os
from PIL import Image
import shutil

def create_folder(folder_path, reset=False):
    if os.path.exists(folder_path):
        if reset:
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)

def save_frames_as_gif_with_pillow(exp_folder, reco, filename='output_cine', vmax=1, saturation=1.0, fps=30):
    frames = []
    file_path = exp_folder + filename + '.gif'
    gif_images_folder = exp_folder + 'gif_images_' + filename +  '/'
    create_folder(gif_images_folder)
    
    # Calculate the duration for each frame in milliseconds (since Pillow uses ms)
    duration = int(1000 / fps)

    for frame in range(reco.shape[2]):
        temp_file_path = gif_images_folder + 'image_' + str(frame) + '.png'  # Temporary file path
        plt.figure(figsize=(8, 8))
        img_data = onp.clip(onp.abs(reco[:, :, frame]), 0, saturation * vmax)
        plt.imshow(img_data, cmap='bone', vmin=0, vmax=saturation * vmax)
        plt.axis('off')

        # Save each frame as a PNG file with high DPI to preserve quality
        plt.savefig(temp_file_path, bbox_inches='tight', pad_inches=0, dpi=30)
        plt.close()

        # Open the saved image and convert it to RGB mode for GIF compatibility
        img = Image.open(temp_file_path).convert("RGB")
        frames.append(img)

    # Save the frames as an optimized GIF with adaptive palette
    frames[0].save(
        file_path,
        save_all=True,
        append_images=frames[1:], 
        duration=duration,
        loop=0,
        optimize=True,  # Optimize the palette to retain better color quality
        disposal=2  # Ensures that each frame fully replaces the previous one
    )

import time

def recon_bart(times, kdata_sorted, ktraj_sorted, bart_files_folder, dataset_name):
    # csm
    csm, hollow_mask, bac_csm = get_csm_format(ktraj_sorted, kdata_sorted, bart_files_folder, dataset_name)
    
    # prepare recons
    ktraj_b, kdata_b, time_b = balance_spokes_duplicating(times, kdata_sorted, ktraj_sorted)
    
    # GRASP reconstruction
    bac_name_grasp = dataset_name + '_grasp_b'
    bac_grasp = bart_acquisition_from_arrays(ktraj_b, kdata_b, bart_files_folder, bac_name_grasp)
    iters = 100
    lagrangian_value = 5.
    lambda_value = 0.01
    grasp_exp_name = bac_name_grasp + str(iters) + '_lagrangian' + str(lagrangian_value).replace(".", "_") + '_lambda' + str(lambda_value).replace(".", "_")

    t0 = time.time()
    recon_grasp = bac_grasp.calculate_bart_reco_with_external_csmap(
        bac_csm, grasp_exp_name,
        lmbda=lambda_value, lagrangian=lagrangian_value, iters=iters
    )
    t1 = time.time()
    time_grasp = t1 - t0
    recon_grasp = recon_grasp.reshape((recon_grasp.shape[0], recon_grasp.shape[1], recon_grasp.shape[-1]))

    # SENSE reconstruction
    bac_name_sense = dataset_name + '_sense_b'
    t0 = time.time()
    recon_sense = bac_grasp.make_sense_reconstruction(bac_csm.csmappath(), bac_name_sense)
    t1 = time.time()
    time_sense = t1 - t0
    recon_sense = recon_sense.reshape((recon_sense.shape[0], recon_sense.shape[1], recon_sense.shape[-1]))

    return csm, hollow_mask, recon_grasp, recon_sense, time_grasp, time_sense


def plot_zoom_images(array_images, array_names, frame_to_show=0, iterative_zoom=3):
    fig, axes = plt.subplots(iterative_zoom, len(array_images), figsize=(len(array_images)*3, iterative_zoom*3))
    for i in range(iterative_zoom):
        for j in range(len(array_images)):
            current_image = array_images[j]
            crop_size_x = current_image.shape[0] / (i+1)
            crop_size_y = current_image.shape[1] / (i+1)
            x1 = int(current_image.shape[0]//2 - crop_size_x//2)
            x2 = int(current_image.shape[0]//2 + crop_size_x//2 - 1)
            y1 = int(current_image.shape[1]//2 - crop_size_y//2)
            y2 = int(current_image.shape[1]//2 + crop_size_y//2 - 1)
            axes[i,j].imshow(onp.abs(current_image[x1:x2, y1:y2, frame_to_show]), cmap='bone')
            if i == 0:
                axes[i,j].set_title(array_names[j])
            axes[i,j].axis('off')  # Turn off axis

def plot_traj(traj, spokes_per_bin_list, vplot = 2, hplot = 4):
    # inputs
    # traj: (spokes, readout, 2)
    # spokes_per_bin_list: (bins)
    fig, axes = plt.subplots(vplot, hplot, figsize=(hplot*5, vplot*5))
    ns = 0
    for hindex in range(hplot):
        for vindex in range(vplot):
            n_bin_index = vindex + hindex*vplot
            spokes_per_bin = spokes_per_bin_list[n_bin_index]
            for _ in range(spokes_per_bin):
                axes[vindex,hindex].scatter(traj[ns,:,0], traj[ns,:,1])
                ns += 1
            axes[vindex,hindex].set_title("Bin " + str(n_bin_index+1), fontsize=18)
            axes[vindex,hindex].set_xlabel("kx")
            axes[vindex,hindex].set_ylabel("ky")

def plot_kdata(kdata, spokes_per_bin_list, vplot = 2, hplot = 4):
    # inputs
    # kdata: (readout, spokes, coils)
    # spokes_per_bin_list: (bins)
    fig, axes = plt.subplots(vplot, hplot, figsize=(hplot*5, vplot*5))
    ns = 0
    for hindex in range(hplot):
        for vindex in range(vplot):
            n_bin_index = vindex + hindex*vplot
            spokes_per_bin = spokes_per_bin_list[n_bin_index]
            for _ in range(spokes_per_bin):
                axes[vindex,hindex].plot(kdata[:,ns,n_bin_index])
                ns += 1
            axes[vindex,hindex].set_title("Bin " + str(n_bin_index+1), fontsize=18)
            axes[vindex,hindex].set_xlabel("kx")
            axes[vindex,hindex].set_ylabel("ky")

def plot_csm(csm, spokes_per_bin_list, vplot = 2, hplot = 4):
    # inputs
    # csm: (coil, readout, readout)
    # spokes_per_bin_list: (bins)
    fig, axes = plt.subplots(vplot, hplot, figsize=(hplot*5, vplot*5))
    for hindex in range(hplot):
        for vindex in range(vplot):
            n_coil_index = vindex + hindex*vplot
            axes[vindex,hindex].imshow( onp.abs( csm[n_coil_index,:,:] ) )
            axes[vindex,hindex].set_title("Coil " + str(n_coil_index+1), fontsize=18)
            axes[vindex,hindex].axis('off')

from inrmri.utils import is_inside_of_radial_lim, meshgrid_from_subdiv_autolims # total_variation_batch_complex, save_matrix_and_dict_in_zpy, load_matrix_and_dict_from_zpy    
from inrmri.utils import denoise_loss_batch
from jax.lax import map as laxmap 
from inrmri.new_radon import ForwardRadonOperator, get_weight_freqs
from inrmri.basic_nn import weighted_loss 


def pad_axis_to_length(arr, axis, target_length, pad_value=0):
    """
    Pads or crops a specific axis of `arr` to `target_length`.
    
    Parameters:
    - arr: np.ndarray (any dtype)
    - axis: int, the axis to pad/crop
    - target_length: int, desired size along that axis
    - pad_value: scalar (e.g., 0, 0.0, 0+0j)

    Returns:
    - arr_out: np.ndarray with modified shape
    """
    current_length = arr.shape[axis]

    if current_length == target_length:
        return arr
    elif current_length < target_length:
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (0, target_length - current_length)
        return np.pad(arr, pad_width, mode='constant', constant_values=pad_value)
    else:
        # Crop if longer
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(0, target_length)
        return arr[tuple(slices)]


def get_center(arr):
    h, w, _ = arr.shape
    h_start = h // 4
    h_end   = 3 * h // 4
    w_start = w // 4
    w_end   = 3 * w // 4
    return arr[h_start:h_end, w_start:w_end, :]

def safe_normalize(arr, epsilon=1e-8):
    abs_arr = np.abs(arr)
    max_val = np.max(abs_arr)
    return abs_arr / (max_val + epsilon)

def get_varying_keys(combinations):
    """
    Determine which keys vary across all configurations.
    """
    all_keys = combinations[0].keys()
    varying_keys = []
    for key in all_keys:
        values = {config[key] for config in combinations}
        if len(values) > 1:
            varying_keys.append(key)
    return varying_keys

def config_to_foldername(config, varying_keys, maxlen=100):
    """
    Generate folder name using only varying parameters.
    """
    parts = []
    for key in varying_keys:
        val = config[key]
        val_str = str(val).replace('/', '_').replace(' ', '')
        parts.append(f"{key}={val_str}")
    
    foldername = "__".join(parts)
    return foldername[:maxlen]  # Optional truncation

def plot_curves(log_list, exp_names, variable_name, save_path='plot.png', width=12, height=4, dpi=300, xlim=None):
    plt.figure(figsize=(width, height), dpi=dpi)
    ymin, ymax = np.inf, -np.inf

    for i in range(len(log_list)):
        results = log_list[i]
        curve = np.array(results[variable_name])
        iterations = np.array(results['it'])

        if xlim is not None:
            mask = (iterations >= xlim[0]) & (iterations <= xlim[1])
            visible_curve = curve[mask]
        else:
            visible_curve = curve

        if visible_curve.size > 0:
            ymin = min(ymin, np.nanmin(visible_curve))
            ymax = max(ymax, np.nanmax(visible_curve))

        plt.plot(iterations, curve, color='C'+str(i), label=exp_names[i])

    plt.legend()
    plt.xlabel('iteration')

    if xlim is not None:
        plt.xlim(xlim)
    if ymin < ymax:  # To avoid error if no points are visible
        plt.ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_curves_and_mins(log_list, exp_names, variable_name, save_path='plot.png', width=12, height=4, dpi=300, xlim=None):
    plt.figure(figsize=(width, height), dpi=dpi)
    ymin, ymax = np.inf, -np.inf

    for i in range(len(log_list)):
        results = log_list[i]
        curve = np.array(results[variable_name])
        iterations = np.array(results['it'])

        if xlim is not None:
            mask = (iterations >= xlim[0]) & (iterations <= xlim[1])
            visible_curve = curve[mask]
        else:
            visible_curve = curve

        if visible_curve.size > 0:
            ymin = min(ymin, np.nanmin(visible_curve))
            ymax = max(ymax, np.nanmax(visible_curve))

        plt.plot(iterations, curve, color='C'+str(i), label=exp_names[i])
        plt.axvline(iterations[np.nanargmin(curve)], linestyle='--', color='C'+str(i), label='min ' + exp_names[i])

    plt.legend()
    plt.xlabel('iteration')

    if xlim is not None:
        plt.xlim(xlim)
    if ymin < ymax:
        plt.ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

def multiple_images_visualization(images, names, frame=0, save_path='best_images.png', dpi=300, saturation=0.3):
    num_images = len(images)

    fig, axes = plt.subplots(1, num_images, figsize=(2*num_images, 2), dpi=dpi)
    if num_images == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, names):
        img_frame = img[:,:,frame]
        ax.imshow(img_frame / np.max(img_frame), cmap='bone', vmax=saturation)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

def get_predim_direct(net, hollow_mask, key, h_params, final_param):
    predim = laxmap(lambda t: net.train_forward_pass(final_param, key, t[None])[0][0], onp.arange(h_params['NFRAMES'])) # TO DO make lax version, vectorized evaluation takes too much memory 
    predim = onp.moveaxis(predim, 0, -1)
    predim = post_processing(predim, hollow_mask, h_params)
    predim = get_center(predim)
    predim = safe_normalize(predim)
    return predim

def get_predim(net, results, hollow_mask, key, h_params, eval_iteration):
    final_param = results['param_history'][f'param-{eval_iteration}']
    predim = laxmap(lambda t: net.train_forward_pass(final_param, key, t[None])[0][0], onp.arange(h_params['NFRAMES'])) # TO DO make lax version, vectorized evaluation takes too much memory 
    predim = onp.moveaxis(predim, 0, -1)
    predim = post_processing(predim, hollow_mask, h_params)
    predim = get_center(predim)
    predim = safe_normalize(predim)
    return predim

def get_predim_direct_ms(net, hollow_mask, key, h_params, slice_idx, final_param):
    predim = laxmap(lambda t: net.train_forward_pass(final_param, key, t[None], slice_idx)[0][0], np.arange(30)) # TODO make lax version, vectorized evaluation takes too much memory 
    predim = np.moveaxis(predim, 0, -1)
    predim = post_processing(predim, hollow_mask, h_params)
    predim = get_center(predim)
    predim = safe_normalize(predim)
    return predim

def post_processing(im, hollow_mask, h_params): # im.shape (px,py,nframes)
    grid = meshgrid_from_subdiv_autolims(  [h_params['N'], h_params['N']]  )
    is_inside = is_inside_of_radial_lim(grid, 1.)
    masks = (1 - hollow_mask) * is_inside
    return im * masks[:,:,None]

def loss_detailed(net, params, h_params, radon_operator, X, Y, key): 
    # precomputations
    WEIGHT_FREQS = get_weight_freqs(  h_params['N'], h_params['str_filter']  )
    if h_params['sqrt_filter']:
        WEIGHT_FREQS = onp.sqrt(  WEIGHT_FREQS  )
    grid = meshgrid_from_subdiv_autolims(  [h_params['N'], h_params['N']]  )
    alphas, times = X[:,0], X[:,1]
    t_idx = onp.int32(times * h_params['NFRAMES'])
    ims, update = net.train_forward_pass(params, key, t_idx) # (frames, px, py)    
    spoke_radon_kspace = radon_operator.radon_transform(ims, alphas) # (batch-frame-alpha, ncoils, N)
    Y = Y[...,0] # (frames, cmap, nx)
    loss_value = weighted_loss(spoke_radon_kspace, Y, (1. + WEIGHT_FREQS)[None,None,:])
    denoise_val  = denoise_loss_batch(onp.moveaxis(ims, 0, -1), h_params['denoise_type'])
    return loss_value, denoise_val 

def get_loss_lists(net, h_params, radon_operator, results, X_data, Y_data, key, eval_iteration):
    final_param = results['param_history'][f'param-{eval_iteration}']
    consistency_loss_list = []
    tv_loss_list = []
    for spoke in range(X_data.shape[0]):
        X_data_s = X_data[[spoke]]
        Y_data_s = Y_data[[spoke]]
        consistency_loss, tv_loss = loss_detailed(net, final_param, h_params, radon_operator, X_data_s, Y_data_s, key)
        consistency_loss_list.append( consistency_loss )
        tv_loss_list.append( tv_loss )
    return consistency_loss_list, tv_loss_list

def plot_spokes_and_hb(spokes_list, hb_list):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(spokes_list)
    axes[0].set_title("Spockes Bins")
    axes[1].plot(hb_list)
    axes[1].set_title("Spockes HB")
    plt.show()

def plot_angles_and_times_from_xdata(X_data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(X_data[:,0])
    axes[0].set_title("Angles")
    axes[1].plot(X_data[:,1])
    axes[1].set_title("Times")
    plt.show()


def circular_distance_matrix(n):
    matrix = onp.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            direct_distance = abs(i - j)
            circular_distance = n - direct_distance
            matrix[i, j] = min(direct_distance, circular_distance)
    return matrix

def get_exp_weight_descrite(a, b, n):
    dist_matrix = circular_distance_matrix(n)
    dist_matrix_w = (a * onp.exp(- onp.abs(b * dist_matrix) )).astype(int)
    return dist_matrix_w

def soft_binning_descrite(times, kdata_ordered, ktraj_ordered, a=8, b=1.0):
    print(kdata_ordered.shape, ktraj_ordered.shape)
    # statistics
    unique_values, counts = onp.unique(times, return_counts=True)
    n_frames = len(unique_values)
    samples_matrix = get_exp_weight_descrite(a, b, n_frames)
    max_num_spokes = onp.max(counts) + onp.sum(samples_matrix[0]) - a
    # outputs
    ktraj_output = onp.random.random((n_frames, max_num_spokes, ktraj_ordered.shape[1], ktraj_ordered.shape[2]))
    kdata_output = onp.random.random((kdata_ordered.shape[2], n_frames, max_num_spokes, kdata_ordered.shape[0])) + 1j*onp.random.random((kdata_ordered.shape[2], n_frames, max_num_spokes, kdata_ordered.shape[0]))
    times_output = []
    # soft binning
    for i in range(len(unique_values)):
        unique_value = unique_values[i]
        times_output += [unique_value] * max_num_spokes
        index_list = list(onp.where(times == unique_value)[0])
        for j in range(len(unique_values)):
            required_samples = samples_matrix[i, j]
            if (i!=j) and required_samples>0:
                unique_value_j = unique_values[j]
                index_list_j = list(onp.where(times == unique_value_j)[0])
                index_samples_j = random.sample(index_list_j, required_samples)
                index_list = index_list + index_samples_j
        # balance
        needed_spokes = max_num_spokes - len(index_list)
        index_repited_spokes = list(onp.random.choice(index_list, size=needed_spokes, replace=True))
        index_list = index_list + index_repited_spokes
        # save the data
        ktraj_aux = ktraj_ordered[index_list, :, :]
        kdata_aux = kdata_ordered[:, index_list, :]
        # k traj dimensions
        ktraj_aux = onp.expand_dims(ktraj_aux, axis=0)
        # k data dimensions
        kdata_aux = onp.transpose(kdata_aux, (2, 1, 0))
        ktraj_output[i,:,:,:] = ktraj_aux
        kdata_output[:,i,:,:] = kdata_aux
    return ktraj_output, kdata_output, times_output


def read_ref_dataset(path_dataset):
    data               = onp.load(path_dataset)
    recon_grasp        = data["recon_grasp"]
    recon_sense        = data["recon_sense"]
    recon_fs           = data["recon_fs"]
    time_grasp         = data["time_grasp"]
    time_sense         = data["time_sense"] 
    return recon_fs, recon_grasp, recon_sense, time_grasp, time_sense


import nibabel as nib
from scipy.ndimage import binary_fill_holes
def read_segmentation(path2file, fill=True):
    nii_file = nib.load(path2file)
    segmentation = nii_file.get_fdata()
    binary_segmentation = (segmentation > 0).astype(onp.uint8)

    if fill:
        area_list = []
        filled_segmentation = onp.copy(binary_segmentation)
        for frame_idx in range(binary_segmentation.shape[-1]):
            filled_frame = binary_fill_holes(binary_segmentation[:, :, frame_idx])
            filled_segmentation[:, :, frame_idx] = filled_frame
            area_list.append(int(onp.sum(filled_frame)))
        return binary_segmentation, filled_segmentation, area_list

    return binary_segmentation

def get_info_volunteer(dataset, name, slice):
    dict_plot = {
        'DATA_1.5T': {
            'GC': {
                1: {'crop_ns': [35,50,55,30], 'saturation': 0.25},
                2: {'crop_ns': [35,50,55,30], 'saturation': 0.25}, # done
                3: {'crop_ns': [32,57,55,27], 'saturation': 0.3}, # done
                4: {'crop_ns': [32,47,57,27], 'saturation': 0.4}, # done
                5: {'crop_ns': [25,55,50,30], 'saturation': 0.5}, # done
                6: {'crop_ns': [23,55,48,30], 'saturation': 0.5}, # done
                7: {'crop_ns': [25,55,50,30], 'saturation': 0.5},
                8: {'crop_ns': [25,55,50,30], 'saturation': 0.5},
            },
            'CM': {
                1: {'crop_ns': [32,43,30,40], 'saturation': 0.4},
                2: {'crop_ns': [32,43,30,40], 'saturation': 0.4},
                3: {'crop_ns': [32,43,30,40], 'saturation': 0.4}, # done
                4: {'crop_ns': [35,40,35,40], 'saturation': 0.4}, # done
                5: {'crop_ns': [35,40,35,40], 'saturation': 0.4}, # done
                6: {'crop_ns': [35,40,40,34], 'saturation': 0.4}, # done
                7: {'crop_ns': [35,40,35,40], 'saturation': 0.4}, # done
                8: {'crop_ns': [35,40,35,40], 'saturation': 0.4},
            },
            'DW': {
                1: {'crop_ns': [28,45,50,25], 'saturation': 0.6},
                2: {'crop_ns': [28,45,50,25], 'saturation': 0.6}, # done
                3: {'crop_ns': [28,45,50,25], 'saturation': 0.6}, # done
                4: {'crop_ns': [28,45,50,25], 'saturation': 0.6}, # done
                5: {'crop_ns': [30,45,50,25], 'saturation': 0.6}, # done
                6: {'crop_ns': [30,45,50,25], 'saturation': 0.6},
                7: {'crop_ns': [30,45,50,25], 'saturation': 0.6}, 
                8: {'crop_ns': [30,45,50,25], 'saturation': 0.6},
            },
        },
        'DATA_0.55T': {
            'FB': {
                1: {'crop_ns': [15,55,35,40], 'saturation': 0.2}, # done
                2: {'crop_ns': [15,55,35,40], 'saturation': 0.2}, # done
                3: {'crop_ns': [15,55,35,40], 'saturation': 0.2}, # done
                4: {'crop_ns': [15,55,35,40], 'saturation': 0.2}, # done
                5: {'crop_ns': [15,55,40,40], 'saturation': 0.2}, # done
                6: {'crop_ns': [15,55,40,40], 'saturation': 0.2},
                7: {'crop_ns': [15,55,40,40], 'saturation': 0.2},
                8: {'crop_ns': [15,55,40,40], 'saturation': 0.2},
            },
            'MP': {
                1: {'crop_ns': [40,35,35,40], 'saturation': 0.25}, # done
                2: {'crop_ns': [40,35,35,40], 'saturation': 0.25}, # done
                3: {'crop_ns': [40,35,35,40], 'saturation': 0.3}, # done
                4: {'crop_ns': [40,35,35,40], 'saturation': 0.3}, # done
                5: {'crop_ns': [40,40,40,40], 'saturation': 0.3}, # done
                6: {'crop_ns': [40,40,40,40], 'saturation': 0.3},
                7: {'crop_ns': [40,40,40,40], 'saturation': 0.25},
                8: {'crop_ns': [40,40,40,40], 'saturation': 0.3},
            },
            'FH': {
                1: {'crop_ns': [32,40,35,35], 'saturation': 0.3}, 
                2: {'crop_ns': [32,40,35,35], 'saturation': 0.3}, # done
                3: {'crop_ns': [32,40,33,38], 'saturation': 0.3}, # done
                4: {'crop_ns': [32,40,33,38], 'saturation': 0.3}, # done
                5: {'crop_ns': [32,40,33,38], 'saturation': 0.3}, # done
                6: {'crop_ns': [32,40,33,38], 'saturation': 0.3}, # done
                7: {'crop_ns': [32,40,33,38], 'saturation': 0.3},
                8: {'crop_ns': [32,40,33,38], 'saturation': 0.3},
            }, 
            'TC': {
                1: {'crop_ns': [32,42,48,37], 'saturation': 0.2},
                2: {'crop_ns': [40,40,45,40], 'saturation': 0.2}, # done
                3: {'crop_ns': [40,40,45,40], 'saturation': 0.2}, # done
                4: {'crop_ns': [40,40,45,40], 'saturation': 0.2}, # done
                5: {'crop_ns': [40,45,45,40], 'saturation': 0.2}, # done
                6: {'crop_ns': [40,45,45,40], 'saturation': 0.25}, # done
                7: {'crop_ns': [40,45,45,40], 'saturation': 0.25},
                8: {'crop_ns': [40,45,45,40], 'saturation': 0.25},
            }, 
            'RF': {
                1: {'crop_ns': [38,42,45,40], 'saturation': 0.2},
                2: {'crop_ns': [38,42,45,40], 'saturation': 0.2}, # done
                3: {'crop_ns': [38,42,45,40], 'saturation': 0.2}, # done
                4: {'crop_ns': [38,42,45,40], 'saturation': 0.25}, # done
                5: {'crop_ns': [38,42,45,40], 'saturation': 0.25}, # done
                6: {'crop_ns': [38,42,45,40], 'saturation': 0.25}, # done
                7: {'crop_ns': [38,42,45,40], 'saturation': 0.25},
                8: {'crop_ns': [38,42,45,40], 'saturation': 0.25},
            }, 
            'JB': {
                1: {'crop_ns': [40,38,48,30], 'saturation': 0.4},
                2: {'crop_ns': [40,38,48,30], 'saturation': 0.4}, # done
                3: {'crop_ns': [40,38,48,30], 'saturation': 0.5}, # done
                4: {'crop_ns': [40,38,48,30], 'saturation': 0.5}, # done
                5: {'crop_ns': [40,38,48,30], 'saturation': 0.5}, # done
                6: {'crop_ns': [40,38,48,30], 'saturation': 0.5}, # done
                7: {'crop_ns': [40,38,48,30], 'saturation': 0.5},
                8: {'crop_ns': [40,38,48,30], 'saturation': 0.5},
            }, 
            'FE': {
                1: {'crop_ns': [40,40,42,38], 'saturation': 0.25},
                2: {'crop_ns': [40,40,42,38], 'saturation': 0.25}, # done
                3: {'crop_ns': [40,40,42,38], 'saturation': 0.25}, # done
                4: {'crop_ns': [38,42,38,36], 'saturation': 0.25}, # done
                5: {'crop_ns': [36,42,38,36], 'saturation': 0.35}, # done 
                6: {'crop_ns': [36,42,38,36], 'saturation': 0.35}, # done 
                7: {'crop_ns': [36,42,38,36], 'saturation': 0.4}, # done
                8: {'crop_ns': [36,42,38,36], 'saturation': 0.5}, # done
            }, 
            'DB': {
                1: {'crop_ns': [34,40,45,30], 'saturation': 0.25},
                2: {'crop_ns': [34,40,45,30], 'saturation': 0.25}, # done
                3: {'crop_ns': [34,40,45,30], 'saturation': 0.25}, # done
                4: {'crop_ns': [34,40,45,30], 'saturation': 0.25}, # done
                5: {'crop_ns': [34,40,45,30], 'saturation': 0.25}, # done
                6: {'crop_ns': [34,42,45,30], 'saturation': 0.25}, # done
                7: {'crop_ns': [34,42,45,30], 'saturation': 0.25},
                8: {'crop_ns': [34,42,45,30], 'saturation': 0.25},
            }, 
            'FP': {
                1: {'crop_ns': [35,42,45,30], 'saturation': 0.35},
                2: {'crop_ns': [35,42,45,30], 'saturation': 0.35}, # done
                3: {'crop_ns': [35,42,45,30], 'saturation': 0.35}, # done
                4: {'crop_ns': [35,42,45,30], 'saturation': 0.35}, # done
                5: {'crop_ns': [35,42,45,30], 'saturation': 0.35},
                6: {'crop_ns': [35,42,45,30], 'saturation': 0.35},
                7: {'crop_ns': [35,42,45,30], 'saturation': 0.35}, # done
                8: {'crop_ns': [35,42,45,30], 'saturation': 0.35},
            }, 
            'AA': {
                1: {'crop_ns': [35,42,45,30], 'saturation': 0.3},
                2: {'crop_ns': [35,42,45,30], 'saturation': 0.3}, # done
                3: {'crop_ns': [35,42,45,30], 'saturation': 0.3}, # done
                4: {'crop_ns': [35,42,45,30], 'saturation': 0.3}, # done
                5: {'crop_ns': [35,42,45,30], 'saturation': 0.3}, # done
                6: {'crop_ns': [35,42,45,30], 'saturation': 0.3}, # done
                7: {'crop_ns': [35,42,45,30], 'saturation': 0.3},
                8: {'crop_ns': [35,42,45,30], 'saturation': 0.3},
            }, 
            'DP': {
                1: {'crop_ns': [25,60,40,35], 'saturation': 0.2},
                2: {'crop_ns': [25,60,40,35], 'saturation': 0.2}, # done
                3: {'crop_ns': [25,55,40,35], 'saturation': 0.2}, # done
                4: {'crop_ns': [25,55,40,35], 'saturation': 0.2}, # done
                5: {'crop_ns': [25,55,40,35], 'saturation': 0.2}, # done
                6: {'crop_ns': [25,55,40,35], 'saturation': 0.2}, # done
                7: {'crop_ns': [25,55,40,40], 'saturation': 0.2}, # done
                8: {'crop_ns': [25,55,40,40], 'saturation': 0.2},
            }, 
        }
    }
    dict_trans = {
        'DATA_1.5T': {
            'GC': {
                'flip_h': False,
                'flip_v': False,
                'rot': 0,
            },
            'CM': {
                'flip_h': False,
                'flip_v': False,
                'rot': 0,
            },
            'DW': {
                'flip_h': False,
                'flip_v': False,
                'rot': 0,
            },
        },
        'DATA_0.55T': {
            'FB': {
                'flip_h': False,
                'flip_v': False,
                'rot': 0,
            },
            'MP': {
                'flip_h': True,
                'flip_v': True,
                'rot': 0,
            },
            'FH': {
                'flip_h': True,
                'flip_v': True,
                'rot': 0,
            }, 
            'TC': {
                'flip_h': True,
                'flip_v': True,
                'rot': 0,
            }, 
            'RF': {
                'flip_h': True,
                'flip_v': True,
                'rot': 0,
            }, 
            'JB': {
                'flip_h': True,
                'flip_v': True,
                'rot': 0,
            }, 
            'FE': {
                'flip_h': True,
                'flip_v': False,
                'rot': 0,
            }, 
            'DB': {
                'flip_h': True,
                'flip_v': False,
                'rot': 0,
            },
            'FP': {
                'flip_h': False,
                'flip_v': False,
                'rot': 0,
            },
            'AA': {
                'flip_h': True,
                'flip_v': False,
                'rot': 0,
            },
            'DP': {
                'flip_h': False,
                'flip_v': False,
                'rot': 0,
            },
        }
    }
    dict_EF_frames = {
        'DATA_0.55T': {
            'FB': {
                'EDV_gt': 0,
                'ESV_gt': 8,
                'EDV': 0,
                'ESV': 8,
            },
            'MP': {
                'EDV_gt': 0,
                'ESV_gt': 10,
                'EDV': 0,
                'ESV': 10,
            },
            'FH': {
                'EDV_gt': 0,
                'ESV_gt': 10,
                'EDV': 0,
                'ESV': 10,
            }, 
            'TC': {
                'EDV_gt': 0,
                'ESV_gt': 10,
                'EDV': 0,
                'ESV': 10,
            }, 
            'RF': {
                'EDV_gt': 0,
                'ESV_gt': 11,
                'EDV': 0,
                'ESV': 11,
            }, 
            'JB': {
                'EDV_gt': 0,
                'ESV_gt': 9,
                'EDV': 0,
                'ESV': 9,
            }, 
            'FE': {
                'EDV_gt': 0,
                'ESV_gt': 11,
                'EDV': 0,
                'ESV': 11,
            }, 
            'DB': {
                'EDV_gt': 0,
                'ESV_gt': 12,
                'EDV': 0,
                'ESV': 12,
            },
            'FP': {
                'EDV_gt': 0,
                'ESV_gt': 12,
                'EDV': 0,
                'ESV': 12,
            },
            'AA': {
                'EDV_gt': 0,
                'ESV_gt': 11,
                'EDV': 0,
                'ESV': 11,
            },
            'DP': {
                'EDV_gt': 0,
                'ESV_gt': 9,
                'EDV': 0,
                'ESV': 9,
            },
        }
    }
    gt_trans = {
        'DATA_0.55T': {
            'FB': {
                'flip_h': False,
                'flip_v': False,
                'rot': 180,
            },
            'FP': {
                'flip_h': False,
                'flip_v': False,
                'rot': 90,
            },
            'DP': {
                'flip_h': True,
                'flip_v': True,
                'rot': 0,
            },
        }
    }

    output = {}
    output['crop_ns']      =  dict_plot[dataset][name][slice]['crop_ns']
    output['saturation']   =  dict_plot[dataset][name][slice]['saturation']
    output['trans']        =  dict_trans[dataset][name]
    output['EF_frames']    =  dict_EF_frames[dataset][name]

    if dataset in gt_trans and name in gt_trans[dataset]:
        output['trans_gt'] = gt_trans[dataset][name]
    else:
        output['trans_gt'] = dict_trans[dataset][name]


    return output
