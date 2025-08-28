# %% 

from scipy.io import loadmat 
import numpy as onp
import matplotlib.pyplot as plt  

# %%

folder = '/mnt/storage/RafaData/Fantoma/'

data = loadmat(folder + '2D_radial_cine.mat')

# %%

kdata = data['kdata']
RO, NSPOKES, NCOILS = kdata.shape

ktraj = onp.reshape(data['ktraj'], (NSPOKES, RO, 3))
csm = data['csm']

kdata.shape, ktraj.shape, csm.shape
# %%
from src.bart import bart_acquisition_from_arrays

def process_data(ktraj, kdata, folder, name, bins=30):
    """
    # Argumentos: 
    - trajectory: array(float), shape (total_spokes, readout, 3)
    - kdata: array(complex), shape (RO, total_spokes, ncoils)
    """
    assert ktraj.ndim == 3 
    assert kdata.ndim == 3 
    assert ktraj.shape[0] == kdata.shape[1] # same total spokes 
    assert ktraj.shape[1] == kdata.shape[0] # same RO 

    NSPOKES = ktraj.shape[0]
    RO = ktraj.shape[1]
    used_spokes = bins * (NSPOKES // bins)
    new_traj = onp.reshape(ktraj[:used_spokes], (bins, NSPOKES // bins) + ktraj.shape[1:])

    new_data = onp.reshape(kdata[:,:used_spokes], (RO, bins, NSPOKES//bins, kdata.shape[-1])) # (ro, bins, spoke_per_frame, coils)
    new_data = onp.moveaxis(new_data, -1, 0) #  (coils, ro, bins, spoke_per_frame)
    new_data = onp.moveaxis(new_data, 1, -1) #  (coils, bins, spoke_per_frame, ro)

    bac = bart_acquisition_from_arrays(new_traj, new_data, folder, name)
    return bac 

bac = process_data(ktraj, kdata, folder, 'b01', bins=1)

from src.radial_acquisitions import create_radial_acq_from_bart 

ra = create_radial_acq_from_bart(bac)
ra.generate_dataset()
# %%

reco = bac.calculate_inufft_reco('d256', (256,256,1))
reco = onp.squeeze(reco)
reco.shape
# %%
selected_coil = 6 
plt.imshow(onp.abs(reco[:,:,selected_coil]))

# %%
im = onp.sqrt(onp.sum(onp.abs(reco ** 2), axis=(2,)))
# im.shape
plt.imshow(onp.abs(im), cmap='bone')
