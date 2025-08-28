# ------------------------------------------------------------- # 
# Ejemplo del cálculo de la metrica sharpness
# En este notebook miraremos en detalle el funcionamiento de la
# funcion radial_sharpness de inrmri.sharpness, con el objetivo
# de entender mejor el cálculo de la métrica de sharpness.
# ------------------------------------------------------------- # 

# %%
import numpy as np
from inrmri.image_processor import reduce_crop_abs 
from inrmri.sharpness import (
    radial_sharpness,
    extract_line_at_points,
    get_line_points_from_angle,
    find_percentile_idx, 
    get_distance
)
import matplotlib.pyplot as plt 

im_reference   = np.load(f'/mnt/storage/results/rafadata/cines/DB/tddip-tr/recos/slice_5_8_nbins30_gt_7kit-bin5.npy')
im_reference   = im_reference[:,:,33:66]

vmax           = 0.6 * np.abs(im_reference).max()
selected_frame = 20 
crop_ns        = [40, 42, 30, 35]
reduced_im     = reduce_crop_abs(im_reference[...,selected_frame], crop_ns)

plt.imshow(np.abs(reduced_im), cmap='bone', vmax=vmax)

# %%

center        = np.array([16, 25])  # coordenadas del centro del miocardio
radius_pixels = 15                  # radio en pixeles del perfil 

for angle in [0, np.pi/2, 3 * np.pi / 4, np.pi/4]:
    # crear coordenadas para extraer, tambien se obtiene el indice del centro 
    points, idx_center = get_line_points_from_angle(center, angle, reduced_im.shape)

    # la sharpness se calcula como el promedio del perfil derecho y el izquierdo 
    # calcularemos solo la izquierda 
    points        = points[idx_center-radius_pixels:idx_center,:] # mitad izquierda
    profile       = extract_line_at_points(reduced_im, points)

    # find_percentile_idx funciona en un vector ordenado y entrega el indice más
    # cercano al valor min + (max-min)*p/100, con p=20 y 80
    idx_sorting   = np.argsort(profile)
    idx_sorted_20, p20 = find_percentile_idx(profile[idx_sorting], 20) 
    idx_sorted_80, p80 = find_percentile_idx(profile[idx_sorting], 80) 

    # encontrar los índeces del vector original 
    corrected_idxs = np.arange(profile.shape[0])[idx_sorting]
    idx_20 = corrected_idxs[idx_sorted_20]
    idx_80 = corrected_idxs[idx_sorted_80]

    # get_distance(points) es 1 si los puntos están en horizontal o vertical 
    # pero es mayor cuando los puntos están en diagonal
    d_20_80 = np.abs(idx_20 - idx_80) 
    d       = d_20_80 * get_distance(points)
    print(f"La ")

    plt.figure()
    plt.subplot(211)
    plt.imshow(np.abs(reduced_im), cmap='bone', vmax=vmax)
    plt.plot(points[:,1],
            points[:,0],
            color='red'
            )
    plt.subplot(212)
    plt.plot(profile, '.-', color = 'red', label= 'profile')

    p20_color = 'orange'
    plt.hlines([p20], xmin=0, xmax=14, color=p20_color, label='p20', linestyles=(0,(1,3)))
    plt.vlines([idx_20], ymin=profile.min(), ymax=profile.max(), color=p20_color, label='idx_20', linestyles=(0,(1,1)))

    p80_color = 'blue'
    plt.hlines([p80], xmin=0, xmax=14, color=p80_color, label='p80', linestyles=(0, (3,3,1,3)))
    plt.vlines([idx_80], ymin=profile.min(), ymax=profile.max(), color=p80_color, label='idx_80', linestyles=(0, (3,1,1,1)))

    plt.hlines([profile.min()], xmin=idx_20, xmax=idx_80, color='gray', linestyles='-')
    plt.text(x=(idx_20 + idx_80)/2, y=profile.min() + 0.07 * (profile.max() - profile.min()), s='$d_{20, 80} = $' + f'{d_20_80}', ha='center', va='center', color='gray')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('$d_{20,80} = ' f'{d_20_80}$, sharpness 1/d {1 / d:.3f}')
    plt.tight_layout()

# %%

sharpness_vector = []
selected_frame = 20

sharpness_vector = np.array([radial_sharpness(reduced_im, center, angle, radius_pixels) for angle in np.linspace(0, np.pi/2, 180)])
plt.plot(sharpness_vector)
print("Mean sharpness: ", np.mean(sharpness_vector))

# %%
