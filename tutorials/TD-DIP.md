# Some considerations to work with TD-DIP 

If you want to work with TD-DIP [1], you can use `demos/tdDIP-reco.py` as starting point. Here are some tips that could be useful

## Data loading 

You'll probably want to replace my pipeline with yours. I've made changes since my original implementation but I have not cleaned the data loading, thats why there´s a lot of extra processing to the `train_X`, `train_Y`. The important thing is that you should have defined the following variables: 

- `train_X` : should be an `array[float]` of shape `(total_spokes, 2)`: 
  - `train_X[:,0]` contains the angle of each spoke, calculated with `calculate_angle` from `inrmri.radon`.
  - `train_X[:,1]` contains the index of the bin associated to each spoke.
- `train_Y` : `array[complex]` of shape `(total_spokes, ncoils, read-out, 1)`. I would keep the `3*train_Y/np.abs(train_Y).max()` normalization since it usually works well for me.
- `NFRAMES` : number of frames or bins
- `csmap` : coils, `array[complex]` with shape `(ncoils, N, N)`
- `hollow_mask` : `array[{0,1}]`, with shape `(N,N)`. it has value 1 in the zones where the coils have no sensibility
- `spclim` : use 0.5 (I needed it in the old implementation of the radon transform and I haven't removed it from  ForwardRadonOperator`)

## Network 

TD-DIP is implemented in the `TimeDependant_DIP_Net` class. More specifically, the implemented version is Helix ($L=3$) and MapNet ($L=169$) (see Table III in [1]). Some considerations

### Latents

If you check the code you'll see that at initialization it creates a field `net.latent` using a `latent_generator` function. This is the Manifold (see [1]). `net.latent` should have shape `(NFRAMES, d)`, with typically `d=2` or `d=3`. You should have a latent for each bin. Latents are the effective inputs to the *MapNet* (see [1]) and are used to add the manifold information to the network. Typical generators are:
- **Circle:** for periodic video, creates a latent with `d=2`. `inrmri.tddip_net_circle_loader` offers a quick loading of TD-DIP with this generator
- **Helix:** for pseudo periodic video, creates a latent with `d=3` and is implemented in `dip.helix_generator`. You can use it to reconstruct more than one cycle, but it doesn't support cycles of different length.

> **Important.** If you want to create the frames [3, 5] for example, you can use the method `train_forward_pass` from `TimeDependant_DIP_Net` with `t_index=np.array([3,5])`. Notice that the actual inputs to the network are the corresponding latents already saved at inicialization. This means that you can't use `train_forward_pass` without modifications to interpolate frames.

### Changing the size of the output of the network

You can do this with the variables `cnn_latent_shape` and `levels` when initializing `TimeDependant_DIP_Net`. 
- `cnn_latent_shape`: the size of the initial image that is used as input to the `Decoder`. I usually use something between (8,8) and (13,13), [1] uses (8,8).
- `levels`: related with (is not exactly) the number of upsamplings of the Decoder.


> **Warning!** If you check `TimeDependant_DIP_Net.train_forward_pass`, you'll see that the image is cropped before being returned with `y = y[...,:nx,:ny, :]`. If you want to change the size of the output of the network, be very carefull! that croping can be source of a silent problem. You could be generating a huge image, say (1000,1000) and not realizing because of that cropping at the end.

## References 

[1]: [Time-Dependent Deep Image Prior for Dynamic MRI](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9442767)