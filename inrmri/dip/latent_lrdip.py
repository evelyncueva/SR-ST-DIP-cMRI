import flax.linen as nn
from typing import Tuple, Sequence
from inrmri.dip.tddip import MapNet, Decoder 

# ---------------------------------------------------------------------- # 
# Experimental Architecture
# Tiene estructura LR pero en el ruido, no en la salida 
# ---------------------------------------------------------------------- # 

class LatentLowRankTDDIP(nn.Module):
    mapnet_layers:Sequence[int]
    latent_features:int 
    latent_shape:Tuple[int,int]
    decoder_features:int = 64 
    decoder_levels:int = 3

    @nn.compact 
    def __call__(self, encoded_t, latent_x, training:bool):
        """
        - `encoded_t`: shape (batch, t_features)
        - `latent_x`: shape (latent_shape[0], latent_shape[1], latent_features)
        """
        assert latent_x.ndim == 3 
        assert latent_x.shape[-1] == self.latent_features
        assert latent_x.shape[:2] == self.latent_shape

        mapnet = MapNet(self.mapnet_layers, (self.latent_features, 1))
        latent_t = mapnet(encoded_t) # tbatch, self.latent_features
        latent_t = latent_t[:, None, None, :] #  tbatch, 1, 1, self.latent_features
        latent_x = latent_x[None,:,:,:] # (1, latent_shape[0], latent_shape[1], self.latent_features)
        latent = latent_t * latent_x # sumo o no la ultima dim? mmm....
        print(latent.shape)
        decoder = Decoder(self.decoder_features, levels=self.decoder_levels)
        im = decoder(latent, training)
        return im 
 