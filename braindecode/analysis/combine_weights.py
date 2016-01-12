import numpy as np

def combine_temporal_spatial_weights(temporal_weights, spatial_weights):
    # Compute combined the weights. 
    # We have to reverse them with ::-1 to change convolution to cross-correlation


    temporal_weights = temporal_weights[:,:,::-1,::-1]
    spat_filt_weights = spatial_weights[:,:,::-1,::-1]

    combined_weights = np.tensordot(spat_filt_weights, temporal_weights, axes=(1,0))

    combined_weights = combined_weights.squeeze()
    return combined_weights