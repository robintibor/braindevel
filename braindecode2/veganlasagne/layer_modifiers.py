import lasagne
import numpy as np
from braindecode.veganlasagne.layers import transform_to_normal_net

def get_layers(layers_or_layer_obj):
    """Either return layers if already a list or call get_layers function
    of layer object."""
    if hasattr(layers_or_layer_obj, '__len__'):
        return layers_or_layer_obj
    else:
        return layers_or_layer_obj.get_layers()
    
class JustReturn(object):
    def __init__(self, layers):
        self.layers = layers
        
    def get_layers(self):
        return get_layers(self.layers)
    
class TransformToNormalNet(object):
    def __init__(self, layers):
        self.layers = layers
        
    def get_layers(self):
        layers = get_layers(self.layers)
        final_layer = layers[-1]
        assert len(np.setdiff1d(layers, 
            lasagne.layers.get_all_layers(final_layer))) == 0, ("All layers "
            "should be used, unused {:s}".format(str(np.setdiff1d(layers, 
            lasagne.layers.get_all_layers(final_layer)))))
        transformed = transform_to_normal_net(final_layer)
        return lasagne.layers.get_all_layers(transformed)
    
    
        