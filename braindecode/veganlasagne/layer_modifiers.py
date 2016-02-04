import lasagne
import numpy as np
from braindecode.veganlasagne.layers import transform_to_normal_net

class JustReturn(object):
    def __init__(self, layers):
        self.layers = layers
        
    def get_layers(self):
        return self.layers
    
class TransformToNormalNet(object):
    def __init__(self, layers):
        self.layers = layers
        
    def get_layers(self):
        final_layer = self.layers[-1]
        assert len(np.setdiff1d(self.layers, 
            lasagne.layers.get_all_layers(final_layer))) == 0, ("All layers "
            "should be used, unused {:s}".format(str(np.setdiff1d(self.layers, 
            lasagne.layers.get_all_layers(final_layer)))))
        transformed = transform_to_normal_net(self.layers[-1])
        return lasagne.layers.get_all_layers(transformed)
    
    
        