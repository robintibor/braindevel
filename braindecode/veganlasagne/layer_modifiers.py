import lasagne
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
        transformed = transform_to_normal_net(self.layers[-1])
        return lasagne.layers.get_all_layers(transformed)
    
    
        