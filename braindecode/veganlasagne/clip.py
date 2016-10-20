from lasagne.layers.base import Layer
import theano.tensor as T
class ClipLayer(Layer):
    def __init__(self, incoming, min_val, max_val, **kwargs):
        super(ClipLayer, self).__init__(incoming, **kwargs)
        self.min_val = min_val
        self.max_val = max_val

    def get_output_for(self, input, **kwargs):
        return T.clip(input, self.min_val, self.max_val)
