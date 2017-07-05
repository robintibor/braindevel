from lasagne.layers.base import Layer
import theano.tensor as T
class ClipLayer(Layer):
    def __init__(self, incoming, min_val, max_val, **kwargs):
        super(ClipLayer, self).__init__(incoming, **kwargs)
        self.min_val = min_val
        self.max_val = max_val

    def get_output_for(self, input, **kwargs):
        return T.clip(input, self.min_val, self.max_val)

class EpsLayer(Layer):
    """ When value smaller than min_val, add eps
    When value larger than max_val subtract eps."""
    def __init__(self, incoming, min_val, max_val, eps, **kwargs):
        super(EpsLayer, self).__init__(incoming, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.eps = eps

    def get_output_for(self, input, **kwargs):
        output = input * T.ge(input, self.min_val) * T.le(input, self.max_val)
        output += (T.lt(input, self.min_val) * (input + self.eps))
        output += (T.gt(input, self.max_val) * (input - self.eps))
        return output
