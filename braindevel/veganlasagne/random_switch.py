from lasagne.layers import MergeLayer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng
import theano.ifelse

class RandomSwitchLayer(MergeLayer):
    def __init__(self, incoming, alternative, survival_prob, **kwargs):
        super(RandomSwitchLayer, self).__init__(
            [incoming, alternative], **kwargs)
        self._survival_prob = survival_prob
        # ensure different layers are not using same seeded
        # random generators
        # -> else all layers are always taking same option...
        self._rng = RandomStreams(get_rng().randint(1, 2147462579))
        
    def get_output_shape_for(self, input_shapes):
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]
    
    def get_output_for(self, inputs, deterministic=False, **kwargs):
        normal_out, alternative = inputs
        if deterministic:
            # check if this is ok like this for stochastic depth net
            # it is not same as what they are doing with relu
            return (self._survival_prob * normal_out) + (
                (1 - self._survival_prob) * alternative)
        else:
            # then else order inverted from normal
            # for performance reasons
            # thats also why i am using 1 - survival_prob
            take_alternative = self._rng.binomial(size=(1,),n=1,
                p=1-self._survival_prob)[0]
            chosen_out = theano.ifelse.ifelse(take_alternative, 
                alternative, normal_out)
            return chosen_out