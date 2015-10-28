import lasagne
from copy import deepcopy
import logging
log = logging.getLogger(__name__)

class MaxNormConstraint():
    def __init__(self, layer_names_to_norms):
        self.layer_names_to_norms = layer_names_to_norms

    def modify(self, updates, final_layer):
        layer_names_to_norms = deepcopy(self.layer_names_to_norms)
        all_layers = lasagne.layers.get_all_layers(final_layer)
        for layer in all_layers:
            if layer.name in layer_names_to_norms:
                norm = layer_names_to_norms.pop(layer.name)
                log.info("Constraining {:s} to norm {:.2f}".format(
                    layer.name, norm))
                updates[layer.W] = lasagne.updates.norm_constraint(
                    updates[layer.W], max_norm=norm)
        assert len(layer_names_to_norms) == 0, ("All layers specified in max "
            "col norm should be specified, nonexisting layers", layer_names_to_norms)
            
        return updates
        