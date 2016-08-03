import lasagne
from copy import deepcopy
import numpy as np
import logging
from _collections import deque
log = logging.getLogger(__name__)

class MaxNormConstraint():
    def __init__(self, layer_names_to_norms):
        self.layer_names_to_norms = layer_names_to_norms

    def modify(self, updates, final_layer):
        layer_names_to_norms = deepcopy(self.layer_names_to_norms)
        all_layers = lasagne.layers.get_all_layers(final_layer)
        normed_layer_names = set()
        for layer in all_layers:
            if layer.name in layer_names_to_norms:
                norm = layer_names_to_norms[layer.name]
                normed_layer_names.add(layer.name)
                log.info("Constraining {:s} to norm {:.2f}".format(
                    layer.name, norm))
                updates[layer.W] = lasagne.updates.norm_constraint(
                    updates[layer.W], max_norm=norm)
        assert np.array_equal(sorted(layer_names_to_norms.keys()),
            sorted(normed_layer_names)), ("All layers specified in max "
            "col norm should be specified, nonexisting layers", 
            np.setdiff1d(layer_names_to_norms.keys(), normed_layer_names))
            
        return updates
    
    
class MaxNormConstraintWithDefaults(object):
    """ Uses max norm constraint of 2.0 on all intermediate layers
    and constraint of 0.5 on final layers (= layers that are not followed
    by any more layers with parameters)."""
    def __init__(self, layer_names_to_norms):
        self.layer_names_to_norms = layer_names_to_norms
        
    def modify(self, updates, final_layer):
        _, layers_to_succs = (
            layer_to_predecessors_and_successors(final_layer))
        
        layer_names_to_norms = deepcopy(self.layer_names_to_norms)
        all_layers = lasagne.layers.get_all_layers(final_layer)
        normed_layer_names = set()
        for layer in all_layers:
            if layer.name in layer_names_to_norms:
                norm = layer_names_to_norms[layer.name]
                normed_layer_names.add(layer.name)
                log.info("Constraining {:s} to norm {:.2f}".format(
                    layer.name, norm))
                updates[layer.W] = lasagne.updates.norm_constraint(
                    updates[layer.W], max_norm=norm)
            elif hasattr(layer, 'W'):
                # check if any successors also have weights...
                successors = layers_to_succs[layer]
                successors_have_weights = np.any([hasattr(l_succ, 'W')
                    for l_succ in successors])
                if successors_have_weights:
                    norm = 2.0
                else:
                    norm = 0.5
                log.info("Constraining {:s} to norm {:.2f}".format(
                    layer.name, norm))
                updates[layer.W] = lasagne.updates.norm_constraint(
                    updates[layer.W], max_norm=norm)
                    
                
        assert np.array_equal(sorted(layer_names_to_norms.keys()),
            sorted(normed_layer_names)), ("All layers specified in max "
            "col norm should be specified, nonexisting layers", 
            np.setdiff1d(layer_names_to_norms.keys(), normed_layer_names))
            
        return updates
        

def layer_to_predecessors_and_successors(final_layer):
    """ Dicts with predecessor and successor layers per layer."""
    layer_to_pred = {}
    layer_to_succ= {}

    layer_to_succ[final_layer] = []


    queue = deque([final_layer])
    seen = set()


    while queue:
        # Peek at the leftmost node in the queue.
        layer = queue.popleft()
        if layer is None:
            # Some node had an input_layer set to `None`. Just ignore it.
            pass
        elif layer not in seen:
            # We haven't seen this node yet, update predecessors and parents
            # for it and its input layers
            seen.add(layer)

            if hasattr(layer, 'input_layers'):
                layer_to_pred[layer] = layer.input_layers
                for parent in layer.input_layers:
                    layer_to_succ[parent] = layer_to_succ[layer] + [layer]
                for offspring in layer_to_succ[layer]:
                    layer_to_pred[offspring] += layer.input_layers
                queue.extendleft(reversed(layer.input_layers))

            elif hasattr(layer, 'input_layer'):
                layer_to_pred[layer] = [layer.input_layer]
                layer_to_succ[layer.input_layer] = layer_to_succ[layer] + [layer]
                for offspring in layer_to_succ[layer]:
                    layer_to_pred[offspring] += [layer.input_layer]
                queue.appendleft(layer.input_layer)
        else:
            # We've been here before: Either we've finished all its incomings,
            # or we've detected a cycle.
            pass
    return layer_to_pred, layer_to_succ
    
        