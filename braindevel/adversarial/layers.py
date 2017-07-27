import numpy as np
from copy import deepcopy
import lasagne
from braindevel.veganlasagne.layers import get_all_paths
from braindevel.veganlasagne.layer_util import set_to_new_input_layer

def get_longest_path(final_layer):
    all_paths = get_all_paths(final_layer)
    path_lens = [len(p) for p in all_paths]
    i_longest = np.argmax(path_lens)
    return all_paths[i_longest]

def create_adversarial_model(final_layer, i_split_layer):
    final_adv = deepcopy(final_layer)
    longest_path_seiz = get_longest_path(final_layer)
    longest_path_adv = get_longest_path(final_adv)
    
    longest_path_adv[i_split_layer+1].input_layer = longest_path_seiz[
        i_split_layer]
    
    # just in case there is a hanging input layer 
    # maybe this is not the full fix to the problem
    # of multiple paths through the final layer network
    # mostly just a hack for now
    in_l_main = [l for l in lasagne.layers.get_all_layers(final_layer)
             if l.__class__.__name__ == 'InputLayer']
    assert len(in_l_main) == 1
    set_to_new_input_layer(final_adv, in_l_main[0])
    # check if everything is correct, layers up to i split layer
    # are shared, later ones not
    longest_path_adv = get_longest_path(final_adv)
    for i_layer in range(i_split_layer+1):
        assert longest_path_adv[i_layer] == longest_path_seiz[i_layer]
    for i_layer in range(i_split_layer+1, len(longest_path_adv)):
        assert longest_path_adv[i_layer] != longest_path_seiz[i_layer]
    return final_adv
