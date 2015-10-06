import lasagne

def norm_constraint(updates, final_layer):
    # Add norm constraints
    all_layers = lasagne.layers.get_all_layers(final_layer)
    for l in all_layers:
        if isinstance(l, lasagne.layers.Conv2DLayer):
            updates[l.W] = lasagne.updates.norm_constraint(updates[l.W], 
                max_norm=2.0)
        if isinstance(l, lasagne.layers.DenseLayer):
            updates[l.W] = lasagne.updates.norm_constraint(updates[l.W], 
                max_norm=0.5)
    return updates