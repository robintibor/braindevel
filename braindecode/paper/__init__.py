import itertools

unclean_sets = ['AnWe', 'JoBe', 'MaGl', 'NaMa', 'OlIl', 'SvMu']
# for paper: left,right,feet,rest
permute_order_i_class = [1,0,3,2]

def map_i_class(i_class):
    return permute_order_i_class[i_class]
    
resorted_class_names = ("Hand (L)", "Hand (R)", "Feet", "Rest")

def map_i_class_pair(i_class_pair):
    """from new index to old index"""
    n_classes = 4
    original_classes = range(n_classes)
    permuted_classes = [map_i_class(i_class) for i_class in original_classes]
    original_pairs = list(itertools.combinations(original_classes,2))
    # -> [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    permuted_pairs = list(itertools.combinations(permuted_classes,2))
    sorted_permuted_pairs = [tuple(sorted(p)) for p in permuted_pairs]
    # -> [(0, 1), (1, 3), (1, 2), (0, 3), (0, 2), (2, 3)]
    wanted_pair = sorted_permuted_pairs[i_class_pair]
    i_pair_in_original_order = original_pairs.index(wanted_pair)
    
    order_changed = sorted_permuted_pairs[i_class_pair] != permuted_pairs[i_class_pair]
    assert original_pairs[i_pair_in_original_order] == wanted_pair
    return i_pair_in_original_order, wanted_pair, order_changed
 