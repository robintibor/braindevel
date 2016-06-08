unclean_sets = ['AnWe', 'JoBe', 'MaGl', 'NaMa', 'OlIl', 'SvMu']

# for paper: left,right,feet,rest
permute_order_i_class = [1,0,3,2]

def map_i_class(i_class):
    return permute_order_i_class[i_class]

    
resorted_class_names = ("Hand (L)", "Hand (R)", "Feet", "Rest")

