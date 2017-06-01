def get_templates():
    return {}


def get_grid_param_list():
    return [{}]

def sample_config_params(rng, params):
    return params

def get_subject_config(subject_id):
    if subject_id == 'sama':
        all_end_marker_vals = [20, 21, 24, 28, 30]
        train_folders = ['data/robot-hall/SaMa/SaMaNBD1R01-5/']
        test_folders = ['data/robot-hall/SaMa/SaMaNBD1R06-7/']
    elif subject_id == 'hawe':
        all_end_marker_vals = [21, 24, 28, 30]
        train_folders = ['data/robot-hall/HaWe/HaWeNBD3/',
                         'data/robot-hall/HaWe/HaWeNBD4/',
                         'data/robot-hall/HaWe/HaWeNBD5/']
        test_folders = ['data/robot-hall/HaWe/HaWeNBD7/']
    elif subject_id == 'lufi':
        all_end_marker_vals = [20, 21, 22, 23, 24, 28, 30]
        train_folders = ['data/robot-hall/LuFi/LuFiNBD1R01-9/']
        test_folders = ['data/robot-hall/LuFi/LuFiNBD1R10-11/']
    elif subject_id == 'anla':
        all_end_marker_vals = [20, 21, 22, 23, 24, 28, 30]
        train_folders = ['data/robot-hall/AnLa/AnLaNBD1R01-8/', ]
        test_folders = ['data/robot-hall/AnLa/AnLaNBD1R09-10/', ]
    elif subject_id == 'elkh':
        all_end_marker_vals = [20, 21, 22, 23, 24, 28, 30]
        train_folders = ['data/robot-hall/ElKh/ElKhNBD1R01-03/', ]
        test_folders = ['data/robot-hall/ElKh/ElKhNBD1R04-05/', ]
    else:
        assert False
    return all_end_marker_vals, train_folders, test_folders
